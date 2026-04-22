# -*-coding:utf-8 -*-
"""
# Author     ：comi (+ mods)
# version    ：python 3.8
# Description：warmup + extra folders (ratio) + periodic train-dice / per-class pruning
"""
import os
import time
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from glob import glob

import configs
from configs import GC
from utils.EarlyStop import EarlyStopping
from utils.Metrics import Metrics
# from utils.MetricsV2 import MetricsV2
from utils.helper import (
    transforms2d, transforms3d, get_model2d, get_model3d, set_init, avgStd, showTime, save_tmp,
    load_model_k_checkpoint, get_gate
)
from utils.logger import logs
from utils.loss import Loss
from utils.lossV2 import LossV2
from utils.noduleSet import noduleSet
import csv


# ========================= 可调参数 =========================

WARMUP_EPOCHS = 10

REFILL_AFTER_PRUNE = True

EXTRA_ROOT = "./fold5"

EXTRA_FRACTIONS = {
    "xiao": 0.17, #1006
    "zhong": 0.2, #395
    "da": 0.1, #199
}

TRAIN_EVAL_INTERVAL = 3

PRUNE_LOW_DICE_THR  = 0.001   
PRUNE_HIGH_DICE_THR = 0.95   

PRUNE_MIN_RATIO_GLOBAL = 0.6  
PRUNE_MIN_RATIO_CLASS  = 0.6  

EXTRA_SUBFOLDERS = {
    "xiao": "xiao-output",
    "zhong": "zhong-output",
    "da": "da-output",
}

BIG_KEYS   = ("sub6p", "solid8p")
MID_KEYS   = ("solid68",)
SMALL_KEYS = ("solid36", "sub36")

EXTRA_DIR_HINTS = {
    "xiao-output": "small",
    "zhong-output": "mid",
    "da-output": "big"
}

def prune_trainset_by_two_thresholds_with_guards(
    active_paths,
    dice_items,               
    baseline_global_n,
    baseline_class_counts,     
    low_thr=0.50,              
    high_thr=0.95,             
    min_ratio_global=0.60,
    min_ratio_class=0.60
):
    
    def _norm_thr(x): 
        x = float(x); return x/100.0 if x>1.0 else x
    low = _norm_thr(low_thr)
    high = _norm_thr(high_thr)

    if len(active_paths) < int(baseline_global_n * min_ratio_global):
        logs(f"[Prune] Skip globally: current={len(active_paths)} < "
             f"{min_ratio_global:.2f} * baseline={baseline_global_n}")
        return active_paths, {}

    from collections import defaultdict
    curr_counts = defaultdict(int)
    for p in active_paths:
        curr_counts[path_category(p)] += 1

    per_class = {'small': [], 'mid': [], 'big': []}
    for _, dsc, p, cat in dice_items:
        if cat not in per_class:
            cat = 'small' 
        per_class[cat].append((p, float(dsc)))

    kept = set(active_paths)
    removed_stats = {}

    for cls in ('small', 'mid', 'big'):
        base_cls = max(1, baseline_class_counts.get(cls, 0))
        curr_n   = curr_counts.get(cls, 0)
        min_after = int(base_cls * min_ratio_class)

        if curr_n < min_after:
            logs(f"[Prune] Skip class {cls}: current={curr_n} < "
                 f"{min_ratio_class:.2f} * baseline={base_cls}")
            removed_stats[cls] = 0
            continue

        xs = per_class.get(cls, [])

        low_side  = [(p,d) for (p,d) in xs if d <  low]
        high_side = [(p,d) for (p,d) in xs if d >= high]

        removed = []

        budget = max(0, curr_n - min_after)
        take   = min(len(low_side), budget)
        if take > 0:
            low_side.sort(key=lambda t: t[1])   
            removed.extend([p for (p,_) in low_side[:take]])
            curr_n -= take
            budget = max(0, curr_n - min_after)

        take = min(len(high_side), budget)
        if take > 0:
            high_side.sort(key=lambda t: -t[1]) 
            removed.extend([p for (p,_) in high_side[:take]])
            curr_n -= take

        kept -= set(removed)
        removed_stats[cls] = len(removed)

    new_paths = [p for p in active_paths if p in kept]
    logs(f"[Prune] (<{low*100:.1f}% or ≥{high*100:.1f}%) removed per class: {removed_stats} | remain={len(new_paths)}")
    return new_paths, removed_stats


def save_prune_reports_simple(
    dice_items,           
    removed_paths,        
    save_root,            
    fold, epoch,          
    threshold=0.98        
):
    thr = float(threshold)
    if thr > 1.0:
        thr = thr / 100.0

    out_dir = Path(save_root) / "csv" / f"fold{fold}" / "train_dice"
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_all = sorted(dice_items, key=lambda t: t[1], reverse=True)
    full_csv = out_dir / f"ep{epoch:03d}_train_dice_all.csv"
    with open(full_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'dice', 'category', 'path'])
        for rank, (_, dice, path, cat) in enumerate(sorted_all, start=1):
            w.writerow([rank, f"{dice:.6f}", cat, path])

    removed_set = set(removed_paths)
    pruned_high = [(i,d,p,c) for (i,d,p,c) in dice_items if (p in removed_set and d >= thr)]
    pruned_high.sort(key=lambda t: t[1], reverse=True)
    rm_csv = out_dir / f"ep{epoch:03d}_removed_dice_ge_{int(thr*100)}.csv"
    with open(rm_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'dice', 'category', 'path'])
        for rank, (_, dice, path, cat) in enumerate(pruned_high, start=1):
            w.writerow([rank, f"{dice:.6f}", cat, path])

    logs(f"[TrainDice] saved: {full_csv.name}, {rm_csv.name} in {out_dir}")


def save_train_dice_reports(dice_items, class2scores, save_root, fold, epoch, top_pct=0.03):
    out_dir = Path(save_root) / f"fold{fold}" / "train_dice"
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_all = sorted(dice_items, key=lambda t: t[1], reverse=True)
    full_csv = out_dir / f"ep{epoch:03d}_train_dice_all.csv"
    with open(full_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'dice', 'category', 'path'])
        for rank, (_, dice, path, cat) in enumerate(sorted_all, start=1):
            w.writerow([rank, f"{dice:.6f}", cat, path])

    k_overall = max(1, int(len(sorted_all) * max(0.0, min(top_pct, 1.0))))
    top_csv = out_dir / f"ep{epoch:03d}_top{int(top_pct*100)}_overall.csv"
    with open(top_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'dice', 'category', 'path'])
        for rank, (_, dice, path, cat) in enumerate(sorted_all[:k_overall], start=1):
            w.writerow([rank, f"{dice:.6f}", cat, path])

    for cls in ('small', 'mid', 'big'):
        entries = class2scores.get(cls, [])
        if not entries:
            continue
        entries_sorted = sorted(entries, key=lambda t: t[1], reverse=True)
        k_cls = max(1, int(len(entries_sorted) * max(0.0, min(top_pct, 1.0))))
        c_csv = out_dir / f"ep{epoch:03d}_top{int(top_pct*100)}_{cls}.csv"
        with open(c_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'dice', 'path'])
            for rank, (path, dice) in enumerate(entries_sorted[:k_cls], start=1):
                w.writerow([rank, f"{dice:.6f}", path])

    logs(f"[TrainDice] saved: {full_csv.name}, {top_csv.name} and per-class tops in {out_dir}")


def build_extra_dirs_for_fold(k, root=EXTRA_ROOT):
    import os
    return {name: os.path.join(root, f"fold{k}", sub)
            for name, sub in EXTRA_SUBFOLDERS.items()}
# ==========================================================

def build_class2scores(dice_items):
    class2scores = {'small': [], 'mid': [], 'big': []}
    for _, dsc, p, cat in dice_items:
        if cat == 'big':
            class2scores['big'].append((p, dsc))
        elif cat == 'mid':
            class2scores['mid'].append((p, dsc))
        else:
            class2scores['small'].append((p, dsc)) 
    return class2scores

def dice_sum(preds, msk, loss_fn):
    loss0 = loss_fn(preds[0], msk)
    loss1 = loss_fn(preds[1], msk)
    loss2 = loss_fn(preds[2], msk)
    loss3 = loss_fn(preds[3], msk)
    loss4 = loss_fn(preds[4], msk)
    loss = loss0 + loss1 + loss2 + loss3 + loss4
    return loss0, loss

def collect_npy(folder, pattern="*.npy"):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(glob(os.path.join(folder, pattern)))

def sample_fraction(paths, frac: float, seed: int = 123):
    if frac >= 1.0:
        return list(paths)
    n = len(paths)
    k = int(round(n * max(0.0, min(frac, 1.0))))
    if k <= 0:
        return []
    rnd = random.Random(seed)
    idxs = list(range(n))
    rnd.shuffle(idxs)
    idxs = sorted(idxs[:k])
    return [paths[i] for i in idxs]

def path_category(p: str):
    pl = p.lower()
    for token, cat in EXTRA_DIR_HINTS.items():
        if token in pl:
            return {"big":"big","mid":"mid","small":"small"}[cat]
    name = os.path.basename(pl)
    if any(k in name for k in BIG_KEYS):
        return "big"
    if any(k in name for k in MID_KEYS):
        return "mid"
    if any(k in name for k in SMALL_KEYS):
        return "small"
    return "unknown"

def compute_dice_per_sample(model, file_list, mode_tag, device,
                            batch_size=32, num_workers=12):
    model.eval()
    ds = noduleSet(file_list, ['Val', mode_tag], None, show=False)
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=(num_workers > 0),
                        drop_last=False)

    results = []
    global_idx = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Eval-Train(Dice)", leave=False):
            img, msk = data['img'], data['msk']
            img = img.type(torch.FloatTensor)
            msk = msk.type(torch.FloatTensor)
            if device != 'cpu' and torch.cuda.is_available():
                img = img.cuda(non_blocking=True)
                msk = msk.cuda(non_blocking=True)

            preds = model(img)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            # with torch.cuda.amp.autocast():
            probs = torch.sigmoid(preds)
            pred_bin = (probs >= 0.5).float()

            B = img.size(0)
            for b in range(B):
                inter = (pred_bin[b] * msk[b]).sum().item()
                denom = pred_bin[b].sum().item() + msk[b].sum().item()
                dice = (2.0 * inter) / (denom + 1e-6)

                p = file_list[global_idx + b]
                cat = path_category(p)
                results.append((global_idx + b, float(dice), p, cat))
            global_idx += B

    model.train()
    return results

def prune_trainset_by_threshold_with_guards(
    active_paths,              
    dice_items,                
    baseline_global_n,         
    baseline_class_counts,     
    threshold=0.98,            
    min_ratio_global=0.60,     
    min_ratio_class=0.60       
):
    thr = float(threshold)
    if thr > 1.0:
        thr = thr / 100.0

    if len(active_paths) < int(baseline_global_n * min_ratio_global):
        logs(f"[Prune] Skip globally: current={len(active_paths)} < "
             f"{min_ratio_global:.2f} * baseline={baseline_global_n}")
        return active_paths, {}

    from collections import defaultdict
    curr_counts = defaultdict(int)
    for p in active_paths:
        curr_counts[path_category(p)] += 1

    per_class = {'small': [], 'mid': [], 'big': []}
    for _, dsc, p, cat in dice_items:
        if cat not in per_class:
            cat = 'small'  
        per_class[cat].append((p, float(dsc)))

    kept = set(active_paths)
    removed_stats = {}

    for cls in ('small', 'mid', 'big'):
        baseline_cls = max(1, baseline_class_counts.get(cls, 0))
        curr_n = curr_counts.get(cls, 0)

        if curr_n < int(baseline_cls * min_ratio_class):
            logs(f"[Prune] Skip class {cls}: current={curr_n} < "
                 f"{min_ratio_class:.2f} * baseline={baseline_cls}")
            removed_stats[cls] = 0
            continue

        candidates = [(p, d) for (p, d) in per_class.get(cls, []) if d >= thr]
        if not candidates:
            removed_stats[cls] = 0
            continue

        candidates.sort(key=lambda t: t[1], reverse=True)

        min_allowed_after = int(baseline_cls * min_ratio_class)
        max_removable = max(0, curr_n - min_allowed_after)

        to_take = min(len(candidates), max_removable)
        to_remove = {p for (p, _) in candidates[:to_take]}

        kept -= to_remove
        removed_stats[cls] = len(to_remove)

    new_paths = [p for p in active_paths if p in kept]
    logs(f"[Prune] dice>={thr*100:.1f}% removed per class: {removed_stats} | remain={len(new_paths)}")
    return new_paths, removed_stats

def prune_trainset_with_guards(
    active_paths,              
    class2scores,              
    baseline_global_n,         
    baseline_class_counts,    
    top_pct=0.02,              
    min_ratio_global=0.60,     
    min_ratio_class=0.60       
):
    if len(active_paths) < int(baseline_global_n * min_ratio_global):
        logs(f"[Prune] Skip globally: current={len(active_paths)} < "
             f"{min_ratio_global:.2f} * baseline={baseline_global_n}")
        return active_paths, {}

    from collections import defaultdict
    curr_counts = defaultdict(int)
    for p in active_paths:
        curr_counts[path_category(p)] += 1

    kept = set(active_paths)
    removed_stats = {}

    for cls in ('small','mid','big'):
        entries = class2scores.get(cls, [])
        if not entries:
            removed_stats[cls] = 0
            continue

        baseline_cls = max(1, baseline_class_counts.get(cls, 0))
        if curr_counts.get(cls, 0) < int(baseline_cls * min_ratio_class):
            logs(f"[Prune] Skip class {cls}: current={curr_counts.get(cls,0)} < "
                 f"{min_ratio_class:.2f} * baseline={baseline_cls}")
            removed_stats[cls] = 0
            continue

        entries_sorted = sorted(entries, key=lambda t: t[1], reverse=True)
        k = int(len(entries_sorted) * max(0.0, min(top_pct, 1.0)))
        if k <= 0:
            removed_stats[cls] = 0
            continue

        to_remove = {p for p, _ in entries_sorted[:k]}
        kept -= to_remove
        removed_stats[cls] = len(to_remove)

    new_paths = [p for p in active_paths if p in kept]
    logs(f"[Prune] removed per class: {removed_stats} | remain={len(new_paths)}")
    return new_paths, removed_stats

class trainBase(GC):
    seg_path = None
    model_name = None

    def __init__(self, model2d, model3d, lossList):
        super(trainBase, self).__init__(
            train=configs.train, dataset=configs.dataset, log_name=configs.log_name,
            mode=configs.mode, pathV=configs.pathV, LossV=configs.LossV,
            FileV=configs.FileV, MetricsV=configs.MetricsV, sup=configs.sup,
            server=configs.server
        )
        self.model2d = model2d
        self.model3d = model3d
        self.lossList = lossList

        if self.mode == '2d':
            self.transform = transforms2d
            self.models = self.model2d
            self.mode_tag = '2d'
        else:
            self.transform = transforms3d
            self.models = self.model3d
            self.mode_tag = '3d'

        if self.dataset == 'luna':
            self.pth_path = self.pth_luna_path
        elif self.dataset == 'lci':
            self.pth_path = self.pth_lci_path
        else:
            self.pth_path = self.pth_lidc_path

        self.now_k = -1
        self.reroll = 0

    def baseInfo(self):
        logs(
            f" Dataset:{self.dataset},\n"
            f' Train batch:{str(self.train_batch_size)},\n'
            f' Val and Test batch:{str(self.val_and_test_batch_size)},\n'
            f' Optimizer:{self.optimizer},\n'
            f' lr:{self.lr},\n'
            f' k_fold:{self.k_fold},\n'
            f' num worker:{self.num_worker},\n'
            f' early stop:{self.earlyEP},\n'
            f' mode:{self.mode},\n'
            f' device:{self.device},\n'
            f' epoches:{self.epochs},\n'
            f' seg path:{self.seg_path},\n'
            f' pth path:{self.pth_path},\n'
            f' sup :{self.sup}'
        )

    def _build_train_loader_from_list(self, file_list):
        ds = noduleSet(file_list, ['Train', self.mode_tag], self.transform, self.show)
        loader = DataLoader(
            ds, batch_size=self.train_batch_size, shuffle=True,
            num_workers=self.num_worker, pin_memory=True, drop_last=True
        )
        return loader

    def initNetwork(self, k):
        self.now_k = k
        if self.mode == '2d':
            model = get_model2d(self.model_name, self.device)
        else:
            model = get_model3d(self.model_name, self.device)

        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), self.lr, momentum=0.9, weight_decay=1e-4)

        scalar = torch.cuda.amp.GradScaler()
        eStop = EarlyStopping(patience=self.earlyEP, fold=int(k), mode=self.mode, path=self.pth_path, verbose=True)

        train_list, val_and_test_list = [], []
        lists = [train_list, val_and_test_list]

        lossf = Loss(self.loss_name) if self.LossV == 1 else LossV2(self.loss_name)

        if self.FileV == 'npy':
            lists = set_init(k, self.seg_path, None, lists)
        else:
            lists = set_init(k, self.seg_path, None, lists, format='*.nii.gz')
 
        self._orig_train_list = list(train_list)            
        self._baseline_train_n = len(self._orig_train_list) 

        from collections import defaultdict
        self._baseline_class_counts = defaultdict(int)
        for p in self._orig_train_list:
            self._baseline_class_counts[path_category(p)] += 1

        self._train_list_current = list(self._orig_train_list)

        self.extra_dirs = build_extra_dirs_for_fold(k)
        self._extra_lists = {
            "xiao": collect_npy(self.extra_dirs["xiao"]),
            "zhong": collect_npy(self.extra_dirs["zhong"]),
            "da": collect_npy(self.extra_dirs["da"]),
        }
        logs(f"[Warmup Ready] fold{k} extras -> "
             f"xiao={len(self._extra_lists['xiao'])}, "
             f"zhong={len(self._extra_lists['zhong'])}, "
             f"da={len(self._extra_lists['da'])}")
        
        train_iter = self._build_train_loader_from_list(self._train_list_current)

        self._extra_used = set()

        val_and_test_dataset = noduleSet(val_and_test_list, ['Val', self.mode_tag], None, self.show)
        val_and_test_iter = DataLoader(
            val_and_test_dataset, batch_size=self.val_and_test_batch_size,
            num_workers=self.num_worker, pin_memory=True, shuffle=False, drop_last=False
        )
        test_dataset = noduleSet(val_and_test_list, ['test', self.mode_tag], None, self.show)
        test_iter = DataLoader(
            test_dataset, batch_size=self.val_and_test_batch_size,
            num_workers=self.num_worker, pin_memory=True, shuffle=False, drop_last=False
        )

        return optimizer, model, lossf, scalar, eStop, train_iter, val_and_test_iter, test_iter

    def kFoldTrain(self, k, eltSet, pretrained=False):
        now = time.time()
        optimizer, model, lossf, scalar, eStop, train_iter, val_and_test_iter, test_iter = self.initNetwork(k)

        mixed_enabled = False  
        if self.MetricsV == 1:
            if configs.train:
                for ep in range(1, self.epochs + 1):
                    if (not mixed_enabled) and (ep > WARMUP_EPOCHS):
                        extra_added = []
                        for key in ("xiao", "zhong", "da"):
                            frac = float(EXTRA_FRACTIONS.get(key, 1.0))
                            sub = sample_fraction(self._extra_lists.get(key, []), frac, seed=123 + ep)
                            extra_added.extend(sub)
                            logs(f"[Mix] add {key}: total={len(self._extra_lists.get(key, []))}, "
                                 f"frac={frac}, choose={len(sub)}")

                        before = len(self._train_list_current)
                        cur_set = set(self._train_list_current)
                        extra_added = [p for p in extra_added if p not in cur_set]
                        self._train_list_current.extend(extra_added)
                        self._extra_used.update(extra_added)
                        logs(f"[Mix] train_list: {before} -> {len(self._train_list_current)} "
                             f"(added {len(extra_added)})")

                        train_iter = self._build_train_loader_from_list(self._train_list_current)
                        mixed_enabled = True

                    if not eStop.early_stop:
                        train_fps = self.trainFun(ep, train_iter, model, optimizer, lossf, scalar, eStop)
                        eltSet[6].append(float(train_fps))
                        self.validationFun(val_and_test_iter, model, lossf, ep, eStop, k)
                    else:
                        eltSet[8].append(float(eStop.epoch))
                        break

                    if mixed_enabled and (ep % TRAIN_EVAL_INTERVAL == 0):
                        def _norm_thr(x): 
                            x = float(x); return x/100.0 if x > 1.0 else x
                        low  = _norm_thr(PRUNE_LOW_DICE_THR)
                        high = _norm_thr(PRUNE_HIGH_DICE_THR)

                        logs(f"[Prune] epoch {ep}: evaluate train dice & prune (<{low*100:.1f}% or ≥{high*100:.1f}%)")
                        dice_items = compute_dice_per_sample(model, self._train_list_current, self.mode_tag, self.device)

                        new_list, stats = prune_trainset_by_two_thresholds_with_guards(
                            active_paths=self._train_list_current,
                            dice_items=dice_items,
                            baseline_global_n=self._baseline_train_n,
                            baseline_class_counts=self._baseline_class_counts,
                            low_thr=PRUNE_LOW_DICE_THR,        # 0.50
                            high_thr=PRUNE_HIGH_DICE_THR,      # 0.95
                            min_ratio_global=PRUNE_MIN_RATIO_GLOBAL,
                            min_ratio_class=PRUNE_MIN_RATIO_CLASS,
                        )

                        removed_paths = set(self._train_list_current) - set(new_list)

                        save_prune_reports_simple(
                            dice_items=dice_items,
                            removed_paths=removed_paths,
                            save_root=self.pth_path,
                            fold=k,
                            epoch=ep,
                            threshold=PRUNE_HIGH_DICE_THR     
                        )

                        pruned_n = len(self._train_list_current) - len(new_list)
                        self._train_list_current = new_list
                        logs(f"[Prune] stats: {stats} | pruned={pruned_n}")

                        refill_cond = REFILL_AFTER_PRUNE and all(float(EXTRA_FRACTIONS.get(k, 1.0)) < 1.0 for k in ("xiao","zhong","da"))
                        added_total = 0
                        if refill_cond:
                            key_of_cls = {"small":"xiao", "mid":"zhong", "big":"da"}
                            cur_set = set(self._train_list_current)

                            for cls_name, removed_cnt in stats.items():
                                if isinstance(removed_cnt, dict):
                                    removed_cnt = removed_cnt.get("drop", 0)
                                if removed_cnt is None or removed_cnt <= 0:
                                    continue

                                key = key_of_cls.get(cls_name)
                                if key is None:
                                    continue

                                pool = [p for p in self._extra_lists.get(key, [])
                                        if (p not in cur_set) and (p not in self._extra_used)]

                                if not pool:
                                    continue

                                rnd = random.Random(2025 + ep)   
                                rnd.shuffle(pool)
                                take = pool[:removed_cnt]        
                                
                                remain_pool = max(0, len(pool) - removed_cnt)
                                logs(f"[Refill] class={cls_name} pool_before={len(pool)} add={len(take)} pool_after={remain_pool}")

                                
                                if not take:
                                    continue

                                self._train_list_current.extend(take)
                                self._extra_used.update(take)
                                added_total += len(take)
                                logs(f"[Refill] class={cls_name} from={key} add={len(take)}")

                        if pruned_n > 0 or added_total > 0:
                            train_iter = self._build_train_loader_from_list(self._train_list_current)
                            logs(f"[Refill] total added={added_total} | train size={len(self._train_list_current)}")

            load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer, self.loss_name, model, k)
            k_fold_eva = self.testFun(k, test_iter, model, mode='evluation')
            for t in range(len(k_fold_eva)):
                eltSet[t].append(float(k_fold_eva[t]))
            infer_fps = self.testFun(k, test_iter, model)
            eltSet[7].append(float(infer_fps))
            logs(f'Fold {k} Infer {infer_fps:.2f} FPS')
            logs(f'one fold time consumed:{(round((time.time() - now) / 3600))} hours')
            return eltSet

    def trainFun(self, ep, train_iter, model, optimizer, loss_fn, scalar, eStop):
        times = []
        loop = tqdm(train_iter, desc=f"Train Ep{ep}")
        if self.optimizer == 'adam':
            for idx, data in enumerate(loop):
                start_time = time.time()
                img, msk = data['img'], data['msk']
                img = img.type(torch.FloatTensor)
                msk = msk.type(torch.FloatTensor)
                if self.device != 'cpu' and torch.cuda.is_available():
                    img, msk = Variable(img.cuda(), requires_grad=False), Variable(msk.cuda(), requires_grad=False)

                with torch.cuda.amp.autocast():
                    try:
                        with torch.autograd.set_detect_anomaly(False):
                            optimizer.zero_grad()
                            if self.sup:
                                preds = model(img)
                                _, loss = dice_sum(preds, msk, loss_fn)
                            else:
                                preds = model(img)
                                loss = loss_fn(preds, msk)
                            scalar.scale(loss).backward()
                            scalar.step(optimizer)
                            scalar.update()
                            loop.set_postfix(loss=loss.item())
                            end_time = time.time()
                            times.append(end_time - start_time)
                            self.reroll = 0
                    except Exception as e:
                        logs('=' * 30 + 'case error, and rerolling' + '=' * 30)
                        logs(e.args)
                        if self.reroll < 20:
                            torch.cuda.empty_cache()
                            load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer,
                                                    self.loss_name, model, self.now_k, verbose=False)
                            self.reroll += 1
                            continue
                        eStop.early_stop = True
                        torch.cuda.empty_cache()
                        break

        fps = 1.0 / np.mean(times)
        return fps

    def validationFun(self, loader, model, loss_fn, ep, eStop, k):
        model.eval()
        os.makedirs(self.pred_path, exist_ok=True)

        val_metrics = Metrics().to(self.device)
        with torch.no_grad():
            for idx, data in tqdm(enumerate(loader), desc=f"Val Ep{ep}", leave=False):
                img, msk = data['img'], data['msk']
                img = img.type(torch.FloatTensor)
                msk = msk.type(torch.FloatTensor)
                if self.device != 'cpu' and torch.cuda.is_available():
                    img, msk = Variable(img.cuda(), requires_grad=False), Variable(msk.cuda(), requires_grad=False)
                if self.sup:
                    preds = model(img)
                    _, loss = dice_sum(preds, msk, loss_fn)
                    preds = preds[0]
                else:
                    preds = model(img)
                probs = torch.sigmoid(preds)
                pred_bin = (probs >= 0.5).float()
                val_metrics(pred_bin, msk)

        dice = val_metrics.evluation(f"Ep {ep}")
        dice = dice[2]
        if float(dice) > 87.73:
            save_dir = Path(self.pth_path) / f"fold{k}"
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_name = f"{self.mode}_{self.model_name}_{k}_{self.optimizer}_dice_checkpoint{dice}.pth"
            save_path = save_dir / ckpt_name
            torch.save(model.state_dict(), save_path)
        eStop(float(dice), model, ep, self.model_name, self.optimizer, self.loss_name)
        model.train()

    def testFun(self, k_fold, loader, model, mode='infer'):
        model.eval()
        if self.MetricsV == 1:
            metrics = Metrics().to(self.device)
            times = []
            with torch.no_grad():
                for idx, data in tqdm(enumerate(loader), desc="Test", leave=False):
                    test_start_time = time.time()
                    img, msk = data['img'], data['msk']
                    img = img.type(torch.FloatTensor)
                    msk = msk.type(torch.FloatTensor)
                    if self.device != 'cpu' and torch.cuda.is_available():
                        img, msk = Variable(img.cuda(), requires_grad=False), Variable(msk.cuda(), requires_grad=False)
                    if self.sup:
                        preds = model(img)
                        preds = preds[0]
                    else:
                        preds = model(img)
                    preds = torch.sigmoid(preds)
                    preds = (preds >= 0.5).float()
                    test_end_time = time.time()
                    times.append(test_end_time - test_start_time)
                    if mode == 'evluation':
                        metrics(preds, msk)
                    if idx % 50 == 0 and self.mode == '2d':
                        save_tmp(self.pred_path, img[0], msk[0], preds[0], 'test_tmp')
            model.train()
            if mode == 'evluation':
                fprecision, fsensitivity, ff1, fmIou, voe, rve = metrics.evluation(k_fold)
                return [fprecision, fsensitivity, ff1, fmIou, voe, rve]
            else:
                fps = 1.0 / np.mean(times)
                return fps
        else:
            metrics = MetricsV2().to(self.device)
            times = []
            with torch.no_grad():
                for idx, data in tqdm(enumerate(loader), desc="Test", leave=False):
                    test_start_time = time.time()
                    img, msk = data['img'], data['msk']
                    img = img.type(torch.FloatTensor)
                    msk = msk.type(torch.FloatTensor)
                    if self.device != 'cpu' and torch.cuda.is_available():
                        img, msk = Variable(img.cuda(), requires_grad=False), Variable(msk.cuda(), requires_grad=False)
                    if self.sup:
                        preds = model(img)
                        preds = preds[0]
                    else:
                        preds = model(img)
                    preds = torch.sigmoid(preds)
                    preds = (preds > 0.5).float()
                    test_end_time = time.time()
                    times.append(test_end_time - test_start_time)
                    if mode == 'evluation':
                        metrics(preds, msk)
                    if idx % 50 == 0 and self.mode == '2d':
                        save_tmp(self.pred_path, img[0], msk[0], preds[0], 'test_tmp')
            model.train()
            if mode == 'evluation':
                dice, hd, msd = metrics.evluation(k_fold)
                return [dice, hd, msd]
            else:
                fps = 1.0 / np.mean(times)
                return fps

    def run(self, verbose=False):
        pretrained = configs.pretrained
        if verbose:
            self.baseInfo()
        for model in self.models:
            self.model_name = model
            for loss in self.lossList:
                self.loss_name = loss
                start_time = time.time()

                if self.MetricsV == 1:
                    precision, sensitivity, f1, Iou, voe, rve, train_times, infer_times, oc = \
                        [], [], [], [], [], [], [], [], []
                    eltSet = [precision, sensitivity, f1, Iou, voe, rve, train_times, infer_times, oc]
                    logs(f'Fold {5},model {model}')
                    eltSet = self.kFoldTrain(5, eltSet, pretrained=pretrained)

                    logs(
                        f'Final '
                        f'Precision:{avgStd(precision, log=True)},'
                        f'Sensitivity:{avgStd(sensitivity, log=True)},'
                        f'Dsc:{avgStd(f1, log=True)},'
                        f'Iou:{avgStd(Iou, log=True)},\n'
                        f'Voe:{avgStd(voe, log=True)},'
                        f'Rve:{avgStd(rve, log=True)},'
                        f'{self.dataset} fps:{avgStd(train_times, log=True)},'
                        f'Infer fps:{avgStd(infer_times, log=True)},'
                        f'Optimal CVG:{avgStd(oc, log=True)},'
                    )
                    end_time = time.time()
                    showTime('Total', start_time, end_time)