# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import random
import os
import re
import time
from pathlib import Path
import numpy as np
import torch.nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset 
from tqdm import tqdm
from typing import Optional
import math
import configs
from configs import GC
from utils.EarlyStop import EarlyStopping
from utils.Metrics import Metrics
# from utils.MetricsV2 import MetricsV2
from utils.helper import transforms2d, transforms3d, get_model2d, get_model3d, set_init, avgStd, showTime, save_tmp, \
    load_model_k_checkpoint, get_gate
from utils.logger import logs
from utils.loss import Loss
from utils.lossV2 import LossV2
from utils.noduleSet import noduleSet
from glob import glob


def _aug_to_base(aug_name: str) -> Optional[str]:
    name = os.path.basename(aug_name)
    if not name.startswith("condon_") or not name.endswith(".npy"):
        return None
    s = name[len("condon_"):]  
    m = re.match(r"^(.*)_(\d+)\.npy$", s)
    if m:
        return m.group(1) + ".npy"
    return s  

def build_aug_map(extra_dir: str) -> dict:
    files = sorted(glob(os.path.join(extra_dir, "condon_*.npy")))
    mp = {}
    for p in files:
        base = _aug_to_base(p)
        if base is None:
            continue
        mp.setdefault(base, []).append(p)

    def sort_key(path):
        n = os.path.basename(path)
        m = re.match(r".*_(\d+)\.npy$", n)
        return int(m.group(1)) if m else 0  
    for k in mp:
        mp[k].sort(key=sort_key)
    return mp

class _NpyBaseDataset(Dataset):
    def __init__(self, file_list):
        self.paths = list(file_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        arr = np.load(p)  # (2,H,W)
        img = torch.from_numpy(arr[0].astype(np.float32))[None, ...]
        msk = torch.from_numpy(arr[1].astype(np.float32))[None, ...]
        base = os.path.basename(p)
        return {"img": img, "msk": msk, "base": base, "path": p}

def dice_sum(preds, msk, loss_fn):
    loss0 = loss_fn(preds[0], msk)
    loss1 = loss_fn(preds[1], msk)
    loss2 = loss_fn(preds[2], msk)
    loss3 = loss_fn(preds[3], msk)
    loss4 = loss_fn(preds[4], msk)
    loss = loss0 + loss1 + loss2 + loss3 + loss4
    return loss0, loss

def append_extra_to_train(train_list, val_list, extra_dir, pattern="*.npy",
                            recursive=False, exclude_keywords=("sub3c", "solid3c")):
        gpat = os.path.join(extra_dir, "**", pattern) if recursive else os.path.join(extra_dir, pattern)
        extra = sorted(glob(gpat, recursive=recursive))

        if exclude_keywords:
            extra = [p for p in extra if all(k not in p for k in exclude_keywords)]

        val_set   = set(val_list)
        train_set = set(train_list)
        new_items = [p for p in extra if p not in val_set and p not in train_set]

        train_list.extend(new_items)
        return new_items  

class trainBase(GC):
    seg_path = None
    model_name = None

    def __init__(self, model2d, model3d, lossList):
        super(trainBase, self).__init__(train=configs.train, dataset=configs.dataset, log_name=configs.log_name,
                                        mode=configs.mode, pathV=configs.pathV, LossV=configs.LossV,
                                        FileV=configs.FileV, MetricsV=configs.MetricsV, sup=configs.sup,
                                        server=configs.server)
        self.model2d = model2d
        self.model3d = model3d
        self.lossList = lossList
        self.kgenshard, self.kgensmid = (10, 5) #(8, 2)

        if self.mode == '2d':
            self.transform = transforms2d
            self.models = self.model2d
        else:
            self.transform = transforms3d
            self.models = self.model3d

        if self.dataset == 'luna':
            self.pth_path = self.pth_luna_path
        elif self.dataset == 'lci':
            self.pth_path = self.pth_lci_path
        else:
            self.pth_path = self.pth_lidc_path

        self.now_k = -1
        self.reroll = 0
    
    @torch.no_grad()
    def _eval_train_dice_per_base(self, model, base_ids=None, batch_size=None, thr=0.5, smooth=1e-5):
        model.eval()
        if batch_size is None:
            batch_size = self.val_and_test_batch_size
        if base_ids is None:
            base_ids = self._active_base_ids

        file_list = [self._orig_train_map[b] for b in base_ids if b in self._orig_train_map]
        ds = _NpyBaseDataset(file_list)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=self.num_worker, pin_memory=True, drop_last=False)

        inter, denom = {}, {}
        for batch in loader:
            img = batch["img"].to(self.device).float()
            msk = batch["msk"].to(self.device).float()
            bases = batch["base"]
            logits = model(img)
            probs = torch.sigmoid(logits)
            pred  = (probs >= thr).float()
            B = pred.size(0)
            p = pred.view(B, -1); t = msk.view(B, -1)
            inter_b = (p*t).sum(1).cpu().double()
            denom_b = (p.sum(1)+t.sum(1)).cpu().double()
            for j, base in enumerate(bases):
                inter[base] = inter.get(base, 0.0) + inter_b[j].item()
                denom[base] = denom.get(base, 0.0) + denom_b[j].item()

        dice = {b: (2*inter[b] + smooth) / (denom[b] + smooth) for b in inter}
        model.train()
        return dice

    import math

    def _rebuild_train_iter_by_thirds(self, dice_dict, hard_ratio=0.20, easy_ratio=0.20,
                                  k_gens_hard=4, k_gens_mid=2, min_keep_ratio=0.40):

        bases_active = [b for b in self._active_base_ids if b in dice_dict]
        if not bases_active:
            logs("[Dynamic] active 集为空，跳过重建。")
            return None

        bases_sorted = sorted(bases_active, key=lambda b: dice_dict[b])  
        n = len(bases_sorted)
        n_hard = min(n, math.ceil(n * hard_ratio))
        n_easy = min(n - n_hard, math.ceil(n * easy_ratio))
        b1 = n_hard
        b2 = n - n_easy

        hard_ids = set(bases_sorted[:b1])
        mid_ids  = set(bases_sorted[b1:b2])
        easy_ids = set(bases_sorted[b2:])

        min_keep = math.ceil(min_keep_ratio * len(self._orig_train_map))
        future_active = (self._active_base_ids - easy_ids)
        if len(future_active) < min_keep:
            need_keep = min_keep - len(future_active)
            keep_back = set(list(easy_ids)[:need_keep])
            easy_ids -= keep_back
            future_active = (self._active_base_ids - easy_ids)

        self._active_base_ids = set(future_active)

        new_list = []
        for base in hard_ids:
            if base not in self._active_base_ids:
                continue
            if base in self._orig_train_map:
                new_list.append(self._orig_train_map[base])
            gens = self._aug_map.get(base, [])
            new_list.extend(gens[:k_gens_hard])

        for base in mid_ids:
            if base not in self._active_base_ids:
                continue
            if base in self._orig_train_map:
                new_list.append(self._orig_train_map[base])
            gens = self._aug_map.get(base, [])
            new_list.extend(gens[:k_gens_mid])

        val_set = set(self._val_list)
        seen, final_list = set(), []
        for p in new_list:
            if p in val_set or p in seen:
                continue
            seen.add(p); final_list.append(p)

        logs(f"[Dynamic] active={len(self._active_base_ids)} | hard={len(hard_ids)} mid={len(mid_ids)} "
             f"easy-removed={len(easy_ids)} | files={len(final_list)}")

        train_dataset = noduleSet(final_list, ['Train', '2d'], self.transform, self.show)
        train_iter = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                num_workers=self.num_worker, pin_memory=True, drop_last=True)
        return train_iter

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

    def initNetwork(self, k):
        self.now_k = k
        if self.mode == '2d':
            model = get_model2d(self.model_name, self.device)
        else:
            model = get_model3d(self.model_name, self.device)

        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), self.lr, momentum=0.9, weight_decay=1e-4)  # 1e-4

        scalar = torch.cuda.amp.GradScaler()
        eStop = EarlyStopping(patience=self.earlyEP, fold=int(k), mode=self.mode, path=self.pth_path, verbose=True)

        train_list = []
        val_and_test_list = []
        lists = [train_list, val_and_test_list]

        if self.LossV == 1:
            lossf = Loss(self.loss_name)
        else:
            lossf = LossV2(self.loss_name)

        if self.FileV == 'npy':
            lists = set_init(k, self.seg_path, None, lists)
        else:
            lists = set_init(k, self.seg_path, None, lists, format='*.nii.gz')

        self._orig_train_list = list(train_list)
        self._val_list = list(val_and_test_list)

        self._orig_train_map = {os.path.basename(p): p for p in self._orig_train_list}

        self._extra_dir = f"./5foldadd/fold{k}/samples_many_9490"
        self._aug_map = build_aug_map(self._extra_dir)
        logs(f"[AugMap] extras from {self._extra_dir}: {sum(len(v) for v in self._aug_map.values())} files")

        if self.mode == '2d':
            train_dataset = noduleSet(train_list, ['Train', '2d'], self.transform, self.show, )
            train_iter = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                    num_workers=self.num_worker, pin_memory=True, drop_last=True)
            
            val_and_test_dataset = noduleSet(val_and_test_list, ['Val', '2d'], None, self.show)
            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=False,
                                           drop_last=False)
            test_dataset = noduleSet(val_and_test_list, ['test', '2d'], None, self.show)
            test_iter = DataLoader(test_dataset, batch_size=self.val_and_test_batch_size,
                                   num_workers=self.num_worker, pin_memory=True, shuffle=False,
                                   drop_last=False)
            self._orig_train_list = list(train_list)
            self._val_list = list(val_and_test_list)
            self._orig_train_map = {os.path.basename(p): p for p in self._orig_train_list}
            self._extra_dir = f"./5foldadd/fold{k}/samples_many_9490"
            self._aug_map = build_aug_map(self._extra_dir)    
            self._active_base_ids = set(self._orig_train_map.keys())     

        return optimizer, model, lossf, scalar, eStop, train_iter, val_and_test_iter, test_iter

    def kFoldTrain(self, k, eltSet, pretrained=False):
        now = time.time()
        optimizer, model, lossf, scalar, eStop, train_iter, val_and_test_iter, test_iter = self.initNetwork(k)

        if self.MetricsV == 1:
            if configs.train:
                for ep in range(1, self.epochs + 1):
                    if eStop.early_stop:
                        eltSet[8].append(float(eStop.epoch))
                        break

                    self.trainFun(ep, train_iter, model, optimizer, lossf, scalar, eStop)
                    self.validationFun(val_and_test_iter, model, lossf, ep, eStop, k)

                    if (ep > 10) and ((ep - 10) % 3 == 0):
                        dice_dict = self._eval_train_dice_per_base(model, base_ids=self._active_base_ids,
                                                                   batch_size=self.val_and_test_batch_size, thr=0.5)
                        new_train_iter = self._rebuild_train_iter_by_thirds(
                            dice_dict, hard_ratio=0.33, easy_ratio=0.03,
                            k_gens_hard=self.kgenshard, k_gens_mid=self.kgensmid, min_keep_ratio=0.8
                        )
                        if new_train_iter is not None:
                            train_iter = new_train_iter

            load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer, self.loss_name,
                                    model, k)
            k_fold_eva = self.testFun(k, test_iter, model, mode='evluation')
            for t in range(len(k_fold_eva)):
                eltSet[t].append(float(k_fold_eva[t]))

            infer_fps = self.testFun(k, test_iter, model)
            eltSet[7].append(float(infer_fps))
            logs(f'Fold {k} Infer {infer_fps:.2f} FPS')
            logs(f'one fold time consumed:{(round((time.time() - now) / 3600))} hours', )
            return eltSet

    def trainFun(self, ep, train_iter, model, optimizer, loss_fn, scalar, eStop):
        times = []
        loop = tqdm(train_iter)
        if self.optimizer == 'adam':
            for idx, data in enumerate(loop):
                start_time = time.time()

                img, msk = data['img'], data['msk']
                img = img.type(torch.FloatTensor)
                msk = msk.type(torch.FloatTensor)

                if self.device != 'cpu' and torch.cuda.is_available():
                    img, msk = Variable(img.cuda(), requires_grad=False), \
                        Variable(msk.cuda(), requires_grad=False)

                with torch.cuda.amp.autocast():  
                    try:
                        with torch.autograd.set_detect_anomaly(False):
                            optimizer.zero_grad()
                            if self.sup:
                                preds = model(img)
                                loss_tar, loss = dice_sum(preds, msk, loss_fn)
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

    def validationFun(self, loader, model, loss_fn, ep, eStop,k):
        model.eval()
        os.makedirs(self.pred_path, exist_ok=True)
        dicebest = [90.17, 87.11, 89.17, 89.12, 89.54]

        from utils.Metrics import Metrics
        val_metrics = Metrics().to(self.device)
        
        with torch.no_grad():
            for idx, data in tqdm(enumerate(loader)):
                img, msk = data['img'], data['msk']
                img = img.type(torch.FloatTensor)
                msk = msk.type(torch.FloatTensor)

                if self.device != 'cpu' and torch.cuda.is_available():
                    img, msk = Variable(img.cuda(), requires_grad=False), \
                        Variable(msk.cuda(), requires_grad=False)
                preds = model(img)
                probs = torch.sigmoid(preds)
                pred_bin = (probs >= 0.5).float()
                val_metrics(pred_bin, msk)
        dice = val_metrics.evluation(f"Ep {ep}")
        dice = dice[2]
        if(float(dice)>dicebest[k-1]):
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
                for idx, data in tqdm(enumerate(loader)):
                    test_start_time = time.time()

                    img, msk = data['img'], data['msk']
                    img = img.type(torch.FloatTensor)
                    msk = msk.type(torch.FloatTensor)

                    if self.device != 'cpu' and torch.cuda.is_available():
                        img, msk = Variable(img.cuda(), requires_grad=False), \
                            Variable(msk.cuda(), requires_grad=False)
                        
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
                    precision, sensitivity, f1, Iou, voe, rve, train_times, infer_times, oc = [], [], [], [], [], [], [], [], []
                    eltSet = [precision, sensitivity, f1, Iou, voe, rve, train_times, infer_times, oc]

                    logs(f'Fold {1},model {model}')
                    eltSet = self.kFoldTrain(1, eltSet, pretrained=pretrained)
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