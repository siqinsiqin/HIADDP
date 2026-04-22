# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
# -*-coding:utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from utils.helper import avgStd
from utils.logger import logs


class Metrics(nn.Module):
    """
    顺序无关（global micro-avg）的分割指标：
    Precision / Recall / Dice(F1) / IoU / VOE(=1-IoU, 越小越好) / RVE(↑=1-|Vp-Vg|/Vg)
    同时支持按样本记录 Dice 并导出 CSV：name -> dice(%)
    """
    def __init__(self, threshold=0.5, eps=1e-8, debug=False):
        super().__init__()
        self.th = threshold
        self.eps = eps
        self.debug = debug

        # 用 buffer 登记所有计数器，避免“属性不存在”
        for name in ['tp','fp','fn','inter','union','pred_sum','gt_sum']:
            self.register_buffer(name, torch.zeros((), dtype=torch.long))
        for name in ['empty_gt','empty_pred','empty_both',
                     'den0_prec','den0_rec','den0_dice','den0_iou']:
            self.register_buffer(name, torch.zeros((), dtype=torch.long))

        # 诊断 / 导出用的列表
        self.batch_sizes = []
        self.batch_precision = []
        self.batch_recall = []
        self.batch_dice = []
        self.batch_iou = []
        self.case_dice = []
        self.case_records = []   # [{'name': str, 'dice(%)': float}, ...]

    def reset(self):
        # 计数器清零
        self.tp.zero_(); self.fp.zero_(); self.fn.zero_()
        self.inter.zero_(); self.union.zero_()
        self.pred_sum.zero_(); self.gt_sum.zero_()
        self.empty_gt.zero_(); self.empty_pred.zero_(); self.empty_both.zero_()
        self.den0_prec.zero_(); self.den0_rec.zero_(); self.den0_dice.zero_(); self.den0_iou.zero_()
        # 列表清空
        self.batch_sizes.clear()
        self.batch_precision.clear(); self.batch_recall.clear()
        self.batch_dice.clear(); self.batch_iou.clear()
        self.case_dice.clear()
        self.case_records.clear()

    @torch.no_grad()
    def __call__(self, preds, labels, names=None):
        # 设备 & 数值 & 形状对齐
        labels = labels.to(preds.device)
        preds  = torch.nan_to_num(preds,  nan=0.0, posinf=1.0, neginf=0.0)
        labels = torch.nan_to_num(labels, nan=0.0)

        # 常见二值分割：preds=[N,1,...], labels=[N,...] -> 压掉通道维
        if preds.ndim == labels.ndim + 1 and preds.shape[1] == 1:
            preds = preds.squeeze(1)

        # 二值化（不做 inplace）
        preds  = (preds  > self.th)
        labels = (labels >= 0.5)

        # 记录 batch 大小
        self.batch_sizes.append(int(preds.shape[0]))

        # —— 本 batch 的计数（tensor，保持 dtype/device 一致）——
        tp    = (preds & labels).sum()
        fp    = (preds & ~labels).sum()
        fn    = ((~preds) & labels).sum()
        inter = tp
        union = (preds | labels).sum()
        ps    = preds.sum()
        gs    = labels.sum()

        # 累加到全局 buffer
        self.tp += tp; self.fp += fp; self.fn += fn
        self.inter += inter; self.union += union
        self.pred_sum += ps; self.gt_sum += gs

        # 空集诊断计数
        if gs == 0: self.empty_gt += 1
        if ps == 0: self.empty_pred += 1
        if gs == 0 and ps == 0: self.empty_both += 1

        # —— 诊断：按批“逐批算再平均”的值（仅用于展示抖动来源）——
        den_prec = tp + fp
        den_rec  = tp + fn
        den_dice = 2*tp + fp + fn
        den_iou  = union
        self.den0_prec += (den_prec == 0).long()
        self.den0_rec  += (den_rec  == 0).long()
        self.den0_dice += (den_dice == 0).long()
        self.den0_iou  += (den_iou  == 0).long()

        b_prec = (tp.float() / den_prec.float()).item() if den_prec.item() > 0 else np.nan
        b_rec  = (tp.float() / den_rec.float()).item()  if den_rec.item()  > 0 else np.nan
        b_dice = (2*tp.float() / den_dice.float()).item() if den_dice.item() > 0 else np.nan
        b_iou  = (inter.float() / den_iou.float()).item() if den_iou.item() > 0 else np.nan
        self.batch_precision.append(b_prec)
        self.batch_recall.append(b_rec)
        self.batch_dice.append(b_dice)
        self.batch_iou.append(b_iou)

        # —— 按样本 Dice（用于导出 CSV）——
        dims = tuple(range(1, preds.ndim))
        tp_i = ((preds & labels).sum(dim=dims)).float()
        fp_i = ((preds & ~labels).sum(dim=dims)).float()
        fn_i = (((~preds) & labels).sum(dim=dims)).float()
        dice_i = 2*tp_i / (2*tp_i + fp_i + fn_i + self.eps)  # [N], 0~1
        dice_pct = (dice_i * 100).clamp(0, 100).detach().cpu().tolist()
        self.case_dice.extend(dice_pct)

        # 记录 name -> dice(%)
        if names is not None:
            if torch.is_tensor(names):
                names = [str(x) for x in names]
            else:
                names = list(names)
            n = min(len(names), len(dice_pct))
            for nm, d in zip(names[:n], dice_pct[:n]):
                self.case_records.append({"name": str(nm), "dice": float(d)})

    def evluation(self, fold):
        tp = self.tp.item(); fp = self.fp.item(); fn = self.fn.item()
        inter = self.inter.item(); union = self.union.item()
        pred_sum = self.pred_sum.item(); gt_sum = self.gt_sum.item()

        empty_both = (gt_sum == 0 and pred_sum == 0)
        precision   = tp / (tp + fp + self.eps)
        sensitivity = tp / (tp + fn + self.eps)
        f1          = 1.0 if empty_both else 2*tp / (2*tp + fp + fn + self.eps)
        iou         = 1.0 if empty_both else inter / (union + self.eps)
        voe         = 1.0 - iou

        if gt_sum == 0:
            rve_good = 1.0 if pred_sum == 0 else 0.0
        else:
            rve = abs(pred_sum - gt_sum) / (gt_sum + self.eps)
            rve_good = float(np.clip(1.0 - rve, 0.0, 1.0))

        # 兼容 Writer/avgStd（百分制）
        self.precision   = [round(precision*100, 4)]
        self.sensitivity = [round(sensitivity*100, 4)]
        self.f1_score    = [round(f1*100, 4)]
        self.iou         = [round(iou*100, 4)]
        self.voe         = [round(voe*100, 4)]
        self.rve         = [round(rve_good*100, 4)]

        logs(
            f"Fold {fold}"
            f", Precision : " + avgStd(self.precision, log=True) +
            f", Sensitivity: " + avgStd(self.sensitivity, log=True) +
            f", Dsc: "       + avgStd(self.f1_score, log=True) +
            f", IoU: "       + avgStd(self.iou, log=True) +
            f"\n, VOE(↓): "  + avgStd(self.voe, log=True) +
            f", RVE(↑): "    + avgStd(self.rve, log=True)
        )
        return (
            avgStd(self.precision),
            avgStd(self.sensitivity),
            avgStd(self.f1_score),
            avgStd(self.iou),
            avgStd(self.voe),
            avgStd(self.rve),
        )

    # ===== 导出每个样本的 Dice 到 CSV =====
    def per_case_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.case_records, columns=["name", "dice"])

    def save_per_case_csv(self, path):
        df = self.per_case_dataframe()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")


# class Metrics(nn.Module):
#     """
#     Macro（按样本等权平均）的分割指标：
#     先对每个样本算 precision / recall / Dice(F1) / IoU / VOE(=1-IoU, 越小越好) / RVE(↑=1-|Vp-Vg|/Vg)
#     再对所有样本取均值作为最终指标。
#     同时支持导出每个样本的 Dice：name -> dice(%)
#     """
#     def __init__(self, threshold=0.5, eps=1e-8, debug=False):
#         super().__init__()
#         self.th = threshold
#         self.eps = eps
#         self.debug = debug

#         # —— 每样本指标的收集（用于 macro 平均） —— #
#         self.precision_cases = []
#         self.recall_cases    = []
#         self.dice_cases      = []
#         self.iou_cases       = []
#         self.voe_cases       = []
#         self.rve_cases       = []

#         # 导出/诊断
#         self.case_records = []   # [{'name': str, 'dice': float}, ...]
#         self.batch_sizes = []    # 仅做诊断可选

#     def reset(self):
#         self.precision_cases.clear()
#         self.recall_cases.clear()
#         self.dice_cases.clear()
#         self.iou_cases.clear()
#         self.voe_cases.clear()
#         self.rve_cases.clear()
#         self.case_records.clear()
#         self.batch_sizes.clear()

#     @torch.no_grad()
#     def __call__(self, preds, labels, names=None):
#         # 设备 & 数值 & 形状对齐
#         labels = labels.to(preds.device)
#         preds  = torch.nan_to_num(preds,  nan=0.0, posinf=1.0, neginf=0.0)
#         labels = torch.nan_to_num(labels, nan=0.0)

#         # 常见二值分割：preds=[N,1,...], labels=[N,...] -> 压掉通道维
#         if preds.ndim == labels.ndim + 1 and preds.shape[1] == 1:
#             preds = preds.squeeze(1)

#         # 二值化
#         preds_bin  = (preds  > self.th)
#         labels_bin = (labels >= 0.5)

#         N = preds_bin.shape[0]
#         self.batch_sizes.append(int(N))

#         # 按样本统计
#         dims = tuple(range(1, preds_bin.ndim))
#         tp    = ((preds_bin & labels_bin).sum(dim=dims)).float()      # [N]
#         fp    = ((preds_bin & ~labels_bin).sum(dim=dims)).float()     # [N]
#         fn    = (((~preds_bin) & labels_bin).sum(dim=dims)).float()   # [N]
#         inter = tp                                                    # [N]
#         union = ((preds_bin | labels_bin).sum(dim=dims)).float()      # [N]
#         ps    = (preds_bin.sum(dim=dims)).float()                     # [N]
#         gs    = (labels_bin.sum(dim=dims)).float()                    # [N]

#         # 基本分母
#         den_prec = tp + fp
#         den_rec  = tp + fn
#         den_dice = 2*tp + fp + fn

#         # per-case 指标（浮点）
#         precision_i = tp / (den_prec + self.eps)                      # [N]
#         recall_i    = tp / (den_rec  + self.eps)                      # [N]
#         dice_i      = 2*tp / (den_dice + self.eps)                    # [N]
#         iou_i       = inter / (union + self.eps)                      # [N]

#         # “预测与GT都空”视为完美匹配：Dice=1, IoU=1
#         empty_both = (gs == 0) & (ps == 0)
#         iou_i[empty_both]  = 1.0
#         dice_i[empty_both] = 1.0

#         voe_i = 1.0 - iou_i

#         # RVE：|Vp-Vg|/Vg -> 转“越大越好”= 1-RVE，空GT时：pred也空=>1，否则0
#         rve_good = torch.empty_like(gs, dtype=torch.float)
#         has_gt   = (gs > 0)
#         rve_good[has_gt] = (1.0 - (ps[has_gt] - gs[has_gt]).abs() / (gs[has_gt] + self.eps)).clamp(0, 1)
#         rve_good[~has_gt] = (ps[~has_gt] == 0).float()

#         # 收集到 Python 列表（0~1）
#         self.precision_cases.extend(precision_i.detach().cpu().tolist())
#         self.recall_cases.extend(recall_i.detach().cpu().tolist())
#         self.dice_cases.extend(dice_i.detach().cpu().tolist())
#         self.iou_cases.extend(iou_i.detach().cpu().tolist())
#         self.voe_cases.extend(voe_i.detach().cpu().tolist())
#         self.rve_cases.extend(rve_good.detach().cpu().tolist())

#         # 记录每个样本的 dice(%) + 名称
#         dice_pct = (dice_i * 100).clamp(0, 100).detach().cpu().tolist()
#         if names is not None:
#             if torch.is_tensor(names):
#                 names = [str(x) for x in names]
#             else:
#                 names = list(names)
#             for nm, d in zip(names[:len(dice_pct)], dice_pct[:len(names)]):
#                 self.case_records.append({"name": str(nm), "dice": float(d)})
#         else:
#             # 没名字也可以只存 dice
#             for d in dice_pct:
#                 self.case_records.append({"name": "", "dice": float(d)})

#     def evluation(self, fold):
#         # —— macro 平均（按样本等权） —— #
#         def _mean(x):
#             return float(np.mean(x)) if len(x) else 0.0

#         precision   = _mean(self.precision_cases)
#         sensitivity = _mean(self.recall_cases)
#         f1          = _mean(self.dice_cases)
#         iou         = _mean(self.iou_cases)
#         voe         = _mean(self.voe_cases)
#         rve_good    = _mean(self.rve_cases)

#         # 兼容 Writer/avgStd（仍用“单值列表”承载百分制）
#         self.precision   = [round(precision*100, 4)]
#         self.sensitivity = [round(sensitivity*100, 4)]
#         self.f1_score    = [round(f1*100, 4)]
#         self.iou         = [round(iou*100, 4)]
#         self.voe         = [round(voe*100, 4)]
#         self.rve         = [round(rve_good*100, 4)]

#         logs(
#             f"Fold {fold}"
#             f", Precision : " + avgStd(self.precision, log=True) +
#             f", Sensitivity: " + avgStd(self.sensitivity, log=True) +
#             f", Dsc: "       + avgStd(self.f1_score, log=True) +
#             f", IoU: "       + avgStd(self.iou, log=True) +
#             f"\n, VOE(↓): "  + avgStd(self.voe, log=True) +
#             f", RVE(↑): "    + avgStd(self.rve, log=True)
#         )

#         return (
#             avgStd(self.precision),
#             avgStd(self.sensitivity),
#             avgStd(self.f1_score),
#             avgStd(self.iou),
#             avgStd(self.voe),
#             avgStd(self.rve),
#         )

#     # ===== 导出每个样本的 Dice 到 CSV =====
#     def per_case_dataframe(self):
#         import pandas as pd
#         return pd.DataFrame(self.case_records, columns=["name", "dice"])

#     def save_per_case_csv(self, path):
#         df = self.per_case_dataframe()
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         df.to_csv(path, index=False, encoding="utf-8-sig")
