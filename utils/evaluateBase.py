# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os

import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.measure import find_contours
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs
from configs import GC
from utils.Metrics import Metrics
# from utils.MetricsV2 import MetricsV2
from utils.helper import get_model2d, get_model3d, set_init, load_model_k_checkpoint
from utils.logger import logs
from utils.noduleSet import noduleSet
from utils.writer import Writer


# config = config.getConfig()


class evaluateBase(GC):
    """
    评估基类
    """

    def __init__(self, model_lists):
        super(evaluateBase, self).__init__(train=configs.train, dataset=configs.dataset, log_name=configs.log_name,
                                           mode=configs.mode, pathV=configs.pathV, LossV=configs.LossV,
                                           FileV=configs.FileV, MetricsV=configs.MetricsV, sup=configs.sup,
                                           server=configs.server)

        self.seg_path = None
        self.pth_path = None
        self.model_lists = model_lists
        self.loss_lists = ['dice']  # 'dice', 'bce', 'focal'
        logs(f'mode {self.mode}')

    def kFoldMain(self, k, writer, label=None, ):
        val_and_test_iter, model = self.initNetwork(k, label)
        if self.MetricsV == 1:
            if val_and_test_iter is not None:
                fprecision, fsensitivity, ff1, fmIou, voe, rve = self.testfn(k, val_and_test_iter, model)
                writer(fprecision, fsensitivity, ff1, fmIou, voe, rve)
            else:
                writer()
        else:
            if val_and_test_iter is not None:
                dice, hd, msd = self.testfn(k, val_and_test_iter, model)
                writer(dice, hd, msd)
            else:
                writer()

    def toSave(self, model, predGPU, imgGPU, mskGPU, name):

        import os
        import matplotlib.pyplot as plt

        path = f'./predict/{self.dataset}_{model.__class__.__name__}_result/'
        os.makedirs(path, exist_ok=True)

        predcpu = predGPU.cpu().numpy()  
        batch_size = predcpu.shape[0]

        t = 32

        for i in range(batch_size):
            pred = predcpu[i]  
            if pred.ndim == 4:
                mask_slice = pred[0, :, :, t]
            elif pred.ndim == 3:
                mask_slice = pred[0]
            elif pred.ndim == 2:
                mask_slice = pred
            else:
                raise ValueError(f"Unexpected pred ndim: {pred.ndim}, shape: {pred.shape}")

            plt.imsave(os.path.join(path, f'{name[i]}.png'),
               mask_slice,
               cmap='gray')

#     def toSave(self, model, predGPU, imgGPU, mskGPU, name):
#         colors = ['w', 'r', 'c', 'm', ]
#         path = f'./predict/{self.dataset}_{model.__class__.__name__}_result/'
#         os.makedirs(path, exist_ok=True)
#         # to save all test img
#         predcpu = predGPU.cpu().numpy()
#         imgscpu = imgGPU.cpu().numpy()
#         mskscpu = mskGPU.cpu().numpy()
#         for i in range(configs.config.val_and_test_batch_size):
#             pred = predcpu[i]
#             img = imgscpu[i]
#             msk = mskscpu[i]
#             fig, plots = plt.subplots(1, 1)
#             t = 32

#             plots.imshow(img[0, :, :, t], cmap='gray', alpha=1, interpolation='sinc')
#             plots.axis('off')
#             for c in find_contours(msk[0, :, :, t].astype(float), 0.5):
#                 plt.plot(c[:, 1], c[:, 0], colors[0], label='Ground Truth', linewidth=5)

#             plt.tight_layout()
#             plt.savefig(f'{path}/gt_{name[i]}.png', bbox_inches="tight", pad_inches=0)
#             plt.close()
#             out = sitk.GetImageFromArray(msk[0])
#             sitk.WriteImage(out, f'{path}/{name[i]}.nrrd')

#             fig, plots = plt.subplots(1, 1)
#             plots.imshow(img[0, :, :, t], cmap='gray', alpha=1, interpolation='sinc')
#             plots.axis('off')
#             for c in find_contours(pred[0, :, :, t].astype(float), 0.5):
#                 plt.plot(c[:, 1], c[:, 0], colors[0], label='Ground Truth', linewidth=5)

#             plt.tight_layout()
#             plt.savefig(f'{path}/{name[i]}.png', bbox_inches="tight", pad_inches=0)
#             plt.close()

#             out = sitk.GetImageFromArray(pred[0])
#             sitk.WriteImage(out, f'{path}/{name[i]}.nrrd')

    def testfn(self, fold, loader, model):
        model.eval()
        if self.MetricsV == 1:
            metrics = Metrics().to(self.device)
            with torch.no_grad():
                for idx, data in tqdm(enumerate(loader)):
                    img, msk = data['img'], data['msk']
                    names = data.get('name', None) 

                    img = img.type(torch.FloatTensor)
                    msk = msk.type(torch.FloatTensor)

                    if self.device != 'cpu' and torch.cuda.is_available():
                        img = img.cuda(non_blocking=True)
                        msk = msk.cuda(non_blocking=True)

                    pred = model(img)
                    pred = torch.sigmoid(pred)
                    pred_bin = (pred > 0.5).float()

                    if configs.saveshow:
                        self.toSave(model, pred_bin.cpu(), img.cpu(), msk.cpu(), names if names is not None else [])

                    metrics(pred_bin, msk, names=names)

            if configs.tuilitrain:
                import os
                csv_dir = os.path.join(configs.tuilitraincsv)
                os.makedirs(csv_dir, exist_ok=True)
                csv_path = os.path.join(csv_dir, f"{self.dataset}_{self.model_name}_fold{fold}_train_perdice.csv")
                metrics.save_per_case_csv(csv_path)

            fprecision, fsensitivity, ff1, fmIou, voe, rve = metrics.evluation(fold)
            return fprecision, fsensitivity, ff1, fmIou, voe, rve
        else:
            metrics = MetricsV2().to(self.device)
            with torch.no_grad():
                for idx, data in tqdm(enumerate(loader)):

                    img, msk = data['img'], data['msk']
                    img = img.type(torch.FloatTensor)
                    msk = msk.type(torch.FloatTensor)
                    # print(img.shape, msk.shape)
                    if self.device != 'cpu' and torch.cuda.is_available():
                        img, msk = Variable(img.cuda(), requires_grad=False), \
                            Variable(msk.cuda(), requires_grad=False)

                    preds = model(img)
                    preds = torch.sigmoid(preds)
                    preds = (preds > 0.5).float()
                    metrics(preds, msk)

            dice, hd, msd = metrics.evluation(fold)
            return dice, hd, msd

    def initNetwork(self, k, label):
        if self.mode == '2d':
            model = get_model2d(self.model_name, self.device)
        else:
            model = get_model3d(self.model_name, self.device)

        train_list = []
        val_and_test_list = []
        lists = [train_list, val_and_test_list]
        # re = [
        #     "*NonSolidGGO*.npy",
        #     "*NonSolidMixed*.npy",
        #     "*PartSolidMixed*.npy",
        #     "*SolidMixed*.npy",
        #     "*solid*.npy",
        # ]
        # for item in re:
        #     lists = set_init(k, self.seg_path, item, lists)
        if self.FileV == 'npy':
            lists = set_init(k, self.seg_path, None, lists)
        else:
            lists = set_init(k, self.seg_path, None, lists, '*.nii.gz')

        if label is not None:  
            val_list = []
            for item in val_and_test_list:
                if item.find(f'_{label}_') != -1:
                    val_list.append(item)
            val_and_test_list = val_list

        if configs.tuilitrain:
            val_and_test_list = train_list
        
        if len(val_and_test_list) < 12:
            print("len(val_and_test_list) < 12")
            reps = int(np.ceil(12 / len(val_and_test_list)))
            val_and_test_list = (list(val_and_test_list) * reps)[:12]

        if len(val_and_test_list) != 0:
            val_and_test_dataset = noduleSet(val_and_test_list, ['infer', 'Val'], None, self.show)

            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=True, drop_last=False)#11

            load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer, self.loss_name, model, k)
            return val_and_test_iter, model
        else:
            return None, None

    def run(self, labels=None): 
        self.train = False
        if labels is None and self.MetricsV == 1: 
            for model in self.model_lists:
                writer = Writer(self.dataset)
                self.model_name = model
                for loss in self.loss_lists:
                    self.loss_name = loss
                    logs(f'Model {model}, Loss {loss}')
                    if configs.saveshow:
                        print('save png')
                        self.kFoldMain(1, writer, labels, )
                    else:
                        self.kFoldMain(1, writer, labels, )
                    writer(avg=True)
                writer.update(self.model_name)  
                writer.save(self.model_name)
        elif labels is not None and self.MetricsV == 1:
            for model in self.model_lists:
                self.model_name = model
                for label in labels: 
                    writer = Writer(self.dataset)
                    writer.evaluatetype = label
                    for loss in self.loss_lists:
                        self.loss_name = loss
                        logs(f'Model {model}, Loss {loss}, Label {label}')
                        self.kFoldMain(1, writer, label, )
                        writer(avg=True, )
                    writer.update(self.model_name)
                    writer.save(self.model_name)
