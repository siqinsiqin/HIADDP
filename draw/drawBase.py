# -*-coding:utf-8 -*-
"""
# Time       ：2022/6/15 20:51
# Author     ：comi
# version    ：python 3.8
# Description：
todo 得到2d 或者3d的分割结果
"""
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from skimage.measure import find_contours

from configs import config
from utils.helper import get_model2d, load_model_k_checkpoint, set_init, get_model3d, sobel_filter


class DrawBase(nn.Module):

    def __init__(self, models, dataset='luna', losses=None, mode='2d', tc=False):
        super(DrawBase, self).__init__()
        if losses is None:
            losses = ['dice', 'bce', 'focal']
            losses = ['dice']
        self.models = models
        self.dataset = dataset
        print(self.dataset)
        self.losses = losses
        self.mode = mode
        self.fold = 3
        self.tc = tc
        if self.dataset == 'luna':
            if self.mode == '2d':
                self.img_path_luna = config.seg_path_luna_2d
            else:
                self.img_path_luna = config.seg_path_luna_3d
        else:
            if self.mode == '2d':
                self.img_path_lidc = config.seg_path_lidc_2d
                self.img_path_luna = config.seg_path_luna_2d
            else:
                self.img_path_lidc = config.seg_path_lidc_3d
                self.img_path_luna = config.seg_path_luna_3d

        self.main()

    # def draw2d(self, model_name, noduleType, idx):
    #     colors = ['r', 'c', 'm', 'w']
    #     symbol = ['-', '-', '-', '-']
    #
    #     train_list = []
    #     val_and_test_list = []
    #     lists = [train_list, val_and_test_list]
    #
    #     if self.dataset == 'luna':
    #         set_init(self.fold,self.img_path_luna, None, lists)
    #     elif self.dataset == 'lci':
    #         set_init(self.fold, f'/{config.server}/jwj/baseExpV5/$segmentation/seg_lidc_3d_rd2/', None, lists)
    #         set_init(self.fold, self.img_path_lidc, None, lists)
    #     else:
    #         set_init(self.fold, self.img_path_luna, None, lists)
    #         set_init(self.fold, f'/{config.server}/jwj/baseExpV5/$segmentation/seg_lidc_3d_rd2/', None, lists)
    #         set_init(self.fold, self.img_path_lidc, None, lists)
    #
    #     if noduleType is not None:  # 单个标签类型的评估
    #         val_list = []
    #         for item in val_and_test_list:
    #             if item.find(f'_{noduleType}_') != -1:
    #                 val_list.append(item)
    #         val_and_test_list = val_list
    #
    #     if len(val_and_test_list) != 0:
    #         path = f'{config.pred_path}/{self.dataset}_{noduleType}_result/2d/'
    #         os.makedirs(path, exist_ok=True)
    #
    #         lesion = np.load(val_and_test_list[idx])
    #         img, msk = np.split(lesion, 2, axis=0)
    #         # todo 真实标签
    #         fig, plots = plt.subplots(1, 1)
    #         plots.imshow(img[0], cmap='gray', alpha=1, interpolation='sinc')
    #
    #         plots.axis('off')
    #         for c in find_contours(msk[0].astype(float), 0.5):
    #             plt.plot(c[:, 1], c[:, 0], colors[0], label='Ground Truth', linewidth=2.5)
    #
    #         plt.tight_layout()
    #         # plt.savefig(path + val_and_test_list[0].replace('.npy', '.png').split('/')[-1], bbox_inches="tight", pad_inches=0)
    #         plt.savefig(f'{path}/gt_{noduleType}.png', bbox_inches="tight", pad_inches=0)
    #
    #         # out = sitk.GetImageFromArray(img[0])
    #         # sitk.WriteImage(out, f'{path}/img_{noduleType}.nrrd')
    #
    #         plt.close()
    #         # todo # 预测标签
    #         pred = self.load_model_and_predict(model_name, img)
    #
    #         # todo 叠加三通道
    #         # msks = []
    #         # for i, msk in enumerate(pred):
    #         #     msks.append(np.array(msk[0][0]))
    #         #
    #         # msks = np.stack(msks)
    #         #
    #         # fig, plots = plt.subplots(1, 1)
    #         # plots.imshow(img.T * msks.T)
    #         # plt.show()
    #         # plots.axis('off')
    #         # plt.tight_layout()
    #         # plt.close()
    #         #
    #         # fig, plots = plt.subplots(1, 1)
    #         # plots.imshow(msks.T)
    #         # plots.axis('off')
    #         # plt.tight_layout()
    #         # plt.close()
    #
    #         fig, plots = plt.subplots(1, 1)
    #         plots.imshow(img[0], cmap='gray', alpha=1, interpolation='sinc')
    #         plots.axis('off')
    #
    #         # for k in range(3):
    #         #     for c in find_contours(pred[k][0][0].astype(float), 0.5):
    #         #         plt.plot(c[:, 1], c[:, 0], colors[k + 1], linestyle=symbol[k + 1], label='Prediction')
    #         # todo 只画出了dice loss结果
    #         for c in find_contours(pred[0][0][0].astype(float), 0.5):
    #             plt.plot(c[:, 1], c[:, 0], colors[0], linestyle=symbol[1], label='Prediction', linewidth=2.5)
    #
    #         plt.tight_layout()
    #         # plt.savefig(path + f'pred_' + lesions[0].replace('.npy', '.png').split('/')[-1], bbox_inches="tight",
    #         #             pad_inches=0, dpi=300)
    #         plt.savefig(path + f'pred_{model_name}.png', bbox_inches="tight", pad_inches=0, dpi=300)
    #         plt.close()
    #         print(f'{model_name} {noduleType} is Done !')
    #     else:
    #         print(f'{noduleType} is None')

    def draw3d(self, model_name, noduleType, idx):
        colors = ['w', 'r', 'c', 'm', ]
        symbol = ['-', '-', '-', '-']

        train_list = []
        val_and_test_list = []
        lists = [train_list, val_and_test_list]
        if self.dataset == 'luna':
            set_init(self.fold, self.img_path_luna, None, lists)
        elif self.dataset == 'lci':
            set_init(self.fold, f'/{config.server}/jwj/baseExpV5/$segmentation/seg_lidc_3d_rd2/', None, lists)
            set_init(self.fold, self.img_path_lidc, None, lists)
        else:
            set_init(self.fold, self.img_path_luna, None, lists)
            set_init(self.fold, f'/{config.server}/jwj/baseExpV5/$segmentation/seg_lidc_3d_rd2/', None, lists)
            set_init(self.fold, self.img_path_lidc, None, lists)

        if noduleType is not None:  # 单个标签类型的评估
            val_list = []
            for item in val_and_test_list:
                if item.find(f'_{noduleType}_') != -1:
                    val_list.append(item)
            val_and_test_list = val_list

        if len(val_and_test_list) != 0:
            path = f'{config.pred_path}/{self.dataset}_{noduleType}_result/'
            os.makedirs(path, exist_ok=True)

            name = val_and_test_list[idx]
            serious = name.split('/')[-1].split('_')[0]
            import pylidc as pl
            query = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == serious).all()
            noduleType += '_' + query[0].patient_id
            print(noduleType)
            noduleType = 'test'
            # path = '/zsm/jwj/baseExpV5/$segmentation/seg_luna_3d/1.3.6.1.4.1.14519.5.2.1.6279.6001.111258527162678142285870245028_sub36_ExtremelySubtle_SoftTissue_Absent_Ovoid_NearPoorlyDefined_NoLobulation_NearlyNoSpiculation_NonSolidGGO_uncertain_63.npy'
            # lesion = np.load(path)
            print(name)
            lesion = np.load(name)
            img, msk = np.split(lesion, 2, axis=0)
            import SimpleITK as sitk
            out = sitk.GetImageFromArray(msk[0])
            sitk.WriteImage(out, f'{path}/gt_{noduleType}.nrrd')
            # sitk.WriteImage(out, f'/{config.server}/jwj/baseExpV5/gt_{noduleType}.nrrd')

            # todo 1 thin
            if not self.tc:
                def z_score_norm(img):
                    # z-score
                    img_mean = np.mean(img)  # torch
                    img_std = np.std(img)
                    norm_img = (img - img_mean) / img_std
                    return norm_img

                img = z_score_norm(img)

            fig, plots = plt.subplots(1, 1)
            t = img.shape[3] // 2

            plots.imshow(img[0, :, :, t], cmap='gray', alpha=1, interpolation='sinc')
            plots.axis('off')
            for c in find_contours(msk[0, :, :, t].astype(float), 0.5):
                plt.plot(c[:, 1], c[:, 0], colors[0], label='Ground Truth', linewidth=5)

            plt.tight_layout()
            plt.savefig(f'{path}/gt_{noduleType}.png', bbox_inches="tight", pad_inches=0)

            # todo 2 channel
            if self.tc:
                def z_score_norm(img):
                    # z-score
                    img_mean = torch.mean(img)
                    img_std = torch.std(img)
                    norm_img = (img - img_mean) / img_std
                    return norm_img

                def cut_norm(img):
                    # 截断归一化
                    lungwin = torch.tensor([-1000., 400.])
                    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
                    newimg[newimg < 0] = 0
                    newimg[newimg > 1] = 1
                    return newimg

                img = torch.as_tensor(img.copy()).float().contiguous()
                img_z = z_score_norm(img)

                img_s = cut_norm(img)

                img_sobel = sobel_filter(img_s[0])

                img = np.stack([img_z, img_sobel], axis=1).squeeze(axis=0)

            # # todo # 预测标签
            pred = self.load_model_and_predict(model_name, img)

            msk = pred[0][0]

            fig, plots = plt.subplots(1, 1)

            plots.imshow(img[0, :, :, t], cmap='gray', alpha=1, interpolation='sinc')
            plots.axis('off')
            for c in find_contours(msk[0, :, :, t].astype(float), 0.5):
                plt.plot(c[:, 1], c[:, 0], colors[0], label='Ground Truth', linewidth=5)

            plt.tight_layout()
            plt.savefig(f'{path}/{model_name}_{noduleType}.png', bbox_inches="tight", pad_inches=0)
            import SimpleITK as sitk
            out = sitk.GetImageFromArray(msk[0])

            sitk.WriteImage(out, f'{path}/{model_name}_{noduleType}.nrrd')

            print(f'{model_name} {noduleType} is Done !')
        else:
            print(f'{noduleType} is None')

    def load_model_and_predict(self, model_name, img):
        if self.mode == '2d':
            model = get_model2d(model_name, device=config.device, verbose=False)
        else:
            model = get_model3d(model_name, device=config.device, verbose=False)

        model.eval()

        lesion_img = torch.from_numpy(img[np.newaxis])
        lesion_img = lesion_img.type(torch.FloatTensor)
        lesion_img = lesion_img.to(config.device)

        preds = []
        for loss in self.losses:
            config.model_name = model_name
            # config.pth_luna_path
            # pth = config.pth_luna_path
            if dataset == 'luna':
                pth = config.pth_luna_path
            elif dataset == 'lci':
                pth = config.pth_lci_path
            else:
                pth = config.pth_lidc_path
            load_model_k_checkpoint(pth, self.mode, model_name, 'adam', loss, model, self.fold,
                                    verbose=False)
            pred = model(lesion_img)

            # import SimpleITK as sitk
            # out = pred.cpu().detach().numpy()[0, 0, :, :, 32]
            # out = sitk.GetImageFromArray(out)
            # sitk.WriteImage(out, f'/{config.server}/jwj/baseExpV5/out_x.nrrd')

            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            preds.append(pred.cpu().numpy())

        return preds

    def main(self):

        subtlety = ['ExtremelySubtle', ]  # 3d 1,1 / 2d 2,2  'Obvious'
        sphericity = ['OvoidLinear']  # fold 1 ,2d 1,1 / 3d 1,1 , 'Round' Linear
        margin = ['NearPoorlyDefined']  # 3d 1, 6 /2d 1，15 'NearSharp', 'PoorlyDefined',
        lobulation = ['NearMarkedLobulation', 'NoLobulation', ]  # 3d 20, 1 /2d 19，1
        spiculation = ['MediumSpiculation']  # 3d 3，3 /2d 3,3
        texture = ['NonSolidGGO']  # 3d 10，25/2d 1，29 ，, 'tSolid'
        maligancy = ['malignant']  # 3d -,- /2d 5,14 'malignant',
        size = ['solid8p', ]  # 3d 6,10 /2d 9,9  'solid8p', 'sub36'

        # luna
        # subtlety fold 3，1 LIDC-IDRI-0111
        # spiculation 3，3   LIDC-IDRI-0015
        # maligancy 3，10    LIDC-IDRI-0601
        # sphericity  3,7   LIDC-IDRI-0141

        # lci
        # subtlety fold 3，1  勾勒
        # sphericity  3,0
        # maligancy 3,9
        # spiculation  3，2

        self.fold = 1
        for model in self.models:
            for noduleType in spiculation:
                if self.mode == '2d':
                    self.draw2d(model, noduleType, 1)
                else:
                    self.draw3d(model, noduleType, 2)


if __name__ == '__main__':
    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', 'sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    model3d = ['unet', 'ynet', 'unetpp', 'reconnet', 'unetr', 'transbts', 'wingsnet',
               'pcamnet', 'asa']  # 'resunet', 'vnet', 'vtunet',

    mode = '3d'
    dataset = 'luna'
    # tc = False  # 通道控制.
    tc = False  # 通道控制
    if tc:
        model3d = ['dualbd4', ]
    else:
        model3d = ['unet', 'unetpp', 'reconnet', 'pcamnet', 'unetr', 'asa']
        model3d = ['unet']

    if mode == '2d':
        DrawBase(model2d, dataset=dataset, mode=mode, tc=tc)
    else:
        DrawBase(model3d, dataset=dataset, mode=mode, tc=tc)
