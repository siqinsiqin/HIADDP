# -*-coding:utf-8 -*-
"""
# Time       ：2023/9/22 19:37
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import glob

from torch.utils.data import DataLoader

import configs
from configs import config
from demo import compareThick, compareeurThin
from utils.evaluateBase import evaluateBase
from utils.helper import get_model3d, get_model2d, load_model_k_checkpoint
from utils.noduleSet import noduleSet


class evaluateUC4(evaluateBase):

    def __init__(self, model_lists, seg_path, dataset):
        super(evaluateUC4, self).__init__(model_lists)
        if dataset == 'luna':
            self.pth_path = config.pth_luna_path
        else:
            self.pth_path = config.pth_lci_path

        self.dataset = 'luna'
        if self.mode == '2d':
            self.seg_path = config.seg_path_luna_2d
        else:
            self.seg_path = seg_path

        self.run(None)

    def initNetwork(self, k, label):
        if self.mode == '2d':
            model = get_model2d(self.model_name, self.device)
        else:
            model = get_model3d(self.model_name, self.device)
        val_and_test_list = []
        self.seg_path = f'/{configs.server}/jwj/baseExpV7/$segmentation/mthickimg3d/'
        val_and_test_list = glob.glob(self.seg_path + '*.npy')

        ##################
        # 外部测试
        ##################
        # 薄层
        # val_and_test_list.extend(small12)
        # val_and_test_list.extend(small252)
        # val_and_test_list = val_and_test_list[:100]
        ##################
        # 多层厚
        # val_and_test_list.extend(small12)
        # val_and_test_list.extend(small252)
        # val_and_test_list = val_and_test_list[:59]
        # val_and_test_list.extend(greater252)
        # val_and_test_list = val_and_test_list[:100]
        #########
        # 不同层厚验证
        # small1.extend(small12)
        # small25.extend(small252)
        # greater25.extend(greater252)

        # val_and_test_list = small1[:346]
        # val_and_test_list = small25[:346]  # 346
        # val_and_test_list = greater25

        ##########
        # 不同归一化方法
        # small1.extend(small12)
        # small25.extend(small252)
        # greater25.extend(greater252)
        #
        # val_and_test_list.extend(greater25[60:110])
        # val_and_test_list.extend(greater25)

        # val_and_test_list = greater25

        # small1, small25, greater25, alls = compareThick()

        small12, small252, greater252, alls = compareeurThin(6.)
        #
        # small1.extend(small12)
        # small25.extend(small252)
        # greater25.extend(greater252)
        # Sobel算子的有效性验证
        # Sobel avg ,Precision : 81.08±1.59, Sensitivity: 79.20±0.62,DSC: 79.53±1.08,mIou: 82.99±0.70
        # voe, : 66.44±1.38, rve, : 88.77±1.68,
        # df avg ,Precision : 82.39±0.85, Sensitivity: 78.57±0.87,DSC: 79.93±0.44,mIou: 83.24±0.26
        # voe, : 66.92±0.52, rve, : 89.17±1.32,
        # NO Sobel avg ,Precision : 84.08±1.63, Sensitivity: 75.67±2.41,DSC: 79.00±1.14,mIou: 82.67±0.75
        # voe, : 65.79±1.49, rve, : 86.92±2.07,
        # 微小结节有效性验证 461
        # 69.63 69.97
        # 恶性结节
        # 76.96 78.01
        # small1.extend(small25)
        # small1.extend(greater25)
        # # seed_torch()
        # # val_and_test_list = random.sample(greater25, 134)
        # val_and_test_list = small1
        # # 226 #   # 193 # 57
        #   64.24 # 65.43 # 72.36
        # 重采样
        # 250 #
        # 64.62 # 65.36 # 71.28
        # greater25 = random_resampling(250, greater25)

        val_and_test_list = small12
        # # 368 # 306 # 134
        # # 69.60 # 70.47 # 78.48
        # # 70.92 # 70.76 # 78.34
        # mthickimg
        # 213   191     57
        # 68.92 68.72   75.51
        ###############################
        # 不同层厚下模型性能均值表现趋势
        #
        self.val_and_test_batch_size = 1
        val_and_test_dataset = noduleSet(val_and_test_list, ['infer', 'Val'], None, self.show)

        val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                       num_workers=self.num_worker, pin_memory=True, shuffle=True, drop_last=True)

        load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer, self.loss_name, model, k)
        return val_and_test_iter, model


if __name__ == '__main__':
    config.train = False

    """  
    cmd 命令      
    conda activate jwj      
    cd /zsm/jwj/baseExpV7/ 

    cd /zljteam/jwj/baseExpV7/
            
    nohup nice -n 10 python evaluateUC4.py >/dev/null 2>&1 &
    """
    # set_random_seeds(10)
    # todo 2d
    # todo 3d 219404
    loss_lists = ['dice']  # dice, 'bce', 'focal'
    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', '    sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    model3d = ['unet', 'ynet', 'unetpp', 'reconnet', 'transbts', 'wingsnet',
               'unetr', 'pcamnet', 'asa', ]  # 'vtunet', 'resunet',

    model3d = ['unet', 'unetpp', 'reconnet', 'pcamnet', 'unetr', 'asa']
    # model3d = ['reconnet']
    #
    model3d = ['dualbasa', ]
    ''
    # set_random_seeds(0)
    mode = config.mode  #
    # thinPath = '/zsm/jwj/baseExpV7/$segmentation/thin_nodule/'
    # thinPath = '/zsm/jwj/baseExpV7/$segmentation/thick_nodule/'
    # thinPath = '/zsm/jwj/baseExpV7/$segmentation/thick/'
    thinPath = '/zsm/jwj/baseExpV7/$segmentation/thin/'
    evaluateUC4(model3d, thinPath, 'luna').to(config.device)  # 整体评估，读入全部数据
