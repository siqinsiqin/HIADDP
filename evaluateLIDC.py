# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

from torch.utils.data import DataLoader

import configs
from configs import config
from utils.evaluateBase import evaluateBase
from utils.helper import get_model2d, get_model3d, set_init, load_model_k_checkpoint, getAllAttrs
from utils.noduleSet import noduleSet


class evaluateLIDC(evaluateBase):

    def __init__(self, model_lists, labels):
        super(evaluateLIDC, self).__init__(model_lists)

        if configs.dataset == 'lci':
            self.pth_path = config.pth_lci_path
        else:
            self.pth_path = config.pth_lidc_path

        if self.mode == '2d':
            self.seg_path_luna = config.seg_path_luna_2d
            self.seg_path_lidc = config.seg_path_lidc_2d
        else:
            self.seg_path_luna = config.seg_path_luna_3d
            self.seg_path_lidc = config.seg_path_lidc_3d

        self.run(labels)

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

        # lists = set_init(k, self.seg_path_luna, None, lists)
        lists = set_init(k, self.seg_path_lidc, None, lists)
        lists = set_init(k, f'/{config.server}/jwj/baseExpV5/$segmentation/seg_lidc_{config.mode}_rd2/', None,
                         lists)  # 两个以上

        if label is not None:  # 单个标签类型的评估
            val_list = []
            for item in val_and_test_list:
                if item.find(f'_{label}_') != -1:
                    val_list.append(item)
            val_and_test_list = val_list
        # print(len(val_and_test_list))

        # small1, small25, greater25 = compareThick()
        # val_and_test_list = small25[:346]
        if len(val_and_test_list) != 0:
            val_and_test_dataset = noduleSet(val_and_test_list, ['infer', 'Val'], None, self.show)
            val_and_test_iter = DataLoader(val_and_test_dataset, batch_size=self.val_and_test_batch_size,
                                           num_workers=self.num_worker, pin_memory=True, shuffle=False, drop_last=True)
            self.pth_path = config.pth_luna_path

            load_model_k_checkpoint(self.pth_path, self.mode, self.model_name, self.optimizer, self.loss_name, model, k)
            return val_and_test_iter, model
        else:
            return None, None


if __name__ == '__main__':
    loss_lists = ['dice']  # , 'bce', 'focal'
    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', 'sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    model3d = ['unet', 'resunet', 'vnet', 'ynet', 'unetpp', 'reconnet', 'unetr', 'transbts', 'wingsnet', ]  #
    model3d = ['pcamnet', 'vtunet', 'asa']  # 'pcamnet', 'vtunet',

    config.train = False
    """ 
    cmd 命令  
    conda activate jwj     
    
    cd /zsm/jwj/baseExpV7/
    
    cd /zljteam/jwj/baseExpV7/
     
    nohup python evaluateLIDC.py >/dev/null 2>&1 & 
    """

    model3d = ['unet', 'unetpp', 'reconnet', 'pcamnet', 'unetr', 'asa']
    model3d = ['dualbd5df']

    mode = config.mode
    train = config.train  # false

    # evaluateLIDC(model3d, None).to('cuda:0')  # 整体评估

    # model3d = ['vtunet', 'fedcrld', 'asa']  # luna 3d
    for labels in getAllAttrs(True).values():  # todo 分项评估
        evaluateLIDC(model3d, labels).to(config.device)

    # 获取所有标签
    # all_labels = getAllAttrs(True).values()
    # #
    # # 创建线程列表
    # threads = []
    # for labels in all_labels:
    #     thread = threading.Thread(target=evaluateLIDC, args=(model3d, labels))
    #     threads.append(thread)
    #     thread.start()
    #
    # # # 等待所有线程完成
    # for thread in threads:
    #     thread.join(1)
