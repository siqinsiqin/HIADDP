# --coding:utf-8 --
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

from configs import config
from utils.evaluateBase import evaluateBase
from utils.helper import getAllAttrs, seed_torch


class evaluateLuna(evaluateBase):

    def __init__(self, model_lists, labels):
        super(evaluateLuna, self).__init__(model_lists)
        self.pth_path = config.pth_luna_path
        self.dataset = 'luna'
        if self.mode == '2d':
            self.seg_path = config.seg_path_luna_2d
        else:
            self.seg_path = config.seg_path_luna_3d

        self.run(labels)


if __name__ == '__main__':
    config.train = False

    """     
    cmd 命令
    conda activate jwj  
    cd /zsm/jwj/baseExpV7/ 
       
    cd /zljteam/jwj/baseExpV7/
    
    nohup nice -n 10 python evaluateLuna.py >/dev/null 2>&1 &
    """

    # todo 2d
    # todo 3d 219404
    loss_lists = ['dice']  # dice, 'bce', 'focal'
               # 'uctransnet', 'utnet', 'swinunet', 'unext']
    
    model3d = ['unet', 'ynet', 'unetpp', 'reconnet', 'transbts', 'wingsnet',
               'unetr', 'pcamnet', 'asa', ]  # 'vtunet', 'resunet',

    # model3d = []  # 'dualb'

    model3d = ['pcamnet', 'reconnet', 'unetr', 'asa']  # , 'asa' 'unet', 'unetpp',
    # model3d = ['dualbd2', 'dualbd3', 'dualbd4', 'dualbd5', ]  # dualbd5df
    # # model3d = ['dualbd5zs', ]  # dualbd5df
    # model3d = ['dualbd5df']  # dualbsp dualbca/ dualbd5zs 'unetr',
    model3d = ['dualbasa', ]  # dualbsp dualbca/ dualbd5zs
    #  61.35
    model2d = ['unet']
    mode = config.mode  #
    # seed_torch(2)
    evaluateLuna(model2d, None ).to(config.device)  # 整体评估，读入全部数据
    #
    # for labels in getAllAttrs(True).values():  # todo 分项评估
    #     evaluateLuna(model3d, labels).to(config.device)

    # 获取所有标签
    #     all_labelss = getAllAttrs(True).values()
    #
    all_labels = []  # 同时运行多个实例
