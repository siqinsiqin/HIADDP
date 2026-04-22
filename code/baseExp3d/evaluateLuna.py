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
    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', '    sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    
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
    evaluateLuna(model3d, None).to(config.device)  # 整体评估，读入全部数据
    #
    # for labels in getAllAttrs(True).values():  # todo 分项评估
    #     evaluateLuna(model3d, labels).to(config.device)

    # 获取所有标签
    #     all_labelss = getAllAttrs(True).values()
    #
    all_labels = []  # 同时运行多个实例
    #     #
    # #     ['ExtremelySubtle', 'ModeratelySubtle', 'FairlySubtle', 'ModeratelyObvious', 'Obvious']  #
    # #     ['OvoidLinear', 'Ovoid', 'OvoidRound', 'Round']
    # #     ['PoorlyDefined', 'NearPoorlyDefined', 'MediumMargin', 'NearSharp', 'Sharp']
    # #     ['NoLobulation', 'NearlyNoLobulation', 'MediumLobulation', 'NearMarkedLobulation',
    # #                       'MarkedLobulation']
    # #     ['NoSpiculation', 'NearlyNoSpiculation', 'MediumSpiculation', 'NearMarkedSpiculation',
    # #                        'MarkedSpiculation']
    # #     ['NonSolidGGO', 'NonSolidMixed', 'PartSolidMixed', 'SolidMixed', 'tSolid']
    # #     ['benign', 'uncertain', 'malignant']
    # #     ['sub36', 'sub6p', 'solid36', 'solid68', 'solid8p']
    #
    # for i in range(2):
    #     all_labels.append(f'ExtremelySubtle')
    #
    # # 创建线程列表
    # threads = []
    # for labels in all_labels:
    #     thread = threading.Thread(target=evaluateLuna, args=(model3d, labels))
    #     threads.append(thread)
    #     thread.start()
    #
    # # # 等待所有线程完成
    # for thread in threads:
    #     thread.join(1)
    #
    # print(all_labels)
