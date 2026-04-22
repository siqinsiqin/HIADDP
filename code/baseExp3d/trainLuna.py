# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

from configs import config
from utils.trainBase import trainBase


class trainLuna(trainBase):

    def __init__(self, model2d, model3d, lossList):
        super(trainLuna, self).__init__(model2d, model3d, lossList)
        if self.mode == '2d':
            self.seg_path = self.seg_path_luna_2d
        else:
            self.seg_path = self.seg_path_luna_3d
        self.run(True)


if __name__ == '__main__':
    """ top         
    cmd 命令                  
    conda activate jwj    
    cd /zsm/jwj/baseExpV7/     
    nohup nice -n 10 python trainLuna.py >/dev/null 2>&1 &
            
    conda activate jwj      
    cd /zljteam/jwj/baseExpV7/  
    nohup nice -n 10 python trainLuna.py >/dev/null 2>&1 &
    

    进程查看 ps aux | grep 27724                
    杀死所有进程 pkill python          105416    
    """
    shuzhibiao = False
    if shuzhibiao:
        while 1:
            #  3d  zsm
            #  3D  zlj , 3d  luna 57051, lidc 221378
            loss_lists = ['dice', ]
            model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', 'sgunet', 'bionet',
                    'uctransnet', 'utnet', 'swinunet', 'unext']
            model2d = ['dualb']
            model2d = ['unet']

            # model3d = ['unetr', 'pcamnet', 'asa', 'unet', 'unetpp', 'wingsnet', 'ynet', 'reconnet']  # 'transbts',
            # model3d = ['unet', 'vnet']  # u2pcsnet

            # model3d = []
            model3d = ['dualbasa']  # dualbd5zsc
            # model3d = ['unetr', 'asa']  # 'unet', 'unetpp', 'reconnet', 'pcamnet',
            # dualbd2 'dualbd3', 'dualbd4' dualbd5, dualbd5df, dualbd5zs, dualbd5cs, dualbd5zc
            # model3d = ['dualbasa']  # dualbca, dualbasa, dualbsp
            trainLuna(model2d, model3d, loss_lists).to(config.device)
    else:
        loss_lists = ['dice', ]
        model2d = ['unet']
        model3d = ['dualbasa']
        trainLuna(model2d, model3d, loss_lists).to(config.device)

