import os

import numpy as np
import torch
from torch import nn

from utils.Metrics import logs


class EarlyStopping(nn.Module):
    """EarlyStopping class for monitoring the validation loss and stopping training when it does not improve"""

    def __init__(self, fold, patience=7, mode='2d', verbose=False, delta=0.0, path='./'):
        """
        Initialize the EarlyStopping class
         Parameters:
        fold (int): The fold number of the current training
        patience (int): The number of epochs to wait for improvement before stopping
        mode (str): The mode of the model ('2d' or '3d')
        verbose (bool): Whether to print out the best epoch and loss
        delta (float): The minimum improvement in validation loss to consider an improvement
        path (str): The path to save the model checkpoints
        """
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.fold = fold
        self.epoch = 0
        self.last_epoch = 0
        self.mode = mode
        os.makedirs(self.path, exist_ok=True, mode=0o777)

    def __call__(self, val_loss, model, epoch, model_name, optimizer, loss_name):
        """
        Call the EarlyStopping class
         Parameters:
        val_loss (float): The validation loss of the current epoch
        model (nn.Module): The model object
        epoch (int): The current epoch
        model_name (str): The name of the model
        optimizer (str): The name of the optimizer
        loss_name (str): The name of the loss
        """
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, model, model_name, optimizer, loss_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epoch = epoch
            self.counter = 0
            self.save_checkpoint(val_loss, model, model_name, optimizer, loss_name)
            
            

    def save_checkpoint(self, val_loss, model, model_name, optimizer, loss_name):
        """
        Save the checkpoint of the model
         Parameters:
        val_loss (float): The validation loss of the current epoch
        model (nn.Module): The model object
        model_name (str): The name of the model
        optimizer (str): The name of the optimizer
        loss_name (str): The name of the loss
        """
        if self.verbose:
            logs(
                f'*************************Best Epoch {int(self.epoch)}, dice increased ({self.val_loss_min:.2f} --> {val_loss:.2f}).*********************************')

        torch.save(model.state_dict(),
                   self.path + f'/{self.mode}_{model_name}_{str(self.fold)}_{optimizer}_{loss_name}_checkpoint.pth')
        self.val_loss_min = val_loss
