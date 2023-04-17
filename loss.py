import torch
from torch import Tensor
from torch.nn import Module

import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.5,gamma=2.0,reduce='mean'):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self,classifications,targets):  
        alpha = self.alpha
        gamma = self.gamma
        
        classifications = classifications.contiguous().view(-1)  
        targets = targets.contiguous().view(-1)                  
        
        ce_loss = F.binary_cross_entropy_with_logits(classifications, targets.float(), reduction="none")
        #focal loss
        p = torch.sigmoid(classifications)                
        p_t = p * targets + (1 - p) * (1 - targets)       
        loss = ce_loss * ((1 - p_t) ** gamma)             
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets) 
            loss = alpha_t * loss                         
        if self.reduce=='sum':
            loss = loss.sum()
        elif self.reduce=='mean':
            loss = loss.mean()
        else:
            raise ValueError('reduce type is wrong!')
        return loss

class MaskedBMLoss(Module):

    def __init__(self, loss_fn: Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        loss = []
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :, :frame], true[i, :, :frame]))
        return torch.mean(torch.stack(loss))


class MaskedFrameLoss(Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = FocalLoss()

    def forward(self, pred: Tensor, true: Tensor, n_frames: Tensor):
        # input: (B, T)
        loss = []
        
        for i, frame in enumerate(n_frames):
            loss.append(self.loss_fn(pred[i, :frame], true[i, :frame]))
        return torch.mean(torch.stack(loss))


class MaskedContrastLoss(Module):

    def __init__(self, margin: float = 0.99):
        super().__init__()
        self.margin = margin

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor, n_frames: Tensor):
        # input: (B, C, T)
        loss = []
        for i, frame in enumerate(n_frames):
            # mean L2 distance squared
            d = torch.dist(pred1[i, :, :frame], pred2[i, :, :frame], 2)
            if labels[i]:
                # if is positive pair, minimize distance
                loss.append(d ** 2)
            else:
                # if is negative pair, minimize (margin - distance) if distance < margin
                loss.append(torch.clip(self.margin - d, min=0.) ** 2)
        return torch.mean(torch.stack(loss))
