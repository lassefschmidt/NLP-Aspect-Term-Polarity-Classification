import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class FocalLoss(nn.Module):

    def __init__(self, alpha = 1, gamma = 2, reduction = 'mean', eps = 1e-8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = "none")

        prob = torch.sigmoid(inputs)
        prob = torch.clamp(prob, min = self.eps, max = 1.0) # avoid vanishing gradients

        pt = torch.where(targets == 1, prob, 1 - prob)

        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        

class TopKLoss(nn.Module):

    def __init__(self, k, reduction='mean', pos_weight = None):
        super(TopKLoss, self).__init__()
        self.k = k # 0 < k <= 100 (if 100, this is simply BCE_Loss)
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = "none", pos_weight = self.pos_weight)
        num_losses = np.prod(BCE_loss.shape, dtype=np.int64) # with batch size 10, we would have shape (10, 3) and num_losses = 30
        TopK_loss, _ = torch.topk(BCE_loss.view((-1, )), int(num_losses * self.k / 100))
        
        if self.reduction == 'mean':
            return torch.mean(TopK_loss)
        elif self.reduction == 'sum':
            return torch.sum(TopK_loss)
        else:
            return TopK_loss