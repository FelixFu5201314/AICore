# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:https://blog.csdn.net/baidu_36511315/article/details/105217674

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from dao.register import Registers


# Dice系数
def diceCoeff(pred, gt, epsilon=1, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 *intersection + epsilon) / (unionset + epsilon)

    return loss.sum() / N


@Registers.losses.register
class DiceLoss_DAO(_Loss):
    __name__ = 'dice_loss'

    def __init__(self, num_classes=1, reduction='mean', epsilon=1e-5, activation='sigmoid'):
        super(DiceLoss_DAO, self).__init__()
        self.epsilon = epsilon
        self.activation = activation
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), "the size of predict and target must be equal."
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation, epsilon=self.epsilon))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice


if __name__ == "__main__":
    # 1. ---------------第一种情况：预测和标签完全一样
    # shape = torch.Size([1, 3, 4, 4])
    '''
    1 0 0= bladder
    0 1 0 = tumor
    0 0 1= background 
    '''
    smooth_dice = DiceLoss_DAO(num_classes=3, activation="none")

    pred = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [1, 0, 0, 1]]]])
    gt = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [1, 0, 0, 1]]]])
    print('预测和标签完全一样,dice={:.4}'.format(1-smooth_dice(pred, gt)))

    pred = torch.Tensor([[
        [[0, 1, 1, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]]])
    gt = torch.Tensor([[
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]]])
    print('预测和标签不完全一样,dice={:.4}'.format(1 - smooth_dice(pred, gt)))
