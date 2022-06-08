# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dao.losses import DiceLoss, BCELoss
from dao.register import Registers


@Registers.losses.register
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, epsilon=1e-5, activation="sigmoid"):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(epsilon=epsilon, activation='sigmoid')
        self.bce_loss = BCELoss(weight=weight)
        if activation is None or activation == "none":
            activation_fn = lambda x: x
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation == "softmax2d":
            activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
        self.activation_fn = activation_fn

    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets) + self.bce_loss(self.activation_fn(inputs), targets)


if __name__ == "__main__":
    N, C, H, W = 2, 3, 4, 4
    loss = DiceBCELoss()
    # 1. 测试分割
    input = torch.randn((N, C, H, W), requires_grad=True)
    target = torch.empty((N, C, H, W)).random_(C)
    output = loss(input, target)
    print("input:{}".format(input))
    print("target:{}".format(target))
    print("output:{}".format(output))
    torch.mean(output).backward()

    # 3. 测试分割(自定义数据)
    input = [[
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
         [1, 0, 0, 1]]]]
    target = [[
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
         [1, 0, 0, 1]]]]
    input = torch.tensor(input, dtype=torch.float32, requires_grad=True)
    target = torch.tensor(target, dtype=torch.float32)
    output = loss(input, target)
    print("input:{}".format(input))
    print("target:{}".format(target))
    print("output:{}".format(output))
    torch.mean(output).backward()

