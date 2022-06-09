# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import numpy as np

import torch
from torch.nn.modules.loss import _Loss

from dao.register import Registers


@Registers.losses.register
class EuclideanDistance(_Loss):
    __name__ = 'p_loss'

    def __init__(self, p=2, reduction="none"):
        """
        Function:计算p距离
        """
        super(EuclideanDistance, self).__init__()
        self.pdist = torch.nn.PairwiseDistance(p=p)
        self.reduction = reduction

    def forward(self, input, target):
        output = self.pdist(input, target)
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            output = torch.mean(output)
        elif self.reduction == "sum":
            output = torch.sum(output)
        return output


if __name__ == "__main__":
    ed = EuclideanDistance()
    input1 = [[1,0],[3,4]]
    input2 = [[0,0],[0,0]]
    input1 = torch.tensor(input1, requires_grad=True, dtype=torch.float32)
    input2 = torch.tensor(input2, requires_grad=True, dtype=torch.float32)
    output = ed(input1, input2)
    print(output)
    torch.mean(output).backward()

