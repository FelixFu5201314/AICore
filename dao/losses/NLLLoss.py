# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss

import numpy as np

import torch

from dao.register import Registers


@Registers.losses.register
def NLLLoss_DAO(weight=None, reduction="mean"):
    if weight is not None:
        weight = torch.from_numpy(np.array(weight))
    return torch.nn.NLLLoss(weight=weight, reduction=reduction)


if __name__ == "__main__":
    m = torch.nn.LogSoftmax(dim=1)
    loss = NLLLoss_DAO()

    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output.backward()

    # 2D loss example (used, for example, with image inputs)
    N, C = 5, 4
    loss = NLLLoss_DAO(weight=[0.1, 1, 1, 1])
    # input is of size N x C x height x width
    data = torch.randn(N, 16, 10, 10)
    conv = torch.nn.Conv2d(16, C, (3, 3))
    m = torch.nn.LogSoftmax(dim=1)
    # each element in target has to have 0 <= value < C
    target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    output = loss(m(conv(data)), target)
    output.backward()