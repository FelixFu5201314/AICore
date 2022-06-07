# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
import numpy as np
import torch
import torch.nn as nn

from dao.register import Registers


@Registers.losses.register
def BCELoss(weight=None, reduction='mean'):
    if weight is not None:
        weight = torch.from_numpy(np.array(weight))
    return nn.BCELoss(weight=weight,  reduction=reduction)


if __name__ == "__main__":
    m = nn.Sigmoid()
    loss = BCELoss()
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(m(input), target)
    print(output)
    output.backward()
