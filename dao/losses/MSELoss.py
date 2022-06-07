# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss

import numpy as np

import torch

from dao.register import Registers


@Registers.losses.register
def MSELoss_DAO(reduction="mean"):
    return torch.nn.MSELoss(reduction=reduction)


if __name__ == "__main__":
    loss = MSELoss_DAO()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input, target)
    print(output)
    output.backward()
