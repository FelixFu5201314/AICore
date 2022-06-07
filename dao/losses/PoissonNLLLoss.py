# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss

import numpy as np

import torch

from dao.register import Registers


@Registers.losses.register
def PoissonNLLLoss_DAO(reduction="mean"):
    return torch.nn.PoissonNLLLoss(reduction=reduction)


if __name__ == "__main__":
    loss = PoissonNLLLoss_DAO()
    log_input = torch.randn(5, 2, requires_grad=True)
    target = torch.randn(5, 2)
    output = loss(log_input, target)
    output.backward()
