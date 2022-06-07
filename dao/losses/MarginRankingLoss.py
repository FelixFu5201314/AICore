# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss

import numpy as np

import torch

from dao.register import Registers


@Registers.losses.register
def MarginRankingLoss_DAO(margin=0, reduction="mean"):
    return torch.nn.MarginRankingLoss(margin=margin, reduction=reduction)


if __name__ == "__main__":
    loss = MarginRankingLoss_DAO()
    input1 = torch.randn(3, requires_grad=True)
    input2 = torch.randn(3, requires_grad=True)
    target = torch.randn(3).sign()
    output = loss(input1, input2, target)
    output.backward()
