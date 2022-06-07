# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss

import numpy as np

import torch

from dao.register import Registers


@Registers.losses.register
def KLDivLoss_DAO(reduction="mean"):
    return torch.nn.KLDivLoss(reduction=reduction)


if __name__ == "__main__":
    from torch.nn import functional as F
    kl_loss = KLDivLoss_DAO(reduction="batchmean")
    # input should be a distribution in the log space
    input = F.log_softmax(torch.randn(3, 5, requires_grad=True))
    # Sample a batch of distributions. Usually this would come from the dataset
    target = F.softmax(torch.rand(3, 5))
    output = kl_loss(input, target)

