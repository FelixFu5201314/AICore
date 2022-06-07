# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
import numpy as np
import torch
import torch.nn as nn

from dao.register import Registers


@Registers.losses.register
def BCEWithLogitsLoss_DAO(pos_weight=None, reduction='mean'):
    return nn.BCEWithLogitsLoss(weight=pos_weight,  reduction=reduction)


if __name__ == "__main__":
    target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
    output = torch.full([10, 64], 1.5)  # A prediction (logit)
    pos_weight = torch.ones([64])  # All weights are equal to 1
    criterion = BCEWithLogitsLoss_DAO(pos_weight=pos_weight)
    criterion(output, target)  # -log(sigmoid(1.5))
