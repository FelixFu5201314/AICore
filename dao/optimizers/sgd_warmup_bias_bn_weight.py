#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520

"""
torch.optim.SGD 及其变形
"""
import torch.nn as nn
from torch.optim import SGD

from dao.register import Registers


@Registers.optims.register
def sgd_warmup_bias_bn_weight(model=None,
                              lr=0.01,
                              weight_decay=1e-4,
                              momentum=0.9,
                              warmup_lr=0,
                              warmup_epoch=5
                              ):
    """
    model:torch.nn.Module 此trainer的self.model属性
    lr: float 对于整个（多机多卡）batch size的学习率
    weight_decay:float torch.optim.SGD 默认参数
    momentum:float torch.optim.SGD 默认参数
    warmup_lr:float warmup时的学习率
    warmup_epoch:int warmup几个epoch
    """
    if warmup_epoch > 0:
        lr = warmup_lr
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    # 优化器 SGD + nesterov + momentum
    optimizer = SGD(pg0, lr=lr, momentum=momentum, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay":weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})
    return optimizer
