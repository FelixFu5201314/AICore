#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["ModelEMA", "is_parallel", "EMA"]


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.

    模型的指数移动平均可以提升模型的鲁棒性和性能。
    其原理是：对模型额外维护一份指数移动平均模型 ema_model,
    然后在每次迭代模型参数更新后，利用 model 中的参数和 ema_model 计算更新后的ema_model,
    评估阶段使用的是ema_model
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        # 在模型运行前，将模型参数重新copy一份
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def get_weights_copy(self, model):
        weights_path = 'weights_temp.pt'
        torch.save(model.state_dict(), weights_path)
        return torch.load(weights_path)


class EMA:
    def __init__(self, model, decay):
        self.model = model.module if is_parallel(model) else model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}