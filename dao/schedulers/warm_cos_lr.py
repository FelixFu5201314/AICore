#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
import math
from functools import partial

from dao.register import Registers


@Registers.schedulers.register
class warm_cos_lr:
    def __init__(self, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - warmup_epochs
                - warmup_lr_start (default 1e-6)
        """
        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs

        self.__dict__.update(kwargs)    # update self.attr, if not add

        self.lr_func = self._get_lr_func()

    def update_lr(self, iters):
        return self.lr_func(iters)

    def _get_lr_func(self):
        warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
        warmup_lr_start = getattr(self, "warmup_lr_start", 1e-6)
        lr_func = partial(
            fun_lr,
            self.lr,
            self.total_iters,
            warmup_total_iters,
            warmup_lr_start,
        )
        return lr_func


def fun_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(
            warmup_total_iters
        ) + warmup_lr_start
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr
