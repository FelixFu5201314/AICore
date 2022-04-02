#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
import math
from functools import partial
from dao.register import Registers


@Registers.schedulers.register
class cos_lr:
    def __init__(self, lr, iters_per_epoch, total_epochs, **kwargs):
        """
         Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - None
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
        lr_func = partial(fun_lr, self.lr, self.total_iters)
        return lr_func


def fun_lr(lr, total_iters, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))  # from 1 -> 0, because iter:0->total_iters
    return lr
