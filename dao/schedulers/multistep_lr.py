#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
import math
import math
from functools import partial
from dao.register import Registers


@Registers.schedulers.register
class multistep_lr:
    def __init__(self, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - milestones (epochs)
                - gamma (default 0.1)
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
        milestones = [
            int(self.total_iters * milestone / self.total_epochs)
            for milestone in self.milestones
        ]
        gamma = getattr(self, "gamma", 0.1)
        lr_func = partial(fun_lr, self.lr, milestones, gamma)
        return lr_func


def fun_lr(lr, milestones, gamma, iters):
    """MultiStep learning rate"""
    for milestone in milestones:
        lr *= gamma if iters >= milestone else 1.0
    return lr
