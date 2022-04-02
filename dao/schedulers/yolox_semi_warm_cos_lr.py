#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
import math
from functools import partial
from dao.register import Registers


@Registers.schedulers.register
class yolox_semi_warm_cos_lr:
    def __init__(self, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - warmup_epochs
                - warmup_lr_start (default 1e-6)
                - min_lr_ratio
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
        warmup_lr_start = getattr(self, "warmup_lr_start", 0)
        min_lr_ratio = getattr(self, "min_lr_ratio", 0.2)
        warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
        no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
        normal_iters = self.iters_per_epoch * self.semi_epoch
        semi_iters = self.iters_per_epoch_semi * (
                self.total_epochs - self.semi_epoch - self.no_aug_epochs
        )
        lr_func = partial(
            yolox_semi_warm_cos_lr,
            self.lr,
            min_lr_ratio,
            warmup_lr_start,
            self.total_iters,
            normal_iters,
            no_aug_iters,
            warmup_total_iters,
            semi_iters,
            self.iters_per_epoch,
            self.iters_per_epoch_semi,
        )
        return lr_func


def fun_lr(
    lr,
    min_lr_ratio,
    warmup_lr_start,
    total_iters,
    normal_iters,
    no_aug_iters,
    warmup_total_iters,
    semi_iters,
    iters_per_epoch,
    iters_per_epoch_semi,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= normal_iters + semi_iters:
        lr = min_lr
    elif iters <= normal_iters:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iters)
            )
        )
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (
                    normal_iters
                    - warmup_total_iters
                    + (iters - normal_iters)
                    * iters_per_epoch
                    * 1.0
                    / iters_per_epoch_semi
                )
                / (total_iters - warmup_total_iters - no_aug_iters)
            )
        )
    return lr

