#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = mosaic

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.mosaic, idx) for idx in batch]


class BatchSampler(torchBatchSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    在训练中，我们只关心训练数据的“无限流”。
    所以这个sampler产生了无限的索引流，所有workers共同协作，正确shuffle indices和采样不同的indices。
    每个worker中的sample有效地生成' indices[worker_id::num_workers] '
    其中“indices”是无限的索引流，由shuffle(range(size)) + shuffle(range(size)) +…(如果shuffle是真的)
    或者' range(size) + range(size) +…(如果shuffle是假的)
    """
    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size  # 数据总数
        assert size > 0
        self._shuffle = shuffle     # 是否打乱数据的顺序
        self._seed = int(seed)      # 随机数

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        """
        实现了__iter__方法的对象是可迭代的
        :return:
        """
        start = self._rank
        # itertools.islice(iterable, start, stop[,step])
        # 可以返回从迭代器中的start位置到stop位置的元素，如果stop为None，则一直迭代到最后位置。
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size
