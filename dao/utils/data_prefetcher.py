#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcherPath(object):
    def __init__(self, loader, device, stop_after=None):
        """
        self.train_loader = DataPrefetcher(train_loader, device=self.device)
        """
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.next_image_path = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_image_path = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_image_path = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)
            self.next_image_path = self.next_image_path

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            image_path = self.next_image_path
            self.preload()
            count += 1
            yield input, target, image_path
            if type(self.stop_after) is int and (count > self.stop_after):
                break


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.

    在读取每次数据喂给网络的时候，预读取下一次迭代需要的数据
    用于cpu-》gpu提速：
    默认情况下，pytorch将所有涉及到GPU的操作（比如内核操作，cpu->gpu, gpu->cpu)
    都排入同一个stream（default stream）中， 并对同一个流的操作序列化，他们永远不会并行。
    如果想并行，两个操作必须位于不同的stream中。
    而前向传播位于default stream中，要想将一个batch数据的预读取（涉及cpu->gpu) 与当前batch的前向传播并行处理，
    就必须：
    （1）cpu上的数据batch必须pinned；
    （2）预读取操作必须在另外一个stream上进行data prefetch。
    dataloader必须设置pin_memory=True来满足第一个条件
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        # CUDA流表示一个GPU操作队列，该队列中的操作将以添加到流中的先后顺序而依次执行
        # 可以将一个流看作时GPU上的一个任务，不同的任务可以并行执行
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class DataPrefetcherCls:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.

    在读取每次数据喂给网络的时候，预读取下一次迭代需要的数据
    用于cpu-》gpu提速：
    默认情况下，pytorch将所有涉及到GPU的操作（比如内核操作，cpu->gpu, gpu->cpu)
    都排入同一个stream（default stream）中， 并对同一个流的操作序列化，他们永远不会并行。
    如果想并行，两个操作必须位于不同的stream中。
    而前向传播位于default stream中，要想将一个batch数据的预读取（涉及cpu->gpu) 与当前batch的前向传播并行处理，
    就必须：
    （1）cpu上的数据batch必须pinned；
    （2）预读取操作必须在另外一个stream上进行data prefetch。
    dataloader必须设置pin_memory=True来满足第一个条件
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        # CUDA流表示一个GPU操作队列，该队列中的操作将以添加到流中的先后顺序而依次执行
        # 可以将一个流看作时GPU上的一个任务，不同的任务可以并行执行
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_path = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_path = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_path = self.next_path

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        path = self.next_path
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        # if path is not None:
        #     path.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, path

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class DataPrefetcherSeg:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.

    在读取每次数据喂给网络的时候，预读取下一次迭代需要的数据
    用于cpu-》gpu提速：
    默认情况下，pytorch将所有涉及到GPU的操作（比如内核操作，cpu->gpu, gpu->cpu)
    都排入同一个stream（default stream）中， 并对同一个流的操作序列化，他们永远不会并行。
    如果想并行，两个操作必须位于不同的stream中。
    而前向传播位于default stream中，要想将一个batch数据的预读取（涉及cpu->gpu) 与当前batch的前向传播并行处理，
    就必须：
    （1）cpu上的数据batch必须pinned；
    （2）预读取操作必须在另外一个stream上进行data prefetch。
    dataloader必须设置pin_memory=True来满足第一个条件
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        # CUDA流表示一个GPU操作队列，该队列中的操作将以添加到流中的先后顺序而依次执行
        # 可以将一个流看作时GPU上的一个任务，不同的任务可以并行执行
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_path = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_path = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_path = self.next_path

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        path = self.next_path
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        # if path is not None:
        #     path.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, path

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

"""
在读取每次数据喂给网络的时候，预读取下一次迭代需要的数据
用于cpu-》gpu提速：
默认情况下，pytorch将所有涉及到GPU的操作（比如内核操作，cpu->gpu, gpu->cpu)
都排入同一个stream（default stream）中， 并对同一个流的操作序列化，他们永远不会并行。
如果想并行，两个操作必须位于不同的stream中。
而前向传播位于default stream中，要想将一个batch数据的预读取（涉及cpu->gpu) 与当前batch的前向传播并行处理，
就必须：
（1）cpu上的数据batch必须pinned；
（2）预读取操作必须在另外一个stream上进行data prefetch。
dataloader必须设置pin_memory=True来满足第一个条件
"""
class DataPrefetcherDet:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        # CUDA流表示一个GPU操作队列，该队列中的操作将以添加到流中的先后顺序而依次执行
        # 可以将一个流看作时GPU上的一个任务，不同的任务可以并行执行
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.img_info, self.next_id = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_id = None
            self.img_info = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        next_id = self.next_id
        img_info = self.img_info
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, img_info, next_id

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
