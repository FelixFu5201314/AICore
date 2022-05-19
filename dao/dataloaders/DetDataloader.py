#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
from loguru import logger
import torch.multiprocessing

from dao.register import Registers
from dao.dataloaders.augments import get_transformer
from dao.dataloaders.augments import TrainTransform, ValTransform
from dao.dataloaders.dataloading import DataLoader, worker_init_reset_seed
from dao.dataloaders.samplers import InfiniteSampler, YoloBatchSampler
from dao.utils import wait_for_the_master, get_local_rank, get_world_size


@Registers.dataloaders.register
def DetDataloaderTrain(is_distributed=False, batch_size=None, num_workers=None, dataset=None,
                       seed=0, no_aug=False):
    """
    is_distributed : bool 是否是分布式
    batch_size : int batchsize大小
    num_workers : int 读取数据线程数
    dataset : DotMap 数据集配置
    seed : int 随机种子
    """
    # 获得local_rank
    local_rank = get_local_rank()

    # 多个rank读取VOCDetection
    with wait_for_the_master(local_rank):
        dataset_Det = Registers.datasets.get(dataset.dataset1.type)(
            preproc=TrainTransform(**dataset.dataset1.transforms.kwargs),
            **dataset.dataset1.kwargs)
    dataset_Det = Registers.datasets.get(dataset.dataset2.type)(
            dataset_Det,
            preproc=TrainTransform(**dataset.dataset2.transforms.kwargs),
            **dataset.dataset2.kwargs)
    # 如果是分布式，batch size需要改变
    if is_distributed:
        batch_size = batch_size // get_world_size()

    # 无限采样器
    sampler = InfiniteSampler(len(dataset_Det), seed=seed if seed else 0)

    # batch sampler
    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False,
        mosaic=not no_aug,
    )

    # dataloader的kwargs配置
    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True
    }
    dataloader_kwargs["batch_sampler"] = batch_sampler

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    train_loader = DataLoader(dataset_Det, **dataloader_kwargs)
    return train_loader
    # max_iter = len(train_loader)
    # logger.info("init prefetcher, this might take one minute or less...")
    # # to solve https://github.com/pytorch/pytorch/issues/11201
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # train_loader = DataPrefetcherDet(train_loader)
    # return train_loader, max_iter


@Registers.dataloaders.register
def DetDataloaderEval(is_distributed=False, batch_size=None, num_workers=None, dataset=None):
    """
    is_distributed : bool 是否是分布式
    batch_size : int batchsize大小
    num_workers : int 读取数据线程数
    dataset : DotMap 数据集配置
    seed : int 随机种子
    """
    valdataset = Registers.datasets.get(dataset.type)(
        preproc=ValTransform(**dataset.transforms.kwargs),
        **dataset.kwargs
    )
    if is_distributed:
        batch_size = batch_size // get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = batch_size
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
    return val_loader, len(val_loader)
    # return DataPrefetcherSeg(val_loader), len(val_loader)
