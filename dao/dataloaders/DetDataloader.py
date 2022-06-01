#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520

from loguru import logger

import torch.multiprocessing

from dao.dataloaders.augments import get_transformer, get_transformerYOLO
from dao.dataloaders.augments import TrainTransform, ValTransform
from dao.dataloaders.dataloading import DataLoader, worker_init_reset_seed
from dao.dataloaders.samplers import InfiniteSampler, YoloBatchSampler, BatchSampler
from dao.utils import wait_for_the_master, get_local_rank, get_world_size

from dao.register import Registers


@Registers.dataloaders.register
def DetDataloaderTrain(is_distributed=False, batch_size=None, num_workers=None, dataset=None, seed=0, no_aug=False):
    """
    Function： 目标检测DetDataset的数据加载DataLoader

    :param is_distributed: bool 是否是分布式训练
    :param batch_size: int batchsize大小， 是总batchsize，例如有2个机器，每个有8张卡，batchsize为16，那么每张卡可得到1张图片
    :param num_workers: int 读取数据线程数，每个rank的读取数据的线程数
    :param dataset: DotMap 数据集配置， 详细看configs文件夹下的内容
    :param seed: int 随机种子
    :param no_aug: bool 是否进行数据增强
    :return:
        返回dataloader对象
    """
    # 1. 获得local_rank
    local_rank = get_local_rank()

    # 2. 多个rank读取DetDataset, rank=0先读取，其余等待，rank=0读取后，唤醒其他rank
    with wait_for_the_master(local_rank):
        dataset_Det = Registers.datasets.get(dataset.type)(
            preproc=get_transformerYOLO(dataset.transforms.kwargs),
            preproc_pixel=get_transformer(dataset.transforms.kwargs),
            **dataset.kwargs)

    # 3. 如果是分布式，batch size需要改变。 例如有2个机器，每个有8张卡，batchsize为16，那么每张卡可得到1张图片
    if is_distributed:
        batch_size = batch_size // get_world_size()

    # 4. 无限采样器
    sampler = InfiniteSampler(len(dataset_Det), seed=seed if seed else 0)

    # 5. batch sampler
    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)

    # 6. dataloader的kwargs配置
    dataloader_kwargs = {"num_workers": num_workers, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    # 7. 生成Dataloader类， 具体看dataloading文件内容
    train_loader = DataLoader(dataset_Det, **dataloader_kwargs)
    return train_loader


@Registers.dataloaders.register
def DetDataloaderEval(is_distributed=False, batch_size=None, num_workers=None, dataset=None):
    """
    Function： 目标检测DetDataset的数据加载DataLoader

    :param is_distributed: bool 是否是分布式训练
    :param batch_size: int batchsize大小， 是总batchsize，例如有2个机器，每个有8张卡，batchsize为16，那么每张卡可得到1张图片
    :param num_workers: int 读取数据线程数，每个rank的读取数据的线程数
    :param dataset: DotMap 数据集配置， 详细看configs文件夹下的内容
    :param seed: int 随机种子
    :param no_aug: bool 是否进行数据增强
    :return:
        返回dataloader对象
    """

    # 1. 获取数据集
    valdataset = Registers.datasets.get(dataset.type)(
            preproc=get_transformerYOLO(dataset.transforms.kwargs),
            preproc_pixel=get_transformer(dataset.transforms.kwargs),
            **dataset.kwargs)

    # 2. 如果是分布式，batch size需要改变。 例如有2个机器，每个有8张卡，batchsize为16，那么每张卡可得到1张图片
    if is_distributed:
        batch_size = batch_size // get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(valdataset)

    # 3. dataloader的kwargs配置
    dataloader_kwargs = {"num_workers": num_workers, "pin_memory": True, "batch_size":batch_size}

    # 4. 生成Dataloader类， 具体看dataloading文件内容
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
    return val_loader

