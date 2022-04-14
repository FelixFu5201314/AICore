#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
from loguru import logger

import torch.multiprocessing

from dao.dataloaders.augments import get_transformer
from dao.register import Registers
from dao.utils import wait_for_the_master, get_local_rank, get_world_size
from dao.dataloaders.dataloading import DataLoader, worker_init_reset_seed
from dao.dataloaders.samplers import InfiniteSampler, BatchSampler


@Registers.dataloaders.register
def SegDataloaderTrain(is_distributed=False, batch_size=None, num_workers=None, dataset=None, seed=0):
    """
    is_distributed : bool 是否是分布式
    batch_size : int batchsize大小
    num_workers : int 读取数据线程数
    dataset : DotMap 数据集配置
    seed : int 随机种子
    """
    # 获得local_rank
    local_rank = get_local_rank()

    # 多个rank读取dataset
    print("local rank {} start wait_for_the_master".format(get_local_rank()))
    with wait_for_the_master(local_rank):
        train_dataset = Registers.datasets.get(dataset.type)(
            preproc=get_transformer(dataset.transforms.kwargs), **dataset.kwargs)
    print("local rank {}, with wait_for_the master exec finished".format(local_rank))

    # 如果是分布式，batch size需要改变
    if is_distributed:
        batch_size = batch_size // get_world_size()

    # 无限采样器
    sampler = InfiniteSampler(len(train_dataset), seed=seed if seed else 0)

    # batch sampler
    batch_sampler = BatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False
    )

    # dataloader的kwargs配置
    dataloader_kwargs = {"num_workers": num_workers, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler
    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    return train_loader


@Registers.dataloaders.register
def SegDataloaderEval(is_distributed=False, batch_size=None, num_workers=None, dataset=None):
    """
    is_distributed : bool 是否是分布式
    batch_size : int batchsize大小
    num_workers : int 读取数据线程数
    dataset : DotMap 数据集配置
    seed : int 随机种子
    """
    val_dataset = Registers.datasets.get(dataset.type)(
        preproc=get_transformer(dataset.transforms.kwargs),
        **dataset.kwargs
    )
    if is_distributed:
        batch_size = batch_size // get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(val_dataset)

    dataloader_kwargs = {"num_workers": num_workers, "pin_memory": True, "sampler": sampler, "batch_size":batch_size}
    val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)
    return val_loader


if __name__ == "__main__":
    from PIL import Image
    import cv2
    from dotmap import DotMap
    from dao.register import Registers
    from dao.utils import denormalization
    from dao.dataloaders.datasets import SegDataset

    dataloader_c = {
        "type": "SegDataloaderTrain",
        "dataset": {
            "type": "SegDataset",
            "kwargs": {
                "data_dir": "/ai/data/AIDatasets/SemanticSegmentation/4AR6N-L546S-DQSM9-424ZM-N4DZ2/VOC2012_Seg_Aug",
                "image_set": "val.txt",
                "in_channels": 3,
                "input_size": [380, 380],
                "cache": False,
                "image_suffix": ".jpg",
                "mask_suffix": ".png"
            },
            "transforms": {
                "kwargs": {
                    "Resize": {"height": 224, "width": 224, "p": 1},
                    "Normalize": {"mean": [0.398993, 0.431193, 0.452234], "std": [0.285205, 0.273126, 0.276610], "p": 1}
                }
            }
        },
        "kwargs": {
            "num_workers": 4,
            "batch_size": 32
        }
    }

    dataloader_c = DotMap(dataloader_c)
    dataloader_train = Registers.dataloaders.get("SegDataloaderTrain")(
        is_distributed=False, dataset=dataloader_c.dataset, **dataloader_c.kwargs)

    print(dataloader_train)
