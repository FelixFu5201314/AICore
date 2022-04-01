
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520

from loguru import logger

import torch
import torch.multiprocessing
from torch import distributed as dist

from .augments import get_transformer
from dao.dataloaders.dataloading import DataLoader, worker_init_reset_seed
from dao.dataloaders.samplers import InfiniteSampler, BatchSampler
from dao.utils import wait_for_the_master, get_local_rank, get_world_size
from dao.register import Registers


@Registers.dataloaders.register
def ClsDataloaderTrain(is_distributed=False, batch_size=None, num_workers=None, dataset=None, seed=0, **kwargs):
    """
    ClsDataset的dataloader类

    is_distributed:bool 是否是分布式
    batch_size: int batchsize大小，多个GPU的batchsize总和
    num_workers:int 使用线程数
    dataset:ClsDataset类 数据集类的实例
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
def ClsDataloaderEval(is_distributed=False, batch_size=None, num_workers=None, dataset=None, **kwargs):
    """
    ClsDataset的dataloader类

    is_distributed:bool 是否是分布式
    batch_size: int batchsize大小，多个GPU的batchsize总和
    num_workers:int 使用线程数
    dataset:ClsDataset类 配置字典
    """
    val_dataset = Registers.datasets.get(dataset.type)(
        preproc=get_transformer(dataset.transforms.kwargs),
        **dataset.kwargs
    )
    if is_distributed:
        batch_size = batch_size // get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader_kwargs = {"num_workers": num_workers, "pin_memory": True, "sampler": sampler, "batch_size": batch_size}

    dataloader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)
    return dataloader


if __name__ == "__main__":
    from dao.dataloaders.augments import get_transformer
    from dotmap import DotMap
    from dao.dataloaders.datasets import ClsDataset  # 导入时，会自动将ClsDataset注册， 所以这句话不能删
    from dao.register import Registers

    dataloader_c = {
        "type": "ClsDataloaderTrain",
        "dataset": {
            "type": "ClsDataset",
            "kwargs": {
                "data_dir": "/ai/data/AIDatasets/ImageClassification/4AR6N-L546S-DQSM9-424ZM-N4DZ2/screen",
                "image_set": "train.txt",
                "in_channels": 1,
                "input_size": [224, 224],
                "cache": False,
                "images_suffix": [".bmp"]
            },
            "transforms": {
                "kwargs": {
                    "Normalize": {"mean": 0, "std": 1, "p": 1}
                }
            }
        },
        "kwargs": {
            "num_workers": 4,
            "batch_size": 256
        }
    }

    dataloader_c = DotMap(dataloader_c)
    dataloader_train, length = Registers.dataloaders.get("ClsDataloader")(
        is_distributed=False, dataset=dataloader_c.dataset, **dataloader_c.kwargs)
    # is_distributed 测试，使用真正的程序，因为需要RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.
    # dataloader_train, length = Registers.dataloaders.get("ClsDataloader")(
    #     is_distributed=True, dataset=dataloader_c.dataset, **dataloader_c.kwargs)
    print(dataloader_train, length)
