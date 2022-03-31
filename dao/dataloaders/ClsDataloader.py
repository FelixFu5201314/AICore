
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520

from loguru import logger

import torch.multiprocessing

from augments import get_transformer
from utils.dataloading import DataLoader, worker_init_reset_seed
from utils.samplers import InfiniteSampler, BatchSampler
from utils.data_prefetcher import DataPrefetcherCls
from dao.register import Registers


@Registers.dataloaders.register
def ClsDataloaderTrain(
        is_distributed=False,
        batch_size=None,
        num_workers=None,
        dataset=None,
        seed=0,
        **kwargs):
    """
    ClsDataset的dataloader类

    is_distributed:bool 是否是分布式
    batch_size: int batchsize大小，多个GPU的batchsize总和
    num_workers:int 使用线程数
    dataset:ClsDataset类 数据集类的实例
    """
    # 获得local_rank
    local_rank = get_local_rank()

    # 多个rank读取VOCDetection
    with wait_for_the_master(local_rank):
        dataset_CTXT = Registers.datasets.get(dataset.type)(
            preproc=get_transformer(dataset.transforms.kwargs), **dataset.kwargs)

    # 如果是分布式，batch size需要改变
    if is_distributed:
        batch_size = batch_size // get_world_size()

    # 无限采样器
    sampler = InfiniteSampler(len(dataset_CTXT), seed=seed if seed else 0)

    # batch sampler
    batch_sampler = BatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=False
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

    train_loader = DataLoader(dataset_CTXT, **dataloader_kwargs)
    max_iter = len(train_loader)
    logger.info("init prefetcher, this might take one minute or less...")
    # to solve https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataPrefetcherCls(train_loader)
    return train_loader, max_iter


@Registers.dataloaders.register
def ClsDataloaderEval(
        is_distributed=False,
        batch_size=None,
        num_workers=None,
        dataset=None,
        **kwargs):
    """
    ClsDataset的dataloader类

    is_distributed:bool 是否是分布式
    batch_size: int batchsize大小，多个GPU的batchsize总和
    num_workers:int 使用线程数
    dataset:ClsDataset类 数据集类的实例
    """
    valdataset = Registers.datasets.get(dataset.type)(
        preproc=get_transformer(dataset.transforms.kwargs),
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
    # return DataPrefetcherCls(val_loader), len(val_loader)


if __name__ == "__main__":
    from core.modules.dataloaders.augments import get_transformer
    from dotmap import DotMap
    from core.trainers.utils import denormalization
    from core.modules.dataloaders.datasets import ClsDataset
    from PIL import Image
    import cv2
    from core.modules.register import Registers

    dataloader_c = {
        "type": "ClsDataloaderTrain",
        "dataset": {
            "type": "ClsDataset",
            "kwargs": {
                "data_dir": "/root/data/DAO/screen",
                "image_set": "train.txt",
                "in_channels": 1,
                "input_size": [224, 224],
                "cache": True,
                "images_suffix": [".bmp"]
            },
            "transforms": {
                "kwargs": {
                    # "histogram": {"p": 1},
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
    dataloader_train, length = Registers.dataloaders.get("ClsDataloaderTrain")(
        is_distributed=False, dataset=dataloader_c.dataset, **dataloader_c.kwargs)

    print(dataloader_train, length)