#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.


from tqdm import tqdm

import torch

from dao.utils import is_main_process, synchronize, time_synchronized, gather, get_world_size, MeterSegEval
from dao.register import Registers


@Registers.evaluators.register
class SegEvaluator:
    def __init__(self, is_distributed=False, dataloader=None, num_classes=None):
        """
        验证器
        is_distributed:bool 是否是分布式
        dataloader:dict dataloader的配置字典
        num_classes:int 类别数
        """
        self.dataloader = Registers.dataloaders.get(dataloader.type)(
            is_distributed=is_distributed,
            dataset=dataloader.dataset,
            **dataloader.kwargs
        )
        self.iters_per_epoch = len(self.dataloader)
        self.meter = MeterSegEval(num_classes)
        self.num_classes = num_classes

    def evaluate(self, model, distributed=False, half=False, device=None):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        pixAccs = []  # 用于存放all world size汇集的数据
        mIoUs = []  # 用于存放all world size汇集的数据
        Class_IoUs = []  # 用于存放all world size汇集的数据

        # progress_bar = tqdm if is_main_process() else iter
        progress_bar = iter  # 使用tqdm在多GPU时，可能会卡死

        self.meter.reset_metrics()
        for imgs, targets, paths in progress_bar(self.dataloader):
            with torch.no_grad():
                imgs = imgs.to(device=device)
                targets = targets.to(device=device)
                imgs = imgs.type(tensor_type)
                outputs = model(imgs)
                seg_metrics = self.meter.eval_metrics(outputs, targets, self.num_classes)
                self.meter.update_seg_metrics(*seg_metrics)

        pixAcc, mIoU, Class_IoU = self.meter.get_seg_metrics().values()

        if distributed:  # 如果是分布式，将结果gather到0设备上
            pixAccs = gather(pixAcc, dst=0)
            mIoUs = gather(mIoU, dst=0)
            Class_IoUs = gather(Class_IoU, dst=0)
            if is_main_process():
                pixAcc = sum(pixAccs) / get_world_size()
                mIoU = sum(mIoUs) / get_world_size()
                Class_IoU = Class_IoUs[0]
                for classiou in Class_IoUs[1:]:
                    for k, v in Class_IoU.items():
                        Class_IoU[k] += classiou[k]

                for k, v in Class_IoU.items():
                    Class_IoU[k] /= get_world_size()

        if not is_main_process():
            return 0, 0, None

        Class_IoU_dict = {}
        for k, v in Class_IoU.items():
            Class_IoU_dict[self.dataloader.dataset.labels_dict[str(k)]] = v
        return pixAcc, mIoU, Class_IoU_dict
