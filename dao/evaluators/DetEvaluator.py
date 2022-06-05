#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import pickle
import os
from tqdm import tqdm
import numpy as np
import torch
import shutil
from loguru import logger
from PIL import Image

from dao.utils import colorize_mask, get_palette
from dao.utils import is_main_process, synchronize, time_synchronized, gather, get_world_size, MeterDetEval
from dao.register import Registers


@Registers.evaluators.register
class DetEvaluator:
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
        self.meter = MeterDetEval(num_classes)
        self.num_classes = num_classes

    def evaluate(self, model, distributed=False, half=False, device=None, output_dir=None, save_pic=False):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        model.trainable = 1

        pixAccs = []  # 用于存放all world size汇集的数据
        mIoUs = []  # 用于存放all world size汇集的数据
        Class_IoUs = []  # 用于存放all world size汇集的数据

        # progress_bar = tqdm if is_main_process() else iter
        # progress_bar = iter  # 使用tqdm在多GPU时，可能会卡死

        self.meter.reset_metrics()

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        num_images = len(self.dataloader.dataset)
        self.all_boxes = [[[] for _ in range(num_images)]
                        for _ in range(self.num_classes)]

        # timers
        det_file = os.path.join(output_dir, 'detections.pkl')
        for i, (imgs, targets, paths) in enumerate(self.dataloader):
            logger.info(f"evaluator iter:{i}/{self.iters_per_epoch}")
            with torch.no_grad():
                imgs = imgs.to(device=device)
                imgs = imgs.type(tensor_type)
                bboxes, scores, cls_inds  = model(imgs)
                w, h = imgs[0].shape[1], imgs[0].shape[2]
                scale = np.array([[w, h, w, h]])
                bboxes *= scale

                # targets = targets.to(device=device)

                for j in range(self.num_classes):
                    inds = np.where(cls_inds == j)[0]
                    if len(inds) == 0:
                        self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = bboxes[inds]
                    c_scores = scores[inds]
                    c_dets = np.hstack((c_bboxes,
                                        c_scores[:, np.newaxis])).astype(np.float32,
                                                                         copy=False)
                    self.all_boxes[j][i] = c_dets

        with open(det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        self.evaluate_detections(self.all_boxes)

        print('Mean AP: ', self.map)

        #         seg_metrics = self.meter.eval_metrics(outputs, targets, self.num_classes)
        #         self.meter.update_seg_metrics(*seg_metrics)
        #
        # pixAcc, mIoU, Class_IoU = self.meter.get_seg_metrics().values()

        # if distributed:  # 如果是分布式，将结果gather到0设备上
        #     pixAccs = gather(pixAcc, dst=0)
        #     mIoUs = gather(mIoU, dst=0)
        #     Class_IoUs = gather(Class_IoU, dst=0)
        #     if is_main_process():
        #         pixAcc = sum(pixAccs) / get_world_size()
        #         mIoU = sum(mIoUs) / get_world_size()
        #         Class_IoU = Class_IoUs[0]
        #         for classiou in Class_IoUs[1:]:
        #             for k, v in Class_IoU.items():
        #                 Class_IoU[k] += classiou[k]
        #
        #         for k, v in Class_IoU.items():
        #             Class_IoU[k] /= get_world_size()

        # if not is_main_process():
        #     return 0, 0, None
        #
        # Class_IoU_dict = {}
        # for k, v in Class_IoU.items():
        #     Class_IoU_dict[self.dataloader.dataset.labels_dict[str(k)]] = v
        #
        # # synchronize()
        # return pixAcc, mIoU, Class_IoU_dict

