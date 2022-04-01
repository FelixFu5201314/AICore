# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520
import os
import time
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
import functools
from collections import defaultdict, deque


import torch
__all__ = ['MeterDetTrain', 'MeterDetEval', 'MeterBuffer']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


class MeterDetTrain(object):
    """
    监控data_time, batch_time, total_loss, lr
    """
    def __init__(self):
        self.reset_metrics()
        self.lr = 0

    def update_metrics(self, data_time=0, batch_time=0, total_loss=0, lr=0):
        self.batch_time.update(batch_time)
        self.data_time.update(data_time)
        self.total_loss.update(total_loss)
        self.lr = lr

    def reset_metrics(self):
        """重置metrics
            1、训练时间：batch_time
            2、读取数据时间：data_time
            3、损失值：total_loss
        """
        self.batch_time = AverageMeter()    # 训练时间
        self.data_time = AverageMeter()  # 读取数据时间
        self.total_loss = AverageMeter()    # 损失值


class MeterDetEval(object):
    def __init__(self, num_class=38):
        self.reset_metrics()
        self.lr = 0
        self.num_classes = num_class

    def update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct  # 预测正确的像素点
        self.total_label += labeled  # 所有像素点
        self.total_inter += inter  # 21类中的交，即预测正确的像素点（每一个类）, list shape 为21
        self.total_union += union  # 21类中的并，即预测正确与错误的像素点（每一个类）

    def get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)  # 预测正确的像素准确率
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

    def reset_metrics(self):
        """重置metrics
            1、交：total_inter, 并：total_union
            2、准确率：total_correct, total_label
        """
        self.total_inter, self.total_union = 0, 0  # 交、并
        self.total_correct, self.total_label = 0, 0  # 准确率

    def eval_metrics(self, output, target, num_class):
        """
        功能：计算度量
        output(4,21,380,380) target(4,380,380) num_class=21
        """
        # 分割是预测每个像素，所以每个像素都是一个类别，共380*380个像素。所以predict为（4，380，380），eg. predict[0][1][1]=20 代表第0张图片第（1，1）个像素的类别是20
        _, predict = torch.max(output.data, 1)
        predict = predict + 1   # 原来是[0, 20]类，先将背景作为1类，[1, 21] & [256]
        target = target + 1

        # （4,380,380) 属于正常范围（分类值在【1，21】）内的每个像素. 255(+1后为256）是标注的轮廓，不是object， 所以不属于labeled中
        labeled = (target > 0) * (target <= num_class)  # torch.Size([4, 380, 380]) True or False(原value是255的）; 即排除原255像素值的mask
        # 获得标注正确的像素个数，和所有标注的像素个数(除去255）
        correct, num_labeled = self._batch_pix_accuracy(predict, target, labeled)   # correct为预测正确个数， num_labeled为标签标注像素个数
        # 获得predict与target的inter(21 length),union(21)
        inter, union = self._batch_intersection_union(predict, target, num_class,  labeled)
        return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5)]

    def _batch_pix_accuracy(self, predict, target, labeled):
        pixel_labeled = labeled.sum()  # 380*380*batchsize - 像素值为255的个数3228 = 574372
        pixel_correct = ((predict == target) * labeled).sum()  # 类别标注正确的像素个数
        assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
        return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

    def _batch_intersection_union(self, predict, target, num_class, labeled):
        predict = predict * labeled.long()  # labeled的predict， torch.Size([16, 224, 224])
        intersection = predict * (predict == target).long()  # 与target一致的predict, 交, 筛选, torch.Size([16, 224, 224])

        area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)  # 预测正确的直方图（类别），交
        area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)  # 预测的直方图（类别），predict 面积
        area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)  # 标签的直方图（类别），target面积
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
        return area_inter.cpu().numpy(), area_union.cpu().numpy()

class AverageMeter2:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

    @property
    def total(self):
        return self._total

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()

class MeterBuffer(defaultdict):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter2, window_size=window_size)
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key="time"):
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()