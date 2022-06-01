# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# 所有model在__init__.py中导入，是为了自动注册到Registers中

# 分类: Classification
from .datasets import ClsDataset
from .ClsDataloader import ClsDataloaderTrain, ClsDataloaderEval

# 异常检测: AnomalyDetection
from .datasets import MVTecDataset
from .AnomalyDataloader import MVTecDataloader

# 分割: Segmentation
from .datasets import SegDataset
from .SegDataloader import SegDataloaderTrain, SegDataloaderEval

# 目标检测: ObjectDetection
from .datasets import DetDataset
from .DetDataloader import DetDataloaderTrain, DetDataloaderEval
