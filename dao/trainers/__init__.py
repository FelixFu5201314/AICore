# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# 所有model在__init__.py中导入，是为了自动注册到Registers中

# 分类
from .trainerCls import ClsTrainer, ClsEval, ClsDemo, ClsExport
# 异常检测
from .trainerAnomaly import AnomalyTrainer, AnomalyDemo, AnomalyExport
from .trainerAnomaly import AnomalyTrainer2, AnomalyDemo2, AnomalyExport2
# 分割
from .trainerSeg import SegTrainer, SegEval, SegExport, SegDemo
