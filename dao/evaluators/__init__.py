# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# 所有model在__init__.py中导入，是为了自动注册到Registers中

# 分类
from .ClsEvaluator import ClsEvaluator

# 分割
from .SegEvaluator import SegEvaluator

# 目标检测
from .DetEvaluator import DetEvaluator
