# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# 所有model在__init__.py中导入，是为了自动注册到Registers中

# 1, L1范数误差损失、mean absolute error (MAE)
from .L1Loss import L1Loss_DAO

# 2, 均方误差损失 MSELoss
from .MSELoss import MSELoss_DAO

# 3, 负对数似然损失 NLLLoss
from .NLLLoss import NLLLoss_DAO

# 4, 交叉熵损失 CrossEntropyLoss
from .CrossEntropyLoss import CrossEntropyLoss
from .CrossEntropyLoss import CrossEntropyLoss as CrossEntropyLoss_DAO



from .CrossEntropyLoss import CrossEntropyLoss
