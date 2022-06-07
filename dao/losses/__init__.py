# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# 所有model在__init__.py中导入，是为了自动注册到Registers中

# 1, L1 Loss, https://www.cvmart.net/community/detail/4879
# 也称为Mean Absolute Error，即平均绝对误差（MAE），它衡量的是预测值与真实值之间距离的平均误差幅度，作用范围为0到正无穷。
# 优点： 收敛速度快，能够对梯度给予合适的惩罚权重，而不是“一视同仁”，使梯度更新的方向可以更加精确。
# 缺点： 对异常值十分敏感，梯度更新的方向很容易受离群点所主导，不具备鲁棒性。
from .L1Loss import L1Loss_DAO

# 2, L2 Loss, https://www.cvmart.net/community/detail/4879
# 也称为Mean Squred Error，即均方差（MSE），它衡量的是预测值与真实1值之间距离的平方和，作用范围同为0到正无穷。
# 优点： 对离群点（Outliers）或者异常值更具有鲁棒性。
# 缺点： 在0点处的导数不连续，使得求解效率低下，导致收敛速度慢；而对于较小的损失值，其梯度也同其他区间损失值的梯度一样大，所以不利于网络的学习。
from .MSELoss import MSELoss_DAO

# 3, 负对数似然损失 NLLLoss
from .NLLLoss import NLLLoss_DAO

# 4, 交叉熵损失 CrossEntropyLoss
from .CrossEntropyLoss import CrossEntropyLoss
from .CrossEntropyLoss import CrossEntropyLoss as CrossEntropyLoss_DAO

# 5,
from .PoissonNLLLoss import PoissonNLLLoss_DAO

# 6,
from .BCELoss import BCELoss_DAO

# 7,
from .BCEWithLogitsLoss import BCEWithLogitsLoss_DAO

# 8,
from .MarginRankingLoss import MarginRankingLoss_DAO

# 9,
from .Dice import DiceLoss_DAO

# 10,
from .EuclideanDistance import EuclideanDistance




