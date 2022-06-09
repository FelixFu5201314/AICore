# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# 所有model在__init__.py中导入，是为了自动注册到Registers中

"""
Pytorch 官方的一些距离,损失
"""
# 1. negative log likelihood loss
# 概念：极大似然估计：https://zhuanlan.zhihu.com/p/26614750；https://blog.csdn.net/zengxiantao1994/article/details/72787849
# torch API: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
from .NLLLoss import NLLLoss

# 2. L1损失
from .L1Loss import L1Loss

# 3. L2损失 均方误差
from .MSELoss import MSELoss


"""
[A survey of loss functions for semantic segmentation](https://paperswithcode.com/paper/a-survey-of-loss-functions-for-semantic)
"""
# ------------------- Distribution-based loss for cls,seg-----------------------
# 1.Cross-Entropy 交叉熵损失损失, 用于多分类. Cross-Entropy = Softmax + log + NLLLoss
# 概念: https://blog.csdn.net/tsyccnh/article/details/79163834
# torch API(如何使用?): https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
#       在分割中，pred需要输出n_class个通道
from .CrossEntropyLoss import CrossEntropyLoss

# 2.Binary Cross-Entropy 二进制交叉熵损失函数，适用于二分类和多分类
# 概念：https://www.jianshu.com/p/5b01705368bb ； https://blog.csdn.net/zhuangyuan7838/article/details/121267301
# 如何使用：https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
#       在分割中，pred需要输出一个通道；
#       面对二分类问题时，CELoss是Softmax + BCELoss
#       如此BCELoss相比CELoss在解决多分类问题的优势就表现了出来。CELoss只是根据每行的分类结果去取值，而BCELoss考虑了每行全部结果
from .BCELoss import BCELoss
from .BCEWithLogitsLoss import BCEWithLogitsLoss

# 3. Focal Loss
# 概念：https://zhuanlan.zhihu.com/p/28527749;https://blog.csdn.net/cp1314971/article/details/105559545
#       https://cloud.tencent.com/developer/article/1669261
# 如何使用：
# focal loss解决了什么问题？
# （1）不同类别不均衡
# （2）难易样本不均衡
from .FocalLoss import FocalLoss, FocalLoss_det_cls

# 4.DisPenalizedCE
# 概念：
# 如何使用：
# TODO
from .DisPenalizedCE import DisPenalizedCE

# ------------------- Region-based loss -----------------------------
# 1.Dice Loss
# 概念：https://blog.csdn.net/baidu_36511315/article/details/105217674
# 如何使用：
from .Dice import DiceLoss

# 2. IOU loss
# 概念：https://blog.csdn.net/weixin_42990464/article/details/104260043
# 如何使用：
from .IOULoss import IoULoss

# 3.Tversky Loss
# 概念：https://freewechat.com/a/MzI5MDUyMDIxNA==/2247500026/3
# 如何使用：
from .TverskyLoss import TverskyLoss

# 4.FocalTverskyLoss
# TODO

# 5.SensitivitySpecificityLoss
# TODO

# 6. LogCoshDiceLoss
# TODO

# ------------------- Boundary-based loss ---------------------------
# 1.BoundaryLoss
# 概念：TODO 未找到好的解释博客
# 如何使用？：参考代码
from .BoundaryLoss import BoundaryLoss

# 2. SahpeAwareLoss
# TODO

# ------------------- Compounded loss -------------------------------
from .DiceBCELoss import DiceBCELoss


"""
列出所有距离计算
https://www.cvmart.net/community/detail/2982
"""
# ------------------------- 常见的距离算法 --------------------------
# 1. 欧式距离等P-norm距离
from .EuclideanDistance import EuclideanDistance

# 2. EMA Earth Mover's Distance
# TODO

# 3. Manhattan Distance
# TODO

# 4. Jaccard Distance
# 参见IOULoss

# 5. Mahalanobis Distance
# TODO

# 6. Chebyshev Distance 切比雪夫距离
# TODO

# 7. Minkowski Distance 明可夫斯基距离
# TODO

# 8. Mahalanobis Distance 马哈拉诺比斯距离
# TODO

# ---------------------- 常见的相似度（系数）算法 ---------------------
# 1. Cosine Similarity 余弦相似度
from .CosineSimilarity import CosineSimilarity


# 2. Pearson Correlation Coefficient 皮尔森相关系数
# TODO

# 3. Kullback-Leibler Divergence KL散度
from .KLDivLoss import KLDivLoss

# 4. Jaccard Coefficient Jaccard相似系数
# 参考IOULoss

# 5. Tanimoto系数（广义Jaccard相似系数）
# TODO

# 6. Mutual Information 互信息
# TODO








