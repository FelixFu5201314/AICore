# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# 所有model在__init__.py中导入，是为了自动注册到Registers中

"""
[A survey of loss functions for semantic segmentation](https://paperswithcode.com/paper/a-survey-of-loss-functions-for-semantic)
"""
"""
损失函数                         概念                              优点                                 缺点
1,NLLLoss
2,CE
3,BCE
4,Focal
5,
"""
# * negative log likelihood loss
# 概念：极大似然估计：https://zhuanlan.zhihu.com/p/26614750；https://blog.csdn.net/zengxiantao1994/article/details/72787849
# torch API: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
from .NLLLoss import NLLLoss

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


# ------------------- Boundary-based loss ---------------------------
# 1.BoundaryLoss
# 概念：TODO 未找到好的解释博客
# 如何使用？：参考代码
from .BoundaryLoss import BoundaryLoss

# ------------------- Region-based loss -----------------------------
# 1.Dice损失
# 概念：https://blog.csdn.net/baidu_36511315/article/details/105217674
# 如何使用：
from .Dice import DiceLoss

# 2. IOU loss
# 概念：https://blog.csdn.net/weixin_42990464/article/details/104260043
# 如何使用：
from .IOULoss import IoULoss

# 3.


# ------------------- Compounded loss -------------------------------
from .DiceBCELoss import DiceBCELoss









