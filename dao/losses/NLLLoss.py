# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
#       https://blog.csdn.net/zhuangyuan7838/article/details/121267301

import numpy as np

import torch

from dao.register import Registers


@Registers.losses.register
def NLLLoss(weight=None, reduction="mean"):
    return torch.nn.NLLLoss(weight=weight, reduction=reduction)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    N, C, H, W = 2, 3, 4, 4
    m = torch.nn.Sigmoid()

    loss = NLLLoss()
    loss_w = NLLLoss(weight=torch.ones(C))

    # 1. 简单的自定义数据
    # 预测值
    predict = torch.Tensor([[0.5796, 0.4403, 0.9087],
                            [-1.5673, -0.3150, 1.6660]])
    # 真实值
    target = torch.tensor([0, 2])

    # torch计算
    print(loss(predict, target))  # tensor(-1.1228)

    # 手动计算
    result = 0
    for i, j in enumerate(range(target.shape[0])):
        # 分别取出0.5796和1.6660
        # 也就是log_soft_out[0][0]和log_soft_out[1][2]
        result -= predict[i][target[j]]
    print(result / target.shape[0])
    # tensor(-1.1228)

    # 2. 测试分类
    # input is of size N x C = 3 x 5
    input = torch.randn(N, C, requires_grad=True)   # 概率值
    # each element in target has to have 0 <= value < C
    target = torch.empty(N, dtype=torch.long).random_(C)  # [0, C]
    output = loss(m(input), target)
    print(output)
    output.backward()

    # 3. 测试分割
    # 2D loss example (used, for example, with image inputs)
    # input is of size N x C x height x width
    data = torch.randn(N, 16, 10, 10)
    conv = torch.nn.Conv2d(16, C, (3, 3))
    m = torch.nn.LogSoftmax(dim=1)
    # each element in target has to have 0 <= value < C
    target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    output = loss(m(conv(data)), target)
    print(output)
    output.backward()