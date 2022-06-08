# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
import numpy as np
import torch
import torch.nn as nn

from dao.register import Registers


@Registers.losses.register
def BCELoss(weight=None, reduction='mean'):
    if weight is not None:
        weight = torch.from_numpy(np.array(weight))
    return nn.BCELoss(weight=weight,  reduction=reduction)


if __name__ == "__main__":
    N, C, H, W = 2, 3, 4, 4
    m = nn.Sigmoid()
    loss = BCELoss(reduction="none")
    # 1. 测试分类
    input = torch.randn((N, C), requires_grad=True)
    target = torch.empty((N, C)).random_(C)
    output = loss(m(input), target)
    print("input:{}".format(m(input)))
    print("target:{}".format(target))
    print("output:{}".format(output))
    torch.mean(output).backward()

    # 2. 测试分割
    input = torch.randn((N, C, H, W), requires_grad=True)
    target = torch.empty((N, C, H, W)).random_(C)
    output = loss(m(input), target)
    print("input:{}".format(m(input)))
    print("target:{}".format(target))
    print("output:{}".format(output))
    torch.mean(output).backward()

    # 3. 测试分割(自定义数据)
    input = [[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [1, 0, 0, 1]]]]
    target = [[
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [1, 0, 0, 1]]]]
    input = torch.tensor(input, dtype=torch.float32, requires_grad=True)
    target = torch.tensor(target, dtype=torch.float32)
    output = loss(input, target)
    print("input:{}".format(m(input)))
    print("target:{}".format(target))
    print("output:{}".format(output))
    torch.mean(output).backward()

    # 4. 测试分类(Weighted Binary Cross-Entropy加权交叉熵损失函数)
    loss = BCELoss(reduction="none", weight=torch.empty(C).random_(C))
    input = torch.randn((N, C), requires_grad=True)
    target = torch.empty((N, C)).random_(C)
    output = loss(m(input), target)
    print("input:{}".format(m(input)))
    print("target:{}".format(target))
    print("output:{}".format(output))
    torch.mean(output).backward()

    # 4. 测试分割(Weighted Binary Cross-Entropy加权交叉熵损失函数)
    loss = BCELoss(reduction="none", weight=torch.empty(C).random_(C))
    input = torch.randn((N, C, H, W), requires_grad=True)
    target = torch.empty((N, C, H, W)).random_(C)
    output = loss(m(input), target)
    print("input:{}".format(m(input)))
    print("target:{}".format(target))
    print("output:{}".format(output))
    torch.mean(output).backward()