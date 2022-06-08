# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
import numpy as np
import torch
import torch.nn as nn

from dao.register import Registers


@Registers.losses.register
def CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean'):
    if weight is not None:
        weight = torch.from_numpy(np.array(weight))
    return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    N, C, H, W = 2, 3, 4, 4
    loss = CrossEntropyLoss()
    loss_w = CrossEntropyLoss(weight=torch.ones(C))

    # 1. 测试分割(target是整数形式)， 分类同理只是无H,W，特征为一维的
    input = torch.randn((N, C, H, W), requires_grad=True)
    target = torch.empty((N, H, W), dtype=torch.long).random_(C)
    # print("input:{}".format(input))
    # print("target:{}".format(target))
    output = loss(input, target)
    print("output:{}".format(output))
    output_w = loss_w(input, target)
    print("output_w:{}".format(output))
    output.backward()

    # 2. 测试分割（target是概率形式，float32）
    input = torch.randn((N, C, H, W), requires_grad=True)
    target = torch.randn((N, C, H, W)).softmax(dim=1)
    # print("input:{}".format(input))
    # print("target:{}".format(target))
    output = loss(input, target)
    print("output:{}".format(output))
    output_w = loss_w(input, target)
    print("output_w:{}".format(output))
    output.backward()
    
    # output:1.7510757446289062
    # output_w:1.7510757446289062
    # output:1.381339430809021
    # output_w:1.381339430809021
