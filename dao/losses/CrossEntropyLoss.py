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
    """
    使用注意内容：
        1.target有两种形式，int和float，两种对应的输入shape不一样
    """
    N, C, H, W = 2, 3, 4, 4
    loss = CrossEntropyLoss()
    loss_w = CrossEntropyLoss(weight=torch.ones(C))

    # 0.手动计算， CE = softmax + log + nn.NLLLoss
    # 预测值
    # predict的shape是[2,3],表示两个数据对三类任务的预测值
    predict = torch.Tensor([[0.5796, 0.4403, 0.9087],
                            [-1.5673, -0.3150, 1.6660]])
    # 真实值
    # target的长度对应predict的shape[0],最大值为predict的shape[1] - 1
    # 也就是第0行取index=0，第1行取index=2
    target = torch.tensor([0, 2])

    # 这里输入的是原始预测值
    print(loss(predict, target))  # tensor(0.6725)

    soft_max = torch.nn.Softmax(dim=-1)
    soft_out = soft_max(predict)
    # tensor([[0.3068, 0.2669, 0.4263],
    #        [0.0335, 0.1172, 0.8494]])

    log_soft_out = torch.log(soft_max(predict))
    # tensor([[-1.1816, -1.3209, -0.8525],
    #         [-3.3966, -2.1443, -0.1633]])

    nll_loss = torch.nn.NLLLoss()
    # 这里输入的是经过log_softmax的值
    print(nll_loss(log_soft_out, target))
    # tensor(0.6725) = (-1.1816 + -(-0.1633))/2

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
