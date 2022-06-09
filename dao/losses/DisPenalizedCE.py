# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:https://cloud.tencent.com/developer/article/1657195

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dao.register import Registers


@Registers.losses.register
class DisPenalizedCE(torch.nn.Module):
    """
    Only for binary 3D segmentation
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        # print(inp.shape, target.shape) # (batch, 2, xyz), (batch, 2, xyz)
        # compute distance map of ground truth
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(target.cpu().numpy() > 0.5) + 1.0

        dist = torch.from_numpy(dist)
        if dist.device != inp.device:
            dist = dist.to(inp.device).type(torch.float32)
        dist = dist.view(-1, )

        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        log_sm = torch.nn.LogSoftmax(dim=1)
        inp_logs = log_sm(inp)

        target = target.view(-1, )
        # loss = nll_loss(inp_logs, target)
        loss = -inp_logs[range(target.shape[0]), target]
        # print(loss.type(), dist.type())
        weighted_loss = loss * dist

        return loss.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    N, C, H, W = 2, 3, 4, 4
    loss = BoundaryLoss()

    # 1. 测试分割(target是整数形式)， 分类同理只是无H,W，特征为一维的
    input = torch.randn((N, C, H, W), requires_grad=True)
    target = torch.empty((N, H, W), ).random_(C)
    # print("input:{}".format(input))
    # print("target:{}".format(target))
    output = loss(input, target)
    print("output:{}".format(output))
    output.backward()

