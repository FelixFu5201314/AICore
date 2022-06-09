# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity

import torch

from dao.register import Registers


@Registers.losses.register
def CosineSimilarity(dim=1, eps=1e-08):
    return torch.nn.CosineSimilarity(dim=1, eps=1e-08)


if __name__ == "__main__":
    ed = CosineSimilarity()
    input1 = torch.tensor([[1,0],[3,4]], requires_grad=True, dtype=torch.float32)
    input2 = torch.tensor([[1,0],[0,0]], requires_grad=True, dtype=torch.float32)
    output = ed(input1, input2)
    print(output)
    torch.mean(output).backward()


