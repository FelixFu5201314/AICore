# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import torch.nn as nn
from torch.nn import functional as F
import math


class MyAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size=None):
        super().__init__()
        self.sz = output_size

    def forward(self, x):
        inp_size = x.size()
        kernel_width, kernel_height = inp_size[2], inp_size[3]
        if self.sz is not None:
            if isinstance(self.sz, int):
                kernel_width = math.ceil(inp_size[2] / self.sz)
                kernel_height = math.ceil(inp_size[3] / self.sz)
            elif isinstance(self.sz, list) or isinstance(self.sz, tuple):
                assert len(self.sz) == 2
                kernel_width = math.ceil(inp_size[2] / self.sz[0])
                kernel_height = math.ceil(inp_size[3] / self.sz[1])
            return F.avg_pool2d(input=x,
                                ceil_mode=False,
                                kernel_size=(kernel_width, kernel_height))
