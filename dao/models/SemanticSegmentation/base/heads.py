import torch.nn as nn
from .modules import Activation


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, num_classes=21, pooling="avg", dropout=0.2, activation=None, is_mask=True,
                 mid_channels=512, stride=32):
        if is_mask:     # 使用mask作为辅助分支
            conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
            bn1 = nn.BatchNorm2d(mid_channels)
            r1 = nn.ReLU(inplace=True)
            d1 = nn.Dropout2d(0.1)
            conv2 = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
            up = nn.UpsamplingBilinear2d(scale_factor=stride)
            super().__init__(conv1, bn1, r1, d1, conv2, up)
        else:   # 使用分类作为辅助分支
            if pooling not in ("max", "avg"):
                raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
            pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
            flatten = nn.Flatten()
            dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
            linear = nn.Linear(in_channels, num_classes, bias=True)
            activation = Activation(activation)
            super().__init__(pool, flatten, dropout, linear, activation)
