# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:https://github.com/yassouali/pytorch-segmentation
import torch
import torch.nn.functional as F
from torch import nn
from dao.models.SemanticSegmentation.pspnet2 import resnet
from dao.models.SemanticSegmentation.base import MyAdaptiveAvgPool2d
from torchvision import models
from dao.models.SemanticSegmentation.pspnet2.heplers import initialize_weights, set_trainable
from itertools import chain
from dao.register import Registers


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        # prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        prior = MyAdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


@Registers.seg_models.register
class PSPNet2(nn.Module):
    def __init__(self, backbones=None, num_classes=21, in_channels=3, backbone='resnet152', pretrained=True, use_aux=True, freeze_bn=False,
                 freeze_backbone=False, aux_params=None, upsampling=8, isExport=False):
        super(PSPNet2, self).__init__()
        self.isExport = isExport
        norm_layer = nn.BatchNorm2d
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer)
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux

        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz // 2, m_out_sz // 4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        output = self.master_branch(x)
        # output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        # output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux

        if self.isExport and not self.training:  # 是否将(batchsize,numClasses, Height, Width)->(batchsize, Height, Width)
            output = torch.argmax(output, dim=1).float()

        return output

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


# PSP with dense net as the backbone
class PSPDenseNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, backbone='densenet201', pretrained=True, use_aux=True,
                 freeze_bn=False, **_):
        super(PSPDenseNet, self).__init__()
        self.use_aux = use_aux
        model = getattr(models, backbone)(pretrained)
        m_out_sz = model.classifier.in_features
        aux_out_sz = model.features.transition3.conv.out_channels

        if not pretrained or in_channels != 3:
            # If we're training from scratch, better to use 3x3 convs
            block0 = [nn.Conv2d(in_channels, 64, 3, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
            block0.extend(
                [nn.Conv2d(64, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)] * 2
            )
            self.block0 = nn.Sequential(
                *block0,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.block0)
        else:
            self.block0 = nn.Sequential(*list(model.features.children())[:4])

        self.block1 = model.features.denseblock1
        self.block2 = model.features.denseblock2
        self.block3 = model.features.denseblock3
        self.block4 = model.features.denseblock4

        self.transition1 = model.features.transition1
        # No pooling
        self.transition2 = nn.Sequential(
            *list(model.features.transition2.children())[:-1])
        self.transition3 = nn.Sequential(
            *list(model.features.transition3.children())[:-1])

        for n, m in self.block3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (2, 2), (2, 2)
        for n, m in self.block4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (4, 4), (4, 4)

        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(aux_out_sz, m_out_sz // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(m_out_sz // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        x = self.block0(x)
        x = self.block1(x)
        x = self.transition1(x)
        x = self.block2(x)
        x = self.transition2(x)
        x = self.block3(x)
        x_aux = self.transition3(x)
        x = self.block4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.block0.parameters(), self.block1.parameters(), self.block2.parameters(),
                     self.block3.parameters(), self.transition1.parameters(), self.transition2.parameters(),
                     self.transition3.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    import torch
    from dotmap import DotMap

    model_kwargs = DotMap({
        "type": "PSPNet2",
        "kwargs": {
            "num_classes": 21,
            "in_channels": 3,
            "backbone": "resnet50",
            "pretrained": True,
            "use_aux": False,
            "freeze_bn": False,
            "freeze_backbone": False
        }
    })
    x = torch.rand(8, 3, 256, 256)
    model = PSPNet2(**model_kwargs.kwargs)

    output = model(x)
    # print(output.shape)
    model.eval()
    # TODO
    # RuntimeError: Unsupported: ONNX export of operator adaptive_avg_pool2d,
    # since output size is not factor of input size.
    # Please feel free to request support or submit a pull request on PyTorch GitHub.
    # 目前torch不支持adaptive_avg_pool2d操作，而PSPNet中用了这种操作，现在新的源码已经支持，但需要重新编译torch，所以等待wheel版本出来
    # 后再说， 2022.1.13
    torch.onnx.export(model,
                      x,
                      "/ai/data/pspnet2.onnx",
                      opset_version=11,
                      input_names=["input"],
                      output_names=["output"]
    )

"""
import onnx 
import torch
import torchvision 
import netron

net = torchvision.models.resnet18(pretrained=True).cuda()
net.eval()
静态：
export_onnx_file = "./resnet18.onnx"
torch.onnx.export(net,  # 待转换的网络模型和参数
                torch.randn(1, 3, 224, 224, device='cuda'), # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                export_onnx_file,  # 输出文件的名称
                verbose=False,      # 是否以字符串的形式显示计算图
                input_names=["input"],# + ["params_%d"%i for i in range(120)],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                output_names=["output"], # 输出节点的名称
                opset_version=10,   # onnx 支持采用的operator set, 应该和pytorch版本相关，目前我这里最高支持10
                do_constant_folding=True, # 是否压缩常量
                )
动态：
export_onnx_file = "./resnet18_dynamic.onnx"
torch.onnx.export(net,  # 待转换的网络模型和参数
                torch.randn(1, 3, 224, 224, device='cuda'), # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                export_onnx_file,  # 输出文件的名称
                verbose=False,      # 是否以字符串的形式显示计算图
                input_names=["input"],# + ["params_%d"%i for i in range(120)],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                output_names=["output"], # 输出节点的名称
                opset_version=10,   # onnx 支持采用的operator set, 应该和pytorch版本相关，目前我这里最高支持10
                do_constant_folding=True, # 是否压缩常量
                dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},} #设置动态维度，此处指明input节点的第0维度可变，命名为batch_size
                )
"""