# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
from typing import Optional, Union
from dao.models.SemanticSegmentation.base import SegmentationModel, SegmentationHead, ClassificationHead
from dao.models.SemanticSegmentation.pspnet.decoder import PSPDecoder
from dao.register import Registers


@Registers.seg_models.register
class PSPNet(SegmentationModel):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation. Consist of
    *encoder* and *Spatial Pyramid* (decoder). Spatial Pyramid build on top of encoder and does not
    use "fine-features" (features of high spatial resolution). PSPNet can be used for multiclass segmentation
    of high resolution images, however it is not good for detecting small objects and producing accurate, pixel-level mask.
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        psp_out_channels: A number of filters in Spatial Pyramid
        psp_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        psp_dropout: Spatial dropout rate in [0, 1) used in Spatial Pyramid
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **PSPNet**
    .. _PSPNet:
        https://arxiv.org/abs/1612.01105
    """

    def __init__(
        self,
        encoder,
        encoder_channels=None,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
        num_classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 8,
        aux_params: Optional[dict] = None
    ):
        super().__init__()

        self.encoder = Registers.backbones.get("TIMM")(encoder)

        self.decoder = PSPDecoder(
            encoder_channels=encoder_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=num_classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        if aux_params:
            self.classification_head = ClassificationHead(in_channels=encoder_channels[-1], stride=32, **aux_params)
        else:
            self.classification_head = None

        self.name = "psp-{}".format(encoder.kwargs.model_name)
        self.initialize()


if __name__ == '__main__':
    import torch
    from dotmap import DotMap
    model_kwargs = DotMap({
        "type": "PSPNet",
        "backbone": {
            "kwargs": {
                "model_name": "resnet50",
                "pretrained": True,
                "checkpoint_path": "",
                "exportable": True,
                "in_chans": 3,
                "features_only": True
            }
        },
        "kwargs": {
            "encoder_channels": [3, 64, 256, 512, 1024, 2048],
            "psp_out_channels": 512,
            "num_classes": 21,
            "upsampling": 32,
            "aux_params": {
                "num_classes": 21,
                "is_mask": True,
                "mid_channels": 512
            }
        }
    })
    x = torch.rand(8, 3, 256, 256)
    model = PSPNet(model_kwargs.backbone, **model_kwargs.kwargs)

    output = model(x)
    # print(output.shape)
    model.eval()
    output = model(x)
    # TODO
    # RuntimeError: Unsupported: ONNX export of operator adaptive_avg_pool2d,
    # since output size is not factor of input size.
    # Please feel free to request support or submit a pull request on PyTorch GitHub.
    # 目前torch不支持adaptive_avg_pool2d操作，而PSPNet中用了这种操作，现在新的源码已经支持，但需要重新编译torch，所以等待wheel版本出来
    # 后再说， 2022.1.13
    torch.onnx.export(model,
                      x,
                      "/ai/data/onnx_name.onnx",
                      opset_version=11,
                      export_params=True,
                      do_constant_folding=True,
                      verbose=True,
                      input_names=["input"],
                      output_names=["output"]
    )
