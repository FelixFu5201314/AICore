#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
# @Ref: https://github.com/qubvel/segmentation_models.pytorch
# @Ref: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/unetplusplus/model.py

from typing import Optional, Union, List
from dao.models.SemanticSegmentation.base import SegmentationModel, SegmentationHead, ClassificationHead
from dao.models.SemanticSegmentation.unetplusplus.decoder import UnetPlusPlusDecoder
from dao.register import Registers


@Registers.seg_models.register
class UnetPlusPlus(SegmentationModel):
    def __init__(
        self,
        encoder,
        encoder_depth=5,
        encoder_channels=None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        num_classes=2,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = Registers.backbones.get("TIMM")(encoder)
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder.kwargs.model_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
        )
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unet-{}".format(encoder.kwargs.model_name)
        self.initialize()


if __name__ == '__main__':
    import torch
    from dotmap import DotMap
    model_kwargs = DotMap({
        "type": "UnetPlusPlus",
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
            "encoder_depth": 5,
            "encoder_channels": [3, 64, 256, 512, 1024, 2048],
            "decoder_channels": [256, 128, 64, 32, 16],
            "num_classes": 21
        }
    })
    x = torch.rand(8, 3, 256, 256)
    model = UnetPlusPlus(model_kwargs.backbone, **model_kwargs.kwargs)

    output = model(x)
    print(output.shape)
    model.eval()
    torch.onnx.export(model,
                      x,
                      "/ai/data/onnx_name.onnx",
                      opset_version=11,
                      input_names=["input"],
                      output_names=["output"])
