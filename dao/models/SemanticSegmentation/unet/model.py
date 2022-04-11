# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..base import SegmentationHead, ClassificationHead, SegmentationModel

from dao.register import Registers


@Registers.seg_models.register
class Unet(SegmentationModel):
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
        """
        Unet 继承自SegmentationModel，由4个组件组成，即encoder（backbone）不在此处定义、decoder、segmentation_Head、classificationHead；
            encoder: dict encoder的配置字典
            encoder_depth: encoder深度
            encoder_channels: encoder 的每一层channel数
            decoder_use_batchnorm: bool 是否使用batchnorm
            decoder_channels: List[int] = (256, 128, 64, 32, 16),   # decoder的通道数
            decoder_attention_type: Optional[str] = None,   decoder的attention类型
            num_classes=2,  类别数，包括背景，假如例如VOC是21，其中一类是背景
            activation: Optional[Union[str, callable]] = None,  激活函数
            aux_params: Optional[dict] = None,  分类辅助分支
        """
        super().__init__()

        self.encoder = Registers.backbones.get("TIMM")(encoder)     # encoder

        self.decoder = UnetDecoder(     # decoder
            encoder_channels=encoder_channels,  # encoder_depth个，每层的encoder通道数
            decoder_channels=decoder_channels,  # encoder_depth个， (256, 128, 64, 32, 16)
            n_blocks=encoder_depth,  # encoder 下采样次数
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
        "type": "Unet",
        "backbone": {
            "kwargs": {
                "model_name": "resnet18",
                "pretrained": True,
                "checkpoint_path": "",
                "exportable": True,
                "in_chans": 3,
                "features_only": True,
                # "out_indices": [1, 3, 4]
            }
        },
        "kwargs": {
            "encoder_depth": 5,
            "encoder_channels": [3, 64, 64, 128, 256, 512],
            "decoder_use_batchnorm": True,
            "decoder_channels": [256, 128, 64, 32, 16],
            "num_classes": 21,
            "aux_params": None,
        }
    })
    x = torch.rand(8, 3, 256, 256)
    model = Unet(model_kwargs.backbone, **model_kwargs.kwargs)
    print(model)
    output = model(x)

    print(output.shape)
    model.eval()
    torch.onnx.export(model,
                      x,
                      "/root/code/onnx_name.onnx",
                      opset_version=9,
                      input_names=["input"],
                      output_names=["output"])