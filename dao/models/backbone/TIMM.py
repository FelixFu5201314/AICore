# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import timm
from loguru import logger
from dao.register import Registers

__all__ = ['TIMM']


@Registers.backbones.register
def TIMM(backbone):
    """
    获取TIMM主干网络

    backbone:dict backbone:{kwargs:{这里面是timm库创建model的参数}}
    """
    # 判断model是否在timm支持列表中
    if backbone.kwargs.model_name not in timm.list_models():
        logger.error("timm {} not supported {}".format(
            timm.__version__,
            backbone.kwargs.model_name))
        raise

    # 判断model是否有pretrained
    if backbone.kwargs.pretrained and backbone.kwargs.model_name not in timm.list_models(pretrained=True):
        logger.error("{} hasn't pretrained weight, please set pretrained False".format(
            backbone.kwargs.model_name
        ))
        raise

    model = timm.create_model(**backbone.kwargs)
    return model


if __name__ == "__main__":
    from dotmap import DotMap
    from dao.register import Registers

    model_c = {
        "type": "TIMMC",
        "summary_size": [1, 224, 224],
        "backbone": {
            "kwargs": {
                "model_name": "efficientnet_b0",
                "pretrained": True,
                "checkpoint_path": "",
                "exportable": True,
                "in_chans": 1,
                "num_classes": 38
            }
        },
        "kwargs": {
        }
    }
    model_c = DotMap(model_c)

    model = Registers.backbones.get("TIMM")(model_c.backbone)
    print(model)
