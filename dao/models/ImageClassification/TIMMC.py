# _*_coding:utf-8_*_
# @auther:FelixFu
# @Data:2021.4.16
# @github:https://github.com/felixfu520

__all__ = ['TIMMC']

from dao.register import Registers


@Registers.cls_models.register
def TIMMC(backbone_kwargs):
    backbone = Registers.backbones.get("TIMM")(backbone_kwargs)
    return backbone


if __name__ == "__main__":
    from dotmap import DotMap
    from dao.register import Registers
    from dao.models.backbone import TIMM
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

    model = Registers.cls_models.get(model_c.type)(model_c.backbone)
    print(model)