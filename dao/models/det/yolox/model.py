# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import torch.nn as nn
from dao.models.det.yolox.yolox import YOLOX_

from dao.register import Registers


@Registers.det_models.register
def YOLOX(backbone=None, head=None):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
    model = YOLOX_(backbone=backbone, head=head)
    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model


if __name__ == '__main__':
    import torch
    from dotmap import DotMap
    model_kwargs = DotMap({
        "type": "YOLOX",
        "summary_size": [3,224,224],
        "kwargs": {
            "backbone": {
                "depth": 1.0,
                "width": 1.0,
                "in_features": ["dark3", "dark4", "dark5"],
                "in_channels": [256, 512, 1024],
                "depthwise": False,
                "act": "silu"
            },
            "head": {
                "num_classes": 80,
                "width": 1.0,
                "strides": [8, 16, 32],
                "in_channels": [256, 512, 1024],
                "act": "silu",
                "depthwise": False
            }

        }
    })
    x = torch.rand(8, 3, 256, 256)
    model = YOLOX(**model_kwargs.kwargs)
    print(model)


    model.eval()
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
                      "/root/export.onnx",
                      opset_version=13,
                      export_params=True,
                      do_constant_folding=True,
                      verbose=True,
                      input_names=["input"],
                      output_names=["output"]
    )
