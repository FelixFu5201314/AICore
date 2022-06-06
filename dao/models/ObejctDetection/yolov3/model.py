# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import (
    darknet19, darknet53, darknet_tiny, darknet_light,
    cspdarknet53, cspdarknet_slim, cspdarknet_tiny,
    resnet18, resnet34, resnet50, resnet101, resnet152,
    resnext50_32x4d, resnext101_32x8d
)
from .utils import build_targets, to_cpu, Conv

from dao.register import Registers


@Registers.det_models.register
class YOLOv3(nn.Module):
    def __init__(self, backbone="darknet53", input_size=None, num_classes=20, anchor_size=None, ignore_thres=0.5):
        """
        Function: 定义YoloV3网络

        :param input_size:int 输入图片大小
        :param num_classes:int 类别数量
        :param anchor_size: list anchor的wh
        """
        super(YOLOv3, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchor_size = anchor_size
        self.num_anchors = len(anchor_size)
        self.ignore_thres = ignore_thres

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100

        self.grid_size = [0, 0, 0]
        self.stride = [32, 16, 8]
        self.grid_x = [0, 0, 0]
        self.grid_y = [0, 0, 0]
        self.scaled_anchors = [0, 0, 0]
        self.anchor_w = [0, 0, 0]
        self.anchor_h = [0, 0, 0]
        self.metrics = [0, 0, 0]

        # backbone
        if backbone == "darknet53":
            self.backbone = darknet53(pretrained=False, hr=False)
        elif backbone == "darknet19":
            self.backbone = darknet19(pretrained=False, hr=False)
        elif backbone == "darknet_tiny":
            self.backbone = darknet_tiny(pretrained=False, hr=False)
        elif backbone == "darknet_light":
            self.backbone = darknet_light(pretrained=False, hr=False)
        elif backbone == "cspdarknet53":
            self.backbone = cspdarknet53(pretrained=False, hr=False)
        elif backbone == "cspdarknet_slim":
            self.backbone = cspdarknet_slim(pretrained=False, hr=False)
        elif backbone == "cspdarknet_tiny":
            self.backbone = cspdarknet_tiny(pretrained=False, hr=False)
        elif backbone == "resnet18":
            self.backbone = resnet18(pretrained=False, hr=False)
        elif backbone == "resnet34":
            self.backbone = resnet34(pretrained=False, hr=False)
        elif backbone == "resnet50":
            self.backbone = resnet50(pretrained=False, hr=False)
        elif backbone == "resnet101":
            self.backbone = resnet101(pretrained=False, hr=False)
        elif backbone == "resnet152":
            self.backbone = resnet152(pretrained=False, hr=False)
        elif backbone == "resnext50_32x4d":
            self.backbone = resnext50_32x4d(pretrained=False, hr=False)
        elif backbone == "resnext101_32x8d":
            self.backbone = resnext101_32x8d(pretrained=False, hr=False)

        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1)
        )
        self.conv_1x1_3 = Conv(512, 256, k=1)
        self.extra_conv_3 = Conv(512, 1024, k=3, p=1)
        self.pred_3 = nn.Conv2d(1024, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv(768, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1)
        )
        self.conv_1x1_2 = Conv(256, 128, k=1)
        self.extra_conv_2 = Conv(256, 512, k=3, p=1)
        self.pred_2 = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv(384, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1)
        )
        self.extra_conv_1 = Conv(128, 256, k=3, p=1)
        self.pred_1 = nn.Conv2d(256, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

    def compute_grid_offsets(self, grid_size, grid_index=0, cuda=True):
        """
        Function：
            1、输出grid * grid的矩阵的index下标：self.grid_x,self.grid_y
            2、标准化anchor_w/h,即将anchor大小缩放到适合grid * grid大小
            额外说明：整个yolo中一张图片有三个衡量大小的坐标，分别是（1）原图、（2）原图/stride 或者说grid大小、（3）yolo网络预测的delta

        :param grid_size:int grid的size
        :param cuda:bool 是否使用cuda
        :param grid_index:int grid的index
        """
        self.grid_size[grid_index] = grid_size
        g = self.grid_size[grid_index]
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride[grid_index] = int(self.input_size / self.grid_size[grid_index])
        # Calculate offsets for each grid
        self.grid_x[grid_index] = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)  # size=(1, 1, g, g)， grid的x轴index
        self.grid_y[grid_index] = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor) # size=(1, 1, g, g)， grid的y轴index
        self.scaled_anchors[grid_index] = FloatTensor([(a_w / self.stride[grid_index], a_h / self.stride[grid_index]) for a_w, a_h in self.anchor_size])    # anchor除以stride
        self.anchor_w[grid_index] = self.scaled_anchors[grid_index][:, 0:1].view((1, self.num_anchors, 1, 1))   # size(1,3,1,1), 一个grid中三个anchor的w
        self.anchor_h[grid_index] = self.scaled_anchors[grid_index][:, 1:2].view((1, self.num_anchors, 1, 1))   # size(1,3,1,1), 一个grid中三个anchor的h

    def forward(self, x, targets=None):  # x:torch.Size([32, 3, 416, 416]), VOC
        # 1. 额外操作
        self.input_size = x.shape[2]
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # 2. forward网络
        c3, c4, c5 = self.backbone(x)   # c3:torch.Size([32, 256, 52, 52]), c4:torch.Size([32, 512, 26, 26]), c5:torch.Size([32, 1024, 13, 13])

        # FPN, 多尺度特征融合
        p5 = self.conv_set_3(c5)    # torch.Size([32, 512, 13, 13])
        p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)   # torch.Size([32, 256, 26, 26])

        p4 = torch.cat([c4, p5_up], 1)  # torch.Size([32, 768, 26, 26])
        p4 = self.conv_set_2(p4)    # torch.Size([32, 256, 26, 26])
        p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)   # torch.Size([32, 128, 52, 52])

        p3 = torch.cat([c3, p4_up], 1)  # torch.Size([32, 384, 52, 52])
        p3 = self.conv_set_1(p3)    # torch.Size([32, 128, 52, 52])

        # head
        # s = 32, 预测大物体
        p5 = self.extra_conv_3(p5)  # torch.Size([32, 1024, 13, 13])
        pred_3 = self.pred_3(p5)    # torch.Size([32, 75, 13, 13])

        # s = 16, 预测中物体
        p4 = self.extra_conv_2(p4)  # torch.Size([32, 512, 26, 26])
        pred_2 = self.pred_2(p4)    # torch.Size([32, 75, 26, 26])

        # s = 8, 预测小物体
        p3 = self.extra_conv_1(p3)  # torch.Size([32, 256, 52, 52])
        pred_1 = self.pred_1(p3)    # torch.Size([32, 75, 52, 52])

        # yolo head
        yolo_outputs = []
        yolo_loss = 0
        for i, pred in enumerate([pred_3, pred_2, pred_1]):  # layer1-3分别处理大/中/小物体, pred_1:torch.Size([32, 75, 52, 52]);pred_2:torch.Size([32, 75, 26, 26]);pred_3:torch.Size([32, 75, 52, 52])
            num_samples = pred.size(0)
            grid_size = pred.size(2)

            # 变形预测layer头, (B,num_anchors,grid_size,grid_size, 85)
            prediction = (
                pred.view(
                    num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size
                    ).permute(0, 1, 3, 4, 2).contiguous()
            )

            # Get outputs，normalized[cx,cy,w,h,pred_conf,pred_cls]
            x = torch.sigmoid(prediction[..., 0])  # Center x: sigma(tx)
            y = torch.sigmoid(prediction[..., 1])  # Center y: sigma(ty)
            w = prediction[..., 2]  # Width:tw
            h = prediction[..., 3]  # Height:th
            pred_conf = torch.sigmoid(prediction[..., 4])  # Conf:(0~1)
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred:(0~1)

            # If grid size does not match current we compute new offsets; 生成匹配的grid_x,grid_y,anchor_x,anchor_y
            if grid_size != self.grid_size[i]:
                self.compute_grid_offsets(grid_size, i, cuda=x.is_cuda)

            # Add offset and scale with anchors，相对于grid * grid 大小
            pred_boxes = FloatTensor(prediction[..., :4].shape)  # prediction size (B, 3, grid, grid, 85)
            pred_boxes[..., 0] = x.data + self.grid_x[i]    # bx = sigma(tx) + cx
            pred_boxes[..., 1] = y.data + self.grid_y[i]    # by = sigma(ty) + cy
            pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w[i]   # bw = pw * e^tw
            pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h[i]   # bh = ph * e^th

            # 通过(bx,by,bw,bh) & pred_conf & pred_cls --> size (B, num_anchors*grid*grid, 4+1+num_classes),全部是对应原始图像大小
            output = torch.cat(
                (
                    pred_boxes.view(num_samples, -1, 4) * self.stride[i],  # 使grid*grid大小的特征图 扩展到 原始大小，相对于图像原始大小; size(B,num_anchors*grid*grid,4)
                    pred_conf.view(num_samples, -1, 1),  # size(B,num_anchors*grid*grid,1)
                    pred_cls.view(num_samples, -1, self.num_classes),  # size(B,num_anchors*grid*grid,num_classes)
                ),
                -1,
            )

            if targets is None:
                yolo_outputs.append(output)
            else:
                iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                    pred_boxes=pred_boxes,  # (bx,by,bw,bh)，相对于grid * grid大小,Size(B, num_anchor, grid, grid, 4])
                    pred_cls=pred_cls,  # 预测的类别， Size(B, num_anchor, grid, grid, num_classes])
                    target=targets,  # 标签 size(N个bbox，6), 6代表的含义（batchsize id, cls id，cx（0～1，相对于整张图),cy（0～1，相对于整张图）,w（0～1，相对于整张图）,h（0～1，相对于整张图）
                    anchors=self.scaled_anchors[i],  # 相对于grid * grid大小的anchor
                    ignore_thres=self.ignore_thres,  # 忽略阈值
                )

                # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
                obj_mask = obj_mask.bool()  # convert int8 to bool
                noobj_mask = noobj_mask.bool()  # convert int8 to bool

                loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
                loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
                loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
                loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
                loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
                loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
                loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
                total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

                # Metrics
                cls_acc = 100 * class_mask[obj_mask].mean()
                conf_obj = pred_conf[obj_mask].mean()
                conf_noobj = pred_conf[noobj_mask].mean()
                conf50 = (pred_conf > 0.5).float()
                iou50 = (iou_scores > 0.5).float()
                iou75 = (iou_scores > 0.75).float()
                detected_mask = conf50 * class_mask * tconf
                precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
                recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
                recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

                self.metrics[i] = {
                    "loss": to_cpu(total_loss).item(),
                    "x": to_cpu(loss_x).item(),
                    "y": to_cpu(loss_y).item(),
                    "w": to_cpu(loss_w).item(),
                    "h": to_cpu(loss_h).item(),
                    "conf": to_cpu(loss_conf).item(),
                    "cls": to_cpu(loss_cls).item(),
                    "cls_acc": to_cpu(cls_acc).item(),
                    "recall50": to_cpu(recall50).item(),
                    "recall75": to_cpu(recall75).item(),
                    "precision": to_cpu(precision).item(),
                    "conf_obj": to_cpu(conf_obj).item(),
                    "conf_noobj": to_cpu(conf_noobj).item(),
                    "grid_size": grid_size,
                }
                yolo_outputs.append(output)
                yolo_loss += total_loss

        yolo_outputs = torch.cat(yolo_outputs, 1)
        return yolo_outputs if targets is None else (yolo_loss, yolo_outputs)


if __name__ == '__main__':
    import torch
    from dotmap import DotMap
    model_kwargs = DotMap({
        "type": "YOLOv3",
        "kwargs": {
            "backbone": "darknet53",
            "input_size": 416,
            'anchor_size': [[116, 90], [156, 198], [373, 326]],
            "num_classes": 80,
        }
    })
    x = torch.rand(16, 3, 416, 416).to(device="cuda:0")
    model = YOLOv3(**model_kwargs.kwargs).to(device="cuda:0")
    labels = torch.tensor([[ 0.0000,  9.0000,  0.5380,  0.5600,  0.3400,  0.2720],
        [ 0.0000, 14.0000,  0.2320,  0.5180,  0.1440,  0.2680],
        [ 0.0000, 14.0000,  0.1090,  0.4900,  0.2140,  0.3240],
        [ 0.0000, 14.0000,  0.8340,  0.3780,  0.2560,  0.4160],
        [ 1.0000,  9.0000,  0.4390,  0.6740,  0.1140,  0.3160],
        [ 1.0000, 14.0000,  0.5840,  0.6210,  0.3000,  0.4220],
        [ 2.0000,  2.0000,  0.5550,  0.6050,  0.5020,  0.3380],
        [ 3.0000,  6.0000,  0.7812,  0.4698,  0.4375,  0.3896],
        [ 3.0000,  6.0000,  0.1635,  0.4177,  0.3229,  0.3146],
        [ 3.0000,  6.0000,  0.5135,  0.5469,  0.6604,  0.4521],
        [ 4.0000, 11.0000,  0.9270,  0.4870,  0.1340,  0.1100],
        [ 4.0000, 16.0000,  0.0730,  0.4660,  0.1420,  0.2680],
        [ 4.0000, 16.0000,  0.2350,  0.4740,  0.1820,  0.2840],
        [ 4.0000, 16.0000,  0.3180,  0.4470,  0.1200,  0.2500],
        [ 5.0000,  0.0000,  0.2290,  0.5040,  0.2660,  0.0800],
        [ 5.0000,  0.0000,  0.6520,  0.4660,  0.2640,  0.1040],
        [ 6.0000, 16.0000,  0.7760,  0.4830,  0.4480,  0.5980],
        [ 6.0000, 16.0000,  0.4080,  0.4810,  0.5480,  0.4260],
        [ 7.0000,  4.0000,  0.5090,  0.6650,  0.2060,  0.1300],
        [ 7.0000,  8.0000,  0.8880,  0.4540,  0.2240,  0.4000],
        [ 7.0000, 10.0000,  0.6590,  0.4890,  0.4700,  0.3540],
        [ 7.0000, 14.0000,  0.7210,  0.7100,  0.3140,  0.2440],
        [ 7.0000, 14.0000,  0.3020,  0.5000,  0.6000,  0.6640],
        [ 8.0000,  1.0000,  0.3930,  0.5940,  0.1740,  0.2080],
        [ 8.0000, 14.0000,  0.4210,  0.5190,  0.2220,  0.2700],
        [ 9.0000,  1.0000,  0.6400,  0.5810,  0.2600,  0.5780],
        [ 9.0000,  1.0000,  0.3800,  0.5790,  0.2720,  0.4740],
        [ 9.0000, 14.0000,  0.6210,  0.3960,  0.3380,  0.6840],
        [ 9.0000, 14.0000,  0.3700,  0.4130,  0.3320,  0.6780],
        [10.0000, 14.0000,  0.1590,  0.6580,  0.0500,  0.1280],
        [11.0000,  4.0000,  0.7276,  0.5605,  0.0271,  0.0647],
        [11.0000,  8.0000,  0.5992,  0.7422,  0.2213,  0.2568],
        [11.0000,  8.0000,  0.5084,  0.7307,  0.1357,  0.2839],
        [11.0000,  8.0000,  0.2787,  0.5846,  0.1065,  0.0793],
        [11.0000, 10.0000,  0.4468,  0.7370,  0.5887,  0.2714],
        [11.0000,  8.0000,  0.3017,  0.7662,  0.2317,  0.2129],
        [11.0000,  8.0000,  0.1691,  0.7474,  0.0501,  0.2505],
        [11.0000, 15.0000,  0.1023,  0.2839,  0.1712,  0.1002],
        [12.0000, 11.0000,  0.5870,  0.5090,  0.3340,  0.1900],
        [12.0000,  4.0000,  0.2180,  0.9350,  0.0960,  0.1300],
        [12.0000, 14.0000,  0.4960,  0.4470,  0.5960,  0.7820],
        [13.0000,  7.0000,  0.4780,  0.5320,  0.4800,  0.5440],
        [14.0000,  7.0000,  0.2910,  0.5950,  0.3420,  0.1900],
        [14.0000, 14.0000,  0.6170,  0.2970,  0.2860,  0.1780],
        [15.0000, 14.0000,  0.4748,  0.5229,  0.0481,  0.1030],
        [15.0000, 14.0000,  0.4920,  0.5835,  0.0732,  0.1739]]).to(device="cuda:0")
    loss, outputs = model(x, targets=labels)

    model.eval()
    torch.onnx.export(model,
                      x,
                      "/ai/data/yolov3.onnx",
                      opset_version=11,
                      verbose=True,
                      input_names=["input"],
                      output_names=["output"])


