# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import darknet53
from modules import Conv
from tools import *

from dao.register import Registers


@Registers.det_models.register
class YOLOv3(nn.Module):
    def __init__(self, device="cpu", input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50, anchor_size=None, hr=False):
        """
        Function: 定义YoloV3网络

        :param device:
        :param input_size:
        :param num_classes:
        :param trainable:
        :param conf_thresh:
        :param nms_thresh:
        :param anchor_size:
        :param hr:
        """
        super(YOLOv3, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchor_size = anchor_size
        self.num_anchors = len(anchor_size)
        self.grid_size = [0, 0, 0]
        self.stride = [32, 16, 8]
        self.grid_x = [0,0,0]
        self.grid_y = [0,0,0]
        self.scaled_anchors = [0,0,0]
        self.anchor_w = [0,0,0]
        self.anchor_h = [0,0,0]

        # self.device = device
        # self.trainable = trainable
        # self.conf_thresh = conf_thresh
        # self.nms_thresh = nms_thresh
        # self.topk = 3000
        # self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)

        # backbone darknet-53
        self.backbone = darknet53(pretrained=False, hr=hr)

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

    # def create_grid(self, input_size):
    #     """
    #     Function: 通过input_size，获得layer1-3的total_grid_xy, total_stride, total_anchor_wh
    #     :param input_size:int
    #     :return:
    #     """
    #     total_grid_xy = []
    #     total_stride = []
    #     total_anchor_wh = []
    #     w, h = input_size, input_size
    #     for ind, s in enumerate(self.stride):
    #         # generate grid cells
    #         ws, hs = w // s, h // s  # grid大小
    #         grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
    #         grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    #         grid_xy = grid_xy.view(1, hs * ws, 1, 2)
    #
    #         # generate stride tensor
    #         stride_tensor = torch.ones([1, hs * ws, self.num_anchors, 2]) * s
    #
    #         # generate anchor_wh tensor
    #         anchor_wh = self.anchor_size[ind].repeat(hs * ws, 1, 1)
    #
    #         total_grid_xy.append(grid_xy)
    #         total_stride.append(stride_tensor)
    #         total_anchor_wh.append(anchor_wh)
    #
    #     total_grid_xy = torch.cat(total_grid_xy, dim=1).to(self.device)
    #     total_stride = torch.cat(total_stride, dim=1).to(self.device)
    #     total_anchor_wh = torch.cat(total_anchor_wh, dim=0).to(self.device).unsqueeze(0)
    #
    #     return total_grid_xy, total_stride, total_anchor_wh

    # def set_grid(self, input_size):
    #     """
    #     Function: 当input_size改变时，重新设置grid_cell, stride_tensor, all_anchors_wh
    #     :param input_size:
    #     :return:
    #     """
    #     self.input_size = input_size
    #     self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)

    # def decode_xywh(self, txtytwth_pred):
    #     """
    #     Function:
    #     将预测的tx,ty,tw,th的bboxes(中心点，宽高)   转成bx,by,bw,bh(中心点，宽高)
    #         Input:
    #             txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
    #         Output:
    #             xywh_pred : [B, H*W*anchor_n, 4] containing [x, y, w, h]
    #     """
    #     # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
    #     B, HW, ab_n, _ = txtytwth_pred.size()
    #     c_xy_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell) * self.stride_tensor
    #     # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
    #     b_wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchors_wh
    #     # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
    #     xywh_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HW * ab_n, 4)
    #
    #     return xywh_pred

    # def decode_boxes(self, txtytwth_pred):
    #     """
    #     Function:
    #     将预测的tx,ty,tw,th的bboxes(中心点，宽高)   转成   x1,y1,x2,y2的bboxes格式(做上角点，右下角点)
    #         Input:
    #             txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
    #         Output:
    #             x1y1x2y2_pred : [B, H*W, anchor_n, 4] containing [xmin, ymin, xmax, ymax]
    #     """
    #     # [B, H*W*anchor_n, 4]
    #     xywh_pred = self.decode_xywh(txtytwth_pred)
    #
    #     # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
    #     x1y1x2y2_pred = torch.zeros_like(xywh_pred)
    #     x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
    #     x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
    #     x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
    #     x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)
    #
    #     return x1y1x2y2_pred

    # def nms(self, dets, scores):
    #     """
    #     Function:Pure Python NMS baseline. 非极大值抑制
    #
    #     :param dets:
    #     :param scores:
    #     :return:
    #     """
    #     x1 = dets[:, 0]  # xmin
    #     y1 = dets[:, 1]  # ymin
    #     x2 = dets[:, 2]  # xmax
    #     y2 = dets[:, 3]  # ymax
    #
    #     areas = (x2 - x1) * (y2 - y1)  # the size of bbox
    #     order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order
    #
    #     keep = []  # store the final bounding boxes
    #     while order.size > 0:
    #         i = order[0]  # the index of the bbox with highest confidence
    #         keep.append(i)  # save it to keep
    #         xx1 = np.maximum(x1[i], x1[order[1:]])
    #         yy1 = np.maximum(y1[i], y1[order[1:]])
    #         xx2 = np.minimum(x2[i], x2[order[1:]])
    #         yy2 = np.minimum(y2[i], y2[order[1:]])
    #
    #         w = np.maximum(1e-28, xx2 - xx1)
    #         h = np.maximum(1e-28, yy2 - yy1)
    #         inter = w * h
    #
    #         # Cross Area / (bbox + particular area - Cross Area)
    #         ovr = inter / (areas[i] + areas[order[1:]] - inter)
    #         # reserve all the boundingbox whose ovr less than thresh
    #         inds = np.where(ovr <= self.nms_thresh)[0]
    #         order = order[inds + 1]
    #
    #     return keep

    # def postprocess(self, bboxes, scores):
    #     """
    #     Function: 后处理
    #     bboxes: (HxW, 4), bsize = 1
    #     scores: (HxW, num_classes), bsize = 1
    #     """
    #
    #     cls_inds = np.argmax(scores, axis=1)
    #     scores = scores[(np.arange(scores.shape[0]), cls_inds)]
    #
    #     # threshold
    #     keep = np.where(scores >= self.conf_thresh)
    #     bboxes = bboxes[keep]
    #     scores = scores[keep]
    #     cls_inds = cls_inds[keep]
    #
    #     # NMS
    #     keep = np.zeros(len(bboxes), dtype=np.int)
    #     for i in range(self.num_classes):
    #         inds = np.where(cls_inds == i)[0]
    #         if len(inds) == 0:
    #             continue
    #         c_bboxes = bboxes[inds]
    #         c_scores = scores[inds]
    #         c_keep = self.nms(c_bboxes, c_scores)
    #         keep[inds[c_keep]] = 1
    #
    #     keep = np.where(keep > 0)
    #     bboxes = bboxes[keep]
    #     scores = scores[keep]
    #     cls_inds = cls_inds[keep]
    #
    #     # topk
    #     scores_sorted, scores_sorted_inds = np.sort(scores), np.argsort(scores)
    #     topk_scores, topk_scores_inds = scores_sorted[:self.topk], scores_sorted_inds[:self.topk]
    #     topk_bboxes = bboxes[topk_scores_inds]
    #     topk_cls_inds = cls_inds[topk_scores_inds]
    #
    #     return topk_bboxes, topk_scores, topk_cls_inds

    def compute_grid_offsets(self, grid_size, grid_index=0, cuda=True):
        """
        Function：
            1、输出grid * grid的矩阵
            2、标准化anchor_w/h,即将anchor大小缩放到适合grid * grid大小
        :param grid_size:
        :param cuda:
        :return:
        """
        self.grid_size[grid_index] = grid_size
        g = self.grid_size[grid_index]
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride[grid_index] = int(self.input_size / self.grid_size[grid_index])
        # Calculate offsets for each grid
        self.grid_x[grid_index] = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)     # size=(1, 1, g, g)
        self.grid_y[grid_index] = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors[grid_index] = FloatTensor([(a_w / self.stride[grid_index], a_h / self.stride[grid_index]) for a_w, a_h in self.anchor_size])
        self.anchor_w[grid_index] = self.scaled_anchors[grid_index][:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h[grid_index] = self.scaled_anchors[grid_index][:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, target=None):  # x:torch.Size([32, 3, 416, 416]), VOC
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
            x = torch.sigmoid(prediction[..., 0])  # Center x
            y = torch.sigmoid(prediction[..., 1])  # Center y
            w = prediction[..., 2]  # Width
            h = prediction[..., 3]  # Height
            pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

            # If grid size does not match current we compute new offsets
            if grid_size != self.grid_size[i]:
                self.compute_grid_offsets(grid_size, i, cuda=x.is_cuda)

            # Add offset and scale with anchors，相对于grid * grid 大小
            pred_boxes = FloatTensor(prediction[..., :4].shape)  # prediction size (B, 3, grid, grid, 85)
            pred_boxes[..., 0] = x.data + self.grid_x[i]
            pred_boxes[..., 1] = y.data + self.grid_y[i]
            pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w[i]
            pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h[i]

            # 通过   预测的（cx, cy, w, h)和anchor的（w, h)   计算出 预测框pred_boxes size=(20, 3, 15, 15, 4)
            output = torch.cat(
                (
                    pred_boxes.view(num_samples, -1, 4) * self.stride[i],  # 使grid*grid大小的特征图 扩展到 原始大小，相对于图像原始大小
                    pred_conf.view(num_samples, -1, 1),
                    pred_cls.view(num_samples, -1, self.num_classes),
                ),
                -1,
            )
            B_, abC_, H_, W_ = pred.size()


            # 对pred 的size做一些view调整，便于后续的处理
            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B_, H_ * W_, abC_)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
            # [B, H*W*anchor_n, 1]
            conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B_, H_ * W_ * self.num_anchors, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(B_, H_ * W_ * self.num_anchors, self.num_classes)
            # [B, H*W, 4*anchor_n]
            txtytwth_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()

            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_txtytwth_pred.append(txtytwth_pred)
            B = B_
            HW += H_ * W_

        # 将所有结果沿着H*W这个维度拼接
        conf_pred = torch.cat(total_conf_pred, dim=1)   # torch.Size([32, 10647, 1])
        cls_pred = torch.cat(total_cls_pred, dim=1)  # torch.Size([32, 10647, 20])
        txtytwth_pred = torch.cat(total_txtytwth_pred, dim=1)   # torch.Size([32, 3549, 12])

        # train or test
        if self.trainable == 0:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.num_anchors, 4)

            # 从txtytwth预测中解算出x1y1x2y2坐标
            x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
            # 计算pred box与gt box之间的IoU
            iou_pred = iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)

            # gt conf，这一操作是保证iou不会回传梯度
            with torch.no_grad():
                gt_conf = iou_pred.clone()

            # 我们讲pred box与gt box之间的iou作为objectness的学习目标.
            # [obj, cls, txtytwth, scale_weight, x1y1x2y2] -> [conf, obj, cls, txtytwth, scale_weight]
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)
            txtytwth_pred = txtytwth_pred.view(B, -1, 4)

            # 计算loss
            conf_loss, cls_loss, bbox_loss, iou_loss = loss(pred_conf=conf_pred,
                                                                  pred_cls=cls_pred,
                                                                  pred_txtytwth=txtytwth_pred,
                                                                  pred_iou=iou_pred,
                                                                  label=target
                                                                  )

            return conf_loss, cls_loss, bbox_loss, iou_loss

            # test
        elif self.trainable == 1:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.num_anchors, 4)
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，
                # 因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W*num_anchor, 4] -> [H*W*num_anchor, 4]
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W*num_anchor, C] -> [H*W*num_anchor, C],
                scores = torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred

                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds
        else:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.num_anchors, 4)
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，
                # 因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W*num_anchor, 4] -> [H*W*num_anchor, 4]
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W*num_anchor, C] -> [H*W*num_anchor, C],
                scores = torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred
            return scores


if __name__ == '__main__':
    import torch
    from dotmap import DotMap
    model_kwargs = DotMap({
        "type": "YOLOv3",
        "kwargs": {
            "input_size": 416,
            'anchor_size': [[116, 90], [156, 198], [373, 326]],
            "num_classes": 80,
            "trainable": 1,
            "conf_thresh": 0.001,
            "nms_thresh": 0.50,
            "hr": False
        }
    })
    x = torch.rand(8, 3, 416, 416).to(device="cuda:0")
    model = YOLOv3(device="cuda:0", **model_kwargs.kwargs).to(device="cuda:0")
    # print(model)
    output = model(x)
    # print(output.shape)

    model.eval()
    torch.onnx.export(model,
                      x,
                      "/ai/data/yolov3.onnx",
                      opset_version=11,
                      verbose=True,
                      input_names=["input"],
                      output_names=["scores"])


