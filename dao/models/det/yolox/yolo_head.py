#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from dao.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)     # torch.Size([16, 85, 80, 80])
                output, grid = self.get_output_and_grid(    # 把output变成相对于整张图的
                    output, k, stride_this_level, xin[0].type()
                )   # output:torch.Size([16, 6400, 85]); grid:torch.Size([1, 6400, 2])
                x_shifts.append(grid[:, :, 0])  # { [6400], ...}
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )   # { [6400] value is 8, ... }
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )  # torch.Size([16, 1, 4, 80, 80])
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )   # torch.Size([16, 6400, 4])
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)  # 添加多个layer到outputs中，即不同的尺度80*80, 40*40, 20*20三种尺度的结果到outputs中；到此，outputs收集了所有layer输出

        if self.training:
            return self.get_losses(
                imgs,   # 原始输入图片torch.Size([16, 3, 640, 640])
                x_shifts,   # grid的x轴坐标{torch.Size([1, 6400]), torch.Size([1, 1600]), torch.Size([1, 400])]
                y_shifts,   # grid的y轴坐标{torch.Size([1, 6400]), torch.Size([1, 1600]), torch.Size([1, 400])]
                expanded_strides,  # grid中(x,y)点缩小步长{torch.Size([1, 6400]) value 8, torch.Size([1, 1600]) value 16, torch.Size([1, 400]) value 32]
                labels,     # 标签 torch.Size([16, 50, 5])
                torch.cat(outputs, 1),  # 三个layer的输出， torch.Size([16, 8400, 85])
                origin_preds,   # {torch.Size([16, 6400, 4]), torch.Size([16, 1600, 4]), torch.Size([16, 400, 4])]
                dtype=xin[0].dtype,  # torch.float16
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])     # 划分为单元网格
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)    # torch.Size([16, 85, 80, 80])->torch.Size([16, 1, 85, 80, 80])
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )   # torch.Size([16, 1, 85, 80, 80])->torch.Size([16, 6400, 85])
        grid = grid.view(1, -1, 2)  # torch.Size([1, 1, 80, 80, 2])->torch.Size([1, 6400, 2])
        output[..., :2] = (output[..., :2] + grid) * stride  # output中的deta_x,deta_y变成相对于整个input_size的位置
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  # output中的width,height变成相对于整个input_size大小
        return output, grid

    # 网络输出包括3个尺度， stride分别是8、16 和 32；
    # 每个输出尺度上又包括3个输出，分别是bbox输出分支、objectness输出分支和cls类别输出分支。
    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            # 计算3个输出层所需要的特征图尺度的坐标，用于bbox解码
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])  # 划分为单元网格
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # 对输出bbox进行解码还原到原图尺度
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    # 分类分支和objectness分支采用bce loss，bbox预测分支采用IoU Loss。
    # 分类分支仅考虑正样本即fg_mask为1的位置，其label同时考虑了预测值和gt bbox的IoU值，用于加强各分支间的一致性;
    # objectness分支需要同时考虑正负样本(8400个)，即所有候选点，起到抑制背景的作用，其label为fg_mask, 非零即1;
    # bbox分支也仅仅考虑正样本，其label就是正样本候选点所对应的解码后的预测值;
    # 附加L1 Loss: 在最后15个epoches，加入了额外的L1 Loss。其作用的对象是原始没有解码的正样本bbox预测值，和对应的gt bbox。

    # 从以上分析可知：分类分支不考虑背景，背景预测功能由objectness分支提供，
    # 而bbox分支联合采用了IoU Loss和L1 Loss，其最大改进在于动态匹配。

    # 在Decoupled Head中，cls_output和obj_output使用了sigmoid函数进行归一化，
    # 但是在训练时，没有使用sigmoid函数，原因是训练时用的nn.BCEWithLogitsLoss函数，已经包含了sigmoid操作。
    def get_losses(
        self,
        imgs,   # 原始输入图片torch.Size([16, 3, 640, 640])
        x_shifts,  # grid的x轴坐标{torch.Size([1, 6400]), torch.Size([1, 1600]), torch.Size([1, 400])]
        y_shifts,  # grid的y轴坐标{torch.Size([1, 6400]), torch.Size([1, 1600]), torch.Size([1, 400])]
        expanded_strides,  # grid中(x,y)点缩小步长{torch.Size([1, 6400]) value 8, torch.Size([1, 1600]) value 16, torch.Size([1, 400]) value 32]
        labels,  # 标签 torch.Size([16, 50, 5])
        outputs,  # 三个layer的输出， torch.Size([16, 8400, 85])
        origin_preds,  # {torch.Size([16, 6400, 4]), torch.Size([16, 1600, 4]), torch.Size([16, 400, 4])]
        dtype,  # torch.float16
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]  # 8400
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all] torch.Size([1, 8400])
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all] torch.Size([1, 8400])
        expanded_strides = torch.cat(expanded_strides, 1)   # torch.Size([1, 8400])
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)   # torch.Size([16, 8400, 4])

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0   # 共有多少个gt

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])     # 第batch_idx有多少个bbox
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))   # torch.Size([0, 80])
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]   # gt的bbox torch.Size([3, 4])
                gt_classes = labels[batch_idx, :num_gt, 0]  # gt的类别
                bboxes_preds_per_image = bbox_preds[batch_idx]  # torch.Size([8400, 4])

                try:
                    (
                        gt_matched_classes,  # gt_matched_classes:提取每个候选点匹配上的gt框信息gt_matched_classes, shape 103
                        fg_mask,  # fg_mask/is_in_boxes_anchor:anchor在gt框内的掩码或在center中，并集 shape是8400；
                        pred_ious_this_matching,  # pred_ious_this_matching:提取对应的预测点和gt框的iou, shape 103
                        matched_gt_inds,  # matched_gt_inds:提取匹配上的gt框索引，即该候选框匹配到哪个gt框, shape 103
                        num_fg_img,  # num_fg: 总共有多少候选锚点, shape 103;
                    ) = self.get_assignments(  # noqa， 标签匹配、标签分配(label assignment)
                        batch_idx,  # batch id 0
                        num_gt,  # bbox个数 3
                        total_num_anchors,   # 预测的anchor个数 8400
                        gt_bboxes_per_image,    # num_gt个bbox torch.Size([3, 4])
                        gt_classes,  # gt的class id， 长度为num_gt , [3,5,23]
                        bboxes_preds_per_image,     # 预测的bbox， torch.Size([8400, 4])
                        expanded_strides,  # grid中(x,y)点缩小步长torch.Size([1, 8400])
                        x_shifts,  # grid的x轴坐标torch.Size([1, 8400])
                        y_shifts,  # grid的y轴坐标torch.Size([1, 8400])
                        cls_preds,      # 预测的cls torch.Size([16, 8400, 80])
                        bbox_preds,     # 预测的bbox torch.Size([16, 8400, 4])
                        obj_preds,      # 预测obj torch.Size([16, 8400, 1])
                        labels,         # gt torch.Size([16, 50, 5])
                        imgs,           # 输入的图片 torch.Size([16, 3, 640, 640])
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)   # torch.Size([103, 80])
                obj_target = fg_mask.unsqueeze(-1)  # torch.Size([8400, 1])
                reg_target = gt_bboxes_per_image[matched_gt_inds]   # torch.Size([103, 4])
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)  # torch.Size([2275, 80])
        reg_targets = torch.cat(reg_targets, 0)  # torch.Size([2275, 4])
        obj_targets = torch.cat(obj_targets, 0)  # torch.Size([134400, 1])
        fg_masks = torch.cat(fg_masks, 0)        # shape 134400
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,  # batch id 0
        num_gt,  # bbox个数 3
        total_num_anchors,   # 预测的anchor个数 8400
        gt_bboxes_per_image,    # num_gt个bbox torch.Size([3, 4])
        gt_classes,  # gt的class id， 长度为num_gt
        bboxes_preds_per_image,     # 预测的bbox， torch.Size([8400, 4])
        expanded_strides,  # grid中(x,y)点缩小步长torch.Size([1, 8400])
        x_shifts,  # grid的x轴坐标torch.Size([1, 8400])
        y_shifts,  # grid的y轴坐标torch.Size([1, 8400])
        cls_preds,      # 预测的cls torch.Size([16, 8400, 80])
        bbox_preds,     # 预测的bbox torch.Size([16, 8400, 4])
        obj_preds,      # 预测obj torch.Size([16, 8400, 1])
        labels,         # gt torch.Size([16, 50, 5])
        imgs,           # 输入的图片 torch.Size([16, 3, 640, 640])
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # 挑选正样本锚点的初步筛选
        # 初步筛选有两种：根据gt框来判断、根据中心区域来判断
        # is_in_boxes_and_center
        # 如果某个位置是True表示该anchor点落在gt框内部并且在距离gt框中心center_radius半径范围内
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,  # num_gt个bbox torch.Size([3, 4])
            expanded_strides,  # grid中(x,y)点缩小步长torch.Size([1, 8400])
            x_shifts,  # grid的x轴坐标torch.Size([1, 8400])
            y_shifts,   # grid的y轴坐标torch.Size([1, 8400])
            total_num_anchors,  # 预测的anchor个数 8400
            num_gt,  # bbox个数 3
        )   # fg_mask/is_in_boxes_anchor:anchor在gt框内的掩码或在center中，并集 shape是8400； is_in_boxes_and_center:anchor在gt框内又在以gt框中心为中心的正方形（边长为5）内的anchor掩码 shape为(3, 1750)

        # fg_mask就是前面计算出的is_in_boxes_anchor，如果某个位置是True代表该anchor点是前景
        # 即落在gt框内部 或 在距离gt框中心center_radius半径范围内，这些为True位置就是正样本候选点

        # 提取对应值:
        # 利用fg_mask提取对应的预测信息，例如假设num_gt是3，一共提取了500个候选预测位置，
        # 则每个gt框都会提取出500个候选位置。
        # 所有锚点的位置和网络最后输出的85*8400特征向量是一一对应的，
        # 根据位置，可以将网络预测的候选检测框位置bboxes_preds、目标性得分obj_preds、类别得分cls_preds等信息提取出来。
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]    # 候选的anchor的bbox torch.Size([1750, 4]), 根据fg_mask从bbox_preds_per_image（8400，4）筛选出torch.Size([7826, 4])个anchor候选
        cls_preds_ = cls_preds[batch_idx][fg_mask]  # 候选的anchor的类别得分 torch.Size([1750, 80])
        obj_preds_ = obj_preds[batch_idx][fg_mask]  # 候选的anchor的置信度 torch.Size([1750, 1])
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]   # 候选anchor个数 1750

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # 计算预测框和gt框的配对iou， torch.Size([3, 4]) torch.Size([1750, 4])-》torch.Size([3, 1750])
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)     # gt_classes:3 --> torch.Size([3, 80])
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )   # torch.Size([3, 1750, 80])
        # iou越大，匹配度越高，相应iou loss需要取负号， torch.Size([3, 1750])， 这个就是cost matrix，有3个gt，1750个anchor
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()  # 候选anchor的cls预测，torch.Size([1750, 80])-->torch.Size([3, 1750, 80])
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()   # 候选anchor的obj预测， torch.Size([1750, 1])->torch.Size([3, 1750, 1])
            )   # torch.Size([3, 1750, 80])
            # 配对的分类Loss，包括了iou分支预测值;
            # 其分类cost在 binary_cross_entropy前有开根号的训练trick
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)   # torch.Size([3, 1750])
        del cls_preds_

        # 计算每个gt框和选择出来的候选预测框的 分类loss + 坐标loss + 中心点和半径约束
        # 值越小，表示匹配度越高
        # (num_gt,n)
        # 在计算代价函数时，如果该预测点是False，表示不在交集is_in_boxes_anchor内部，那么应该不太可能是候选点，
        # 所以给予一个非常大的代价权重 100000.0，该操作可以保证每个gt框最终选择的候选点不会在交集外部
        # 另外，代码中的3.0是论文公式中的加权系数
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )   # torch.Size([3, 1750])

        # 上述计算出的代价值充分考虑了各个分支预测值，也考虑了中心先验，有利于训练稳定和收敛，
        # 同时也为后续的动态匹配提供了全局信息。

        # 挑选正样本锚点的SimOTA
        # dynamic_k_matching： 为每个gt框动态选择k个候选预测值，作为匹配正样本
        (
            num_fg,        # num_fg: 总共有多少候选锚点, shape 103;
            gt_matched_classes,  # gt_matched_classes:提取每个候选点匹配上的gt框信息gt_matched_classes, shape 103
            pred_ious_this_matching,  # pred_ious_this_matching:提取对应的预测点和gt框的iou, shape 103
            matched_gt_inds,  # matched_gt_inds:提取匹配上的gt框索引，即该候选框匹配到哪个gt框, shape 103
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)  # cost: torch.Size([3, 1750]), pair_wise_ious:torch.Size([3, 1750]),, gt_classes:长度为3的列表， num_gt=3, fg_mask:长度为8400的anchor掩码
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,  # gt_matched_classes:提取每个候选点匹配上的gt框信息gt_matched_classes, shape 103
            fg_mask,  # fg_mask/is_in_boxes_anchor:anchor在gt框内的掩码或在center中，并集 shape是8400；
            pred_ious_this_matching,  # pred_ious_this_matching:提取对应的预测点和gt框的iou, shape 103
            matched_gt_inds,  # matched_gt_inds:提取匹配上的gt框索引，即该候选框匹配到哪个gt框, shape 103
            num_fg,   # num_fg: 总共有多少候选锚点, shape 103;
        )

    # 挑选正样本锚点：初步筛选 + SimOTA
    # 初步筛选有两种：根据gt框来判断、根据中心区域来判断
    # (1) 根据gt框来判断：挑选处于gt框矩形范围的anchors
    # (2) 根据中心区域来判断：挑选以gt中心点为基准，边长为5的正方形内的anchors
    # is_in_boxes_and_center
    # 如果某个位置是True表示该anchor点落在gt框内部并且在距离gt框中心center_radius半径范围内。
    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,   # num_gt个bbox torch.Size([3, 4])
        expanded_strides,  # grid中(x,y)点缩小步长torch.Size([1, 8400])
        x_shifts,  # grid的x轴坐标torch.Size([1, 8400])
        y_shifts,  # grid的y轴坐标torch.Size([1, 8400])
        total_num_anchors,   # 预测的anchor个数 8400
        num_gt,    # bbox个数 3
    ):
        expanded_strides_per_image = expanded_strides[0]    # 8400个anchor的步长， shape 8400
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image   # 8400个anchor 的x
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image   # 8400个anchor 的y
        # 获得gt框的[x_center,y_center，w，h]
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor] torch.Size([3, 8400])，得到每个anchor的center x
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # torch.Size([3, 8400])，得到每个anchor的center y

        # 通过gt框的[x_center, y_center，w，h]，计算出每张图片的每个gt框的左上角、右下角坐标
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )   # torch.Size([3, 8400])
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # 计算锚点相对于gt的偏移值
        b_l = x_centers_per_image - gt_bboxes_per_image_l   # bbox left 点torch.Size([3, 8400]) - torch.Size([3, 8400])
        b_r = gt_bboxes_per_image_r - x_centers_per_image   # bbox right 点torch.Size([3, 8400]) - torch.Size([3, 8400])
        b_t = y_centers_per_image - gt_bboxes_per_image_t   # bbox top 点torch.Size([3, 8400]) - torch.Size([3, 8400])
        b_b = gt_bboxes_per_image_b - y_centers_per_image   # bbox bottom 点torch.Size([3, 8400]) - torch.Size([3, 8400])
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # bbox的deta值 torch.Size([3, 8400, 4])

        # 判断偏移值是否都大于0，将落在gt框矩形范围内的anchors提取出来。
        # 只有落在gt矩形范围内的anchor box的中心点，这时的b_l，b_r，b_t，b_b才都大于0。
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # torch.Size([3, 8400]) bool值
        # 计算所有在gt框内部的anchor点的掩码is_in_boxes_all
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        # 引入了超参center_radius = 2.5
        center_radius = 2.5

        # 通过gt的[x_center，y_center，w，h]，绘制一个边长为5的正方形
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        # 计算锚点相对于正方形的偏移值
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        # 判断c_l，c_r，c_t，c_b是否都大于0，将处于边长为5的正方形范围内的anchors提取出来
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # torch.Size([3, 8400])
        # 利用center_radius阈值重新计算在gt框中心center_radius范围内的anchor点的掩码is_in_centers_all
        is_in_centers_all = is_in_centers.sum(dim=0) > 0    # torch.Size([8400])

        # in boxes and in centers
        # 两个掩码取并集得到在gt框内部或处于center_radius范围内的anchor点的掩码is_in_boxes_anchor
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all  # is_in_boxes_anchor 8400，但是里面是true的个数可能大于is_in_boxes_all和is_in_center_all中true的个数之和

        # 同时可以取交集得到每个gt框和哪些anchor点符合gt框内部和处于center_radius范围内的anchor点的掩码is_in_boxes_and_center
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center   # is_in_boxes_anchor:anchor在gt框内的掩码或在center中，并集 shape是8400； is_in_boxes_and_center:anchor在gt框内又在以gt框中心为中心的正方形（边长为5）内的anchor掩码 shape为(3, 1750)

    # 为每个gt框动态选择k个候选预测值，作为匹配正样本
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):   # cost: torch.Size([3, 1750]), pair_wise_ious:torch.Size([3, 1750]),, gt_classes:长度为3的列表， num_gt=3, fg_mask:长度为8400的anchor掩码
        # Dynamic K
        # ---------------------------------------------------------------
        # 初始化gt框和候选点的匹配矩阵为全0，表示全部不匹配
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)     # torch.Size([3, 1750])

        ious_in_boxes_matrix = pair_wise_ious  # torch.Size([3, 1750])
        # 每个gt框选择的候选预测点不超过10个
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        # 从前面的pair_wise_ious中，给每个目标框，挑选10个iou最大的候选框
        # 假定有3个目标，topk_ious的维度为[3，10]
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)   # ious_in_boxes_matrix:torch.Size([3, 1750])
        # 利用前面的匹配代价，给每个gt bbox计算动态k; 长度为3，即每个gt框对应几个anchor，eg. [6, 8, 8]为第一个gt框有6个anchor
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            # 遍历每个gt bbox，提取代价为前动态k个位置，表示匹配上
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            # 匹配上位置设置为1.0
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            # 过滤共用的候选锚点：
            # 如果某个候选锚点匹配了多个gt框，则选择代价最小的，保证每个候选锚点只匹配一个gt框
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        # 每个候选锚点的匹配情况
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        # 总共有多少候选锚点
        num_fg = fg_mask_inboxes.sum().item()

        # 更新前景掩码，在前面中心先验的前提下进一步筛选正样本
        # 更新后的前景掩码fg_mask，其长度和预测点个数相同，其中1表示正样本点，0表示负样本点。
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # 提取匹配上的gt框索引，即该候选框匹配到哪个gt框
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # 提取每个候选点匹配上的gt框信息gt_matched_classes
        gt_matched_classes = gt_classes[matched_gt_inds]

        # 提取对应的预测点和gt框的iou
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        # num_fg: 总共有多少候选锚点, shape 103;
        # gt_matched_classes:提取每个候选点匹配上的gt框信息gt_matched_classes, shape 103
        # pred_ious_this_matching:提取对应的预测点和gt框的iou, shape 103
        # matched_gt_inds:提取匹配上的gt框索引，即该候选框匹配到哪个gt框, shape 103
