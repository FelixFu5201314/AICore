# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, g=1, act=True):
        """
        Function: DBL or DB

        :param in_ch:
        :param out_ch:
        :param k:
        :param p:
        :param s:
        :param d:
        :param g:
        :param act:
        """
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return self.convs(x)


def bbox_wh_iou(wh1, wh2):
    """
    Function：计算iou，真实标签与anchor（3个）

    :param wh1: anchor的 (w,h) 相对于grid*grid大小
    :param wh2: 真实标签的(w,h) 相对于grid*grid大小， size(num_bboxes, 2)
    :return:
    """
    wh2 = wh2.t()   # size (2, num_bboxes)
    w1, h1 = wh1[0], wh1[1]  # w1 和 h1 是标量
    w2, h2 = wh2[0], wh2[1]  # w2 和 h2 都是181大小的向量
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """
    Function: 通过pred_boxes, pred_cls, target 构建目标targets
        额外说明：整个yolo中一张图片有三个衡量大小的坐标，分别是
        （1）相对于原图
        （2）相对于原图/stride 或者说grid大小
        （3）相对于yolo网络预测的delta

    :param pred_boxes:(B, num_anchors, grid, grid, 4)  (bx,by,bw,bh)，相对于grid * grid大小
    :param pred_cls:(B, num_anchors, grid, grid, 80)    预测的类别
    :param target:(num_bbox, 6)  # 标签中的bbox等信息， 6的含义（batchsize id, cls，cx（0～1，相对于整张图),cy（0～1，相对于整张图）,w（0～1，相对于整张图）,h（0～1，相对于整张图）
    :param anchors:(3, 2)   # 相对于grid * grid大小的anchor大小
    :param ignore_thres:0.5
    :return:
    """
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # number of batchsize
    nA = pred_boxes.size(1)  # number of anchor
    nC = pred_cls.size(-1)   # number of class
    nG = pred_boxes.size(2)  # size of grid

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)      # size (B, num_anchors, grid, grid)  有obj的mask
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)    # size (B, num_anchors, grid, grid)  无obj的mask
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)   # size (B, num_anchors, grid, grid)  每个grid 类别的mask
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)   # size (B, num_anchors, grid, grid)  每个grid iou
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)           # size (B, num_anchors, grid, grid)  tx (target标签中真实的)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)           # size (B, num_anchors, grid, grid)  ty (target标签中真实的)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)           # size (B, num_anchors, grid, grid)  tw (target标签中真实的)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)           # size (B, num_anchors, grid, grid)  th (target标签中真实的)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)     # size (B, num_anchors, grid, grid, num_classes) 类别（target标签中真实的）

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG  # 将target的norm(cx,cy,w,h)改为norm(cx,cy,w,h)* grid， target_boxes相对于grid * grid大小
    gxy = target_boxes[:, :2]   # ground truth xy （num_bboxes， 2），相对于grid * grid 大小
    gwh = target_boxes[:, 2:]   # ground truth wh  （num_bboxes， 2），相对于grid * grid 大小

    # Get anchors with best iou，计算真实标签的bboxes与anchors（本例是3个）iou最大的一个。选择真实标签和那个anchors中对应。
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])    # size(3, num_bboxes) 此处anchor相对于grid * grid， 真实g和anchor交集
    best_ious, best_n = ious.max(0)  # ious真实标签框和3个anchor的iou，然后best_iou是3个anchor之中与真实标签iou最好的，best_n对应第几个anchor

    # Separate target values
    b, target_labels = target[:, :2].long().t()     # b表示batchsize id，target_labels表示此bbox的类别
    gx, gy = gxy.t()    # gx真实标签的x size为num_bboxes; gy真实标签的y; 相对于grid*grid
    gw, gh = gwh.t()    # gw 真实标签的w, gh真实标签的h
    gi, gj = gxy.long().t()  # grid的i和j

    # Set masks，target和3个anchor中iou最大的一个（3选1），设置成有obj。其余2个anchor依旧是无obj
    obj_mask[b, best_n, gj, gi] = 1     # 为什么标记的数小于真实标签数，例如target的数量是（129，6），而int(obj_mask[obj_mask==1].size(0)) = 114
    noobj_mask[b, best_n, gj, gi] = 0   # noobj设置为0，即有obj或者忽略

    # Set noobj mask to zero where iou exceeds ignore threshold，
    # best_ious是最大的IOU，但是ious中还有很多是超过ignore_thres的。所以要忽略
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # 真实标签，调整到和pred相同的格式，即delta x/y，
    # Coordinates, (cx,cy)
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)   # gw size 179 相对于grid*grid， anchors:(3, 2)相对于grid * grid大小
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1  # size torch.Size([20, 3, 11, 11, 80])

    # Compute label correctness and iou at best anchor，计算标签的正确性和iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # pred_cls:(20, 3, 15, 15, 80) 预测的类别,
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    # iou_scores torch.Size([B, 3, grid, grid]) ,pred_boxes 和 target_boxes的iou， 对应每个grid * grid

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def to_cpu(tensor):
    return tensor.detach().cpu()