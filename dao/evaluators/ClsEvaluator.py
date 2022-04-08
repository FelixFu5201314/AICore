#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import cv2
import time
import shutil
import itertools
import xlsxwriter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from copy import deepcopy


import torch
from torchcam.methods import SmoothGradCAMpp, CAM
from torchcam.utils import overlay_mask
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image

from dao.utils import MeterClsEval
from dao.utils import is_main_process, synchronize, time_synchronized, gather

from dao.register import Registers


@Registers.evaluators.register
class ClsEvaluator:
    def __init__(self,
                 is_distributed=False,
                 dataloader=None,
                 num_classes=None,
                 is_industry=False,
                 target_layer="conv_head"):
        """
        验证器
        is_distributed:bool 是否是分布式
        dataloader:dict dataloader的配置字典
        num_classes:int 类别数
        is_industry:bool 是否使用工业方法验证，即输出过漏检
        industry:dict 使用工业验证方法所需的参数
        """
        # 获取Dataloader
        self.dataloader = Registers.dataloaders.get(dataloader.type)(
            is_distributed=is_distributed,
            dataset=dataloader.dataset,
            **dataloader.kwargs
        )
        self.iters_per_epoch = len(self.dataloader)

        self.meter = MeterClsEval(num_classes)
        self.num_class = num_classes
        self.is_industry = is_industry
        self.target_layer = target_layer

        # 获取labels字典
        m = self.dataloader.dataset.labels_dict
        self.labels_ = dict(zip(m.values(), m.keys()))
        self.class_names = list(m.keys())
        # 定义混淆矩阵
        # self.confusion_matrix = [[0 for j in range(num_classes)] for i in range(num_classes)]
        self.pred_target_list = []  # 存放混淆矩阵中所用到的数据
        # self.pred_target_list_tolerate_count = {}  # 计算容忍混淆矩阵使用
        self.excel_ng = []
        self.excel_ok = []

    @logger.catch
    def evaluate(self, model, distributed=False, half=False, device=None, output_dir=None):
        """

        :param model: 模型
        :param distributed: 是否是分布式
        :param half: 是否是半精度
        :param device: 当前rank的device id
        :param output_dir: 实现的输出路径，trainer的self.output_dir属性
        :return:
        """
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        if half:
            model = model.half()
        data_list = []  # 存放最终的结果，多个gpu上汇集来的结果
        # progress_bar = tqdm if is_main_process() else iter
        progress_bar = iter

        # for industry
        cam_extractor = CAM(model, target_layer=self.target_layer) if self.is_industry else None    # 热力图
        ng_workbook = xlsxwriter.Workbook(os.path.join(output_dir, "NG.xlsx")) if self.is_industry else None   # NG表格
        ng_worksheet = self.setXLSX(ng_workbook) if self.is_industry else None     # ng Excel表格
        ok_workbook = xlsxwriter.Workbook(os.path.join(output_dir, "OK.xlsx")) if self.is_industry else None   # OK表格
        ok_worksheet = self.setXLSX(ok_workbook) if self.is_industry else None     # ng Excel表格
        iter_now = 0 if self.is_industry else None

        for imgs, targets, paths in progress_bar(self.dataloader):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                outputs = model(imgs)
                if self.is_industry:
                    # 将处理验证集中的每张图片
                    logger.info("{}/{}".format(iter_now, len(self.dataloader)))
                    iter_now += 1
                    self._industry(output_tensor=outputs,
                                   label=targets,
                                   output_dir=output_dir,
                                   img_p=paths[0],
                                   cam_extractor=cam_extractor)
                data_list.append((outputs, targets, paths))

        # 根据是否是分布式，将结果分别存储到output_s,target_s,path_s中
        if distributed:     # 如果是分布式，将每个rank的data_list合并到outpus_s, target_s, path_s中
            output_s = []
            target_s = []
            path_s = []
            data_list = gather(data_list, dst=0)
            for data_ in data_list:     # multi gpu
                for pred, target, path in data_:  # 每个gpu所具有的sample
                    output_s.append(pred)
                    target_s.append(target)
                    path_s.append(path)
        else:   # 如果不是分布式，即单卡情况，将data_list中的数据分别放到output_s,target_s,path_s中
            output_s = []
            target_s = []
            path_s = []
            for data_ in data_list:
                output_s.append(data_[0])
                target_s.append(data_[1])
                path_s.append(data_[2])

        # 根据是否是rank=0进程， 将top1, top2, confu_ma混淆矩阵返回
        if not is_main_process():
            top1, top2, confu_ma = 0, 0, None
        else:
            self.meter.update(
                outputs=torch.cat([output.to(device=device) for output in output_s]),
                targets=torch.cat([output.to(device=device) for output in target_s])
            )  # 更新，计算top1, top2
            self.meter.eval_confusionMatrix(
                preds=torch.cat([output.to(device=device) for output in output_s]),
                labels=torch.cat([output.to(device=device) for output in target_s])
            )  # 计算混淆矩阵

            top1 = self.meter.precision_top1.avg
            top2 = self.meter.precision_top2.avg
            logger.info("top1:{}, top2:{}".format(top1, top2))
            confu_ma = self.meter.confusion_matrix
            # 重置top1, top2, 混淆矩阵，避免下次验证时，累加以前结果
            self.meter.precision_top1.initialized = False
            self.meter.precision_top2.initialized = False
            self.meter.confusion_matrix = [[0 for j in range(self.num_class)] for i in range(self.num_class)]

            if self.is_industry:    # 是工业分支
                logger.info("figure confusion matrix")
                self.plot_confusion_matrix(confu_ma, self.class_names, title="Confusion Matrix",
                                           num_classes=self.num_class,
                                           dst_path=os.path.join(output_dir, "ConfusionMatrix.png"))
                logger.info("writer NG xlsx")
                self.writer_ng_XLSX(ng_worksheet, self.excel_ng, output_dir)
                ng_workbook.close()
                logger.info("writer OK xlsx")
                self.writer_ok_XLSX(ok_worksheet, self.excel_ok, output_dir)
                ok_workbook.close()

        synchronize()

        return top1, top2, confu_ma

    @logger.catch
    def _industry(self, output_tensor=None,
                  label=None,
                  output_dir=None,
                  img_p=None,
                  cam_extractor=None):
        """此函数主要是将原图、直方图拉伸、热力图拷贝到对应位置"""
        # 获得label
        label = str(label.cpu().numpy().item())
        # 获得pred
        prediction = output_tensor.squeeze(0).cpu().detach().numpy()
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        scores = F.softmax(output_tensor.squeeze(0), dim=0).cpu().numpy().max()  # top1 分数
        pred = prediction.item()  # top1下标

        self.pred_target_list.append((pred, label))  # 记录预测值和标签值，供计算混淆矩阵使用 (int,str)

        # 将NG，即label != pred(top1) 放到excel中；同时将OK也放到excel中，只不过没有图片
        excel_ng = {}
        excel_ok = {}
        save_cam = False
        if int(label) != pred:
            top_i = output_tensor.topk(2, dim=1, largest=True, sorted=True)[1].squeeze(0).cpu().numpy()
            top_v = F.softmax(output_tensor.squeeze(0), dim=0).cpu().numpy()

            excel_ng["img_p"] = img_p.replace("\\", "/").split("/")[-1]

            excel_ng["top1_id"] = top_i[0]
            excel_ng["top1_name"] = self.labels_[str(top_i[0])]
            excel_ng["top1_score"] = top_v[top_i[0]]

            excel_ng["top2_id"] = top_i[1]
            excel_ng["top2_name"] = self.labels_[str(top_i[1])]
            excel_ng["top2_score"] = top_v[top_i[1]]

            excel_ng["label"] = int(label)
            excel_ng["label_name"] = self.labels_[label]

            save_cam = True  # 是否保存原图、热力图的标志位
        elif int(label) == pred:
            top_i = output_tensor.topk(3, dim=1, largest=True, sorted=True)[1].squeeze(0).cpu().numpy()
            top_v = F.softmax(output_tensor.squeeze(0), dim=0).cpu().numpy()

            excel_ok["img_p"] = img_p.replace("\\", "/").split("/")[-1]

            excel_ok["top1_id"] = top_i[0]
            excel_ok["top1_name"] = self.labels_[str(top_i[0])]
            excel_ok["top1_score"] = top_v[top_i[0]]

            excel_ok["top2_id"] = top_i[1]
            excel_ok["top2_name"] = self.labels_[str(top_i[1])]
            excel_ok["top2_score"] = top_v[top_i[1]]

            excel_ok["label"] = int(label)
            excel_ok["label_name"] = self.labels_[label]

            self.excel_ok.append(excel_ok)

        # ---------- 图片拷贝
        pic_path = os.path.join(output_dir, "pictures")
        os.makedirs(pic_path, exist_ok=True)
        # 1. 拷贝图片
        img_path = img_p.split("/")[-2] + "_" + img_p.split("/")[-1]   # labelName_imageName
        output_name = "img-{}__pred-{}__target-{}__score-{}.{}".format(img_path,
                                                                       self.labels_[str(pred)],
                                                                       self.labels_[label],
                                                                       str(scores),
                                                                       img_p[-3:]
                                                                       )
        shutil.copy(os.path.join(img_p), os.path.join(pic_path, output_name))

        # 2.拷贝CAM图
        output_name_cam = output_name[:-4] + ".png"
        activation_map = cam_extractor(output_tensor.cpu().squeeze(0).argmax().item(), output_tensor.cpu())
        result = overlay_mask(Image.open(img_p).convert("RGB"), to_pil_image(activation_map[0], mode='F'), alpha=0.5)
        cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(os.path.join(pic_path, output_name_cam))

        if save_cam:
            excel_ng["img"] = os.path.join(pic_path, output_name)
            excel_ng["cam"] = os.path.join(pic_path, output_name_cam)
            self.excel_ng.append(excel_ng)

    @logger.catch
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                              num_classes=38, dst_path=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = np.asarray(cm)
        # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        length = 10 if num_classes < 10 else num_classes // 2
        fig = plt.figure(dpi=100, figsize=(length, length))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(dst_path)

    @logger.catch
    def setXLSX(self, workbook):
        worksheet = workbook.add_worksheet()
        self.col_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        self.col_list_name = ['ImageName',
                              'top1 ID', 'top1 Name', 'top1 Score',
                              'top2 ID', 'top2 Name', 'top2 Score',
                              'label ID', 'label Name',
                              'OriPicture', 'CAMPicture']
        # 写入标题
        for i in range(len(self.col_list)):  # 写入标题
            worksheet.write(self.col_list[i] + '1', self.col_list_name[i])

        return worksheet

    @logger.catch
    def writer_ng_XLSX(self, worksheet, excel_ng, output_dir):
        for m, one_r in enumerate(excel_ng):
            # 设置表格宽高
            worksheet.set_column(0, 9, 10)
            worksheet.set_column(9, 11, 256)
            worksheet.set_row(m + 1, 512)
            img_p = one_r['img_p']
            top1_id = one_r['top1_id']
            top1_name = one_r['top1_name']
            top1_score = one_r['top1_score']
            top2_id = one_r['top2_id']
            top2_name = one_r['top2_name']
            top2_score = one_r['top2_score']
            label = one_r['label']
            label_name = one_r['label_name']
            img = one_r['img']
            cam = one_r['cam']
            to_w = [img_p, top1_id, top1_name, top1_score, top2_id, top2_name, top2_score, label, label_name, img, cam]
            for n in range(len(self.col_list)):
                if n < 9:   # img_p, top1_id, top1_name, top1_score, top2_id, top2_name, top2_score, label, label_name
                    worksheet.write(str(self.col_list[n]) + str(m + 2), to_w[n])
                elif n == 9:   # img 字段
                    param = {
                        'x_offset': 0,
                        'y_offset': 0,
                        'x_scale': 1,
                        'y_scale': 1,
                        "width": 100,
                        "height": 80,
                        'url': None,
                        'tip': None,
                        'image_data': None,
                        'positioning': None,
                    }
                    try:
                        worksheet.insert_image(self.col_list[n] + str(m + 2), to_w[len(self.col_list) - 2], param)
                    except Exception as e:
                        print(e)
                elif n == 10:   # cam字段
                    param = {
                        'x_offset': 0,
                        'y_offset': 0,
                        'x_scale': 1,
                        'y_scale': 1,
                        "width": 100,
                        "height": 80,
                        'url': None,
                        'tip': None,
                        'image_data': None,
                        'positioning': None,
                    }
                    try:
                        worksheet.insert_image(self.col_list[n] + str(m + 2), to_w[len(self.col_list) - 1], param)
                    except Exception as e:
                        logger.error(e)

    @logger.catch
    def writer_ok_XLSX(self, worksheet, excel_ok, output_dir):
        for m, one_r in enumerate(excel_ok):
            # 设置表格宽高
            worksheet.set_column(0, 9, 10)
            worksheet.set_row(m + 1, 10)
            img_p = one_r['img_p']
            top1_id = one_r['top1_id']
            top1_name = one_r['top1_name']
            top1_score = one_r['top1_score']
            top2_id = one_r['top2_id']
            top2_name = one_r['top2_name']
            top2_score = one_r['top2_score']
            label = one_r['label']
            label_name = one_r['label_name']
            to_w = [img_p, top1_id, top1_name, top1_score, top2_id, top2_name, top2_score, label, label_name]
            for n in range(len(self.col_list)-2):
                worksheet.write(str(self.col_list[n]) + str(m + 2), to_w[n])
