# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import os
import numpy as np
import random
import shutil
from loguru import logger

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

coco_class_labels = ('background',
                     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                     'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                     'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                     'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                     'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                     'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(self,
                 data_dir=None,
                 json_file='instances_train2017.json',
                 name='train2017',
                 img_suffix="jpg",
                 label_suffix="txt"
                 ):
        """
        Function: 将COCO数据集转成AICore DetDataset所需格式

        Args:
            data_dir (str): dataset root directory
                root@880d76488018:/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/COCO# tree -d
                    |-- annotations
                        |-- captions_train2017.json
                        |-- captions_val2017.json
                        |-- instances_train2017.json
                        |-- instances_val2017.json
                        |-- person_keypoints_train2017.json
                        `-- person_keypoints_val2017.json
                    |-- test2017
                        |-- *.jpg
                    |-- train2017
                        |-- *.jpg
                    `-- val2017
                        |-- *.jpg
            json_file (str): COCO json file name. instances_train2017.json or instances_val2017.json
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_suffix="jpg",
            label_suffix="txt"
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        self.name = name

        self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())

    def generateDetDataset(self, dstPath=None, train_val_test="train.txt"):
        """
        Function: 生成images文件夹、labels文件夹、train.txt/val.txt/test.txt

        :param dstPath:
        :param train_val_test:
        :return:
        """
        imgDstPath = os.path.join(dstPath, "images")
        os.makedirs(imgDstPath, exist_ok=True)
        labelDstPath = os.path.join(dstPath, "labels")
        os.makedirs(labelDstPath, exist_ok=True)

        Nobbox_cout = 0
        with open(os.path.join(dstPath, train_val_test), 'a') as trainFile:
            logger.info("生成数据集中，共生成 {}".format(str(len(self.ids))))
            for i, id_ in enumerate(self.ids):
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
                annotations = self.coco.loadAnns(anno_ids)

                # 1. copy image
                img_file = os.path.join(self.data_dir, self.name, '{:012}'.format(id_) + '.jpg')
                shutil.copy(img_file, os.path.join(imgDstPath, '{:012}'.format(id_) + "." + self.img_suffix))
                # logger.info("{}...{}".format(str(i), img_file))
                img = cv2.imread(img_file)
                height, width, channels = img.shape

                # 2. load a target
                target = []
                for anno in annotations:
                    if 'bbox' in anno and anno['area'] > 0:
                        xmin = np.max((0, anno['bbox'][0]))
                        ymin = np.max((0, anno['bbox'][1]))
                        xmax = np.min((width - 1, xmin + np.max((0, anno['bbox'][2] - 1))))
                        ymax = np.min((height - 1, ymin + np.max((0, anno['bbox'][3] - 1))))
                        if xmax > xmin and ymax > ymin:
                            label_ind = anno['category_id']
                            cls_id = self.class_ids.index(label_ind)
                            xmin /= width
                            ymin /= height
                            xmax /= width
                            ymax /= height

                            target.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
                    else:
                        logger.error("'bbox' in anno and anno['area'] > 0")

                # 3. check target
                if len(target) == 0:
                    logger.error("{}:{} No bbox, add target [[0, 0, 0, 0, 0]]".format(str(id_), img_file))
                    Nobbox_cout += 1
                    target = [[0, 0, 0, 0, 0]]

                # 4. 写入到labels中
                with open(os.path.join(labelDstPath, '{:012}'.format(id_) + "." + self.label_suffix), 'w') as lableFile:
                    for bbox_label in target:
                        lableFile.write(str(bbox_label[0]) + " ")
                        lableFile.write(str(bbox_label[1]) + " ")
                        lableFile.write(str(bbox_label[2]) + " ")
                        lableFile.write(str(bbox_label[3]) + " ")
                        lableFile.write(str(bbox_label[4]) + "\n")

                # 5. 添加train.txt
                trainFile.write("{}".format('{:012}'.format(id_)) + "\n")
        logger.info("Nobbox numerb is {}".format(str(Nobbox_cout)))

    def generateDetDataset_labels(self, dstPath=None):
        """
        Function: 生成labels.txt文件

        :param dstPath:
        :return:
        """
        with open(os.path.join(dstPath, "labels.txt"), 'w') as labelFile:
            logger.info("生成labels.txt")
            for id, name in enumerate([coco_class_labels[tmp] for tmp in coco_class_index]):
                labelFile.write(str(name) + ":")
                labelFile.write(str(id) + "\n")


if __name__ == "__main__":
    train_dataset = COCODataset(
        data_dir='/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/COCO',
        json_file="instances_train2017.json",
        name='train2017',

    )
    val_dataset = COCODataset(
        data_dir='/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/COCO',
        json_file="instances_val2017.json",
        name='val2017',
    )

    train_dataset.generateDetDataset(
        dstPath="/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/coco2017",
        train_val_test="train.txt")    # 生成训练集
    val_dataset.generateDetDataset(
        dstPath="/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/coco2017",
        train_val_test="val.txt"
    )
    train_dataset.generateDetDataset_labels(
        dstPath="/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/coco2017"
    )    # 生成labels.txt
