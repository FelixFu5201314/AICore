# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import os
import numpy as np
from PIL import Image
import cv2
from loguru import logger

from torch.utils.data import Dataset

from dao.register import Registers


@Registers.datasets.register
class Det2Dataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 preproc=None,
                 image_set="",
                 in_channels=1,
                 input_size=(224, 224),
                 image_suffix=".jpg",
                 mask_suffix=".txt",
                 ):
        """
        目标检测数据集

        data_dir:str  数据集文件夹路径，文件夹要求如下
            |-dataset
                |- images   存放所有图片的文件夹
                    |-图片
                |- labels   存放所有标注文件的图片
                    |-txt文件
                |- train.txt    训练集相对路径
                |- val.txt      验证集相对路径
                |- test.txt     测试集相对路径
                |- labels.txt   标签

        image_set:str "train.txt or val.txt or test.txt"
        in_channels:int  输入图片的通道数，目前只支持1和3通道
        input_size:tuple 输入图片的HW
        preproc:albumentations.Compose 对图片进行预处理
        cache:bool 是否对图片进行内存缓存
        images_suffix:str 可接受的图片后缀
        mask_suffix:str 可接受的图片后缀
        """
        # set attr
        self.root = data_dir    # 数据集路径
        self.preproc = preproc  # albumentations预处理
        self.image_set = image_set  # 训练集、验证集、测试集
        self.in_channels = in_channels  # 图片通道数
        self.img_size = input_size  # 图片宽高
        self.image_suffix = image_suffix    # 图片文件后缀名
        self.mask_suffix = mask_suffix  # 标注文件后缀名

        # 存储数据
        self.ids = []   # 存放图片路径 (image path, mask path)
        self.labels_dict = dict()  # id:name形式

        self._set_ids()  # 获取所有文件名，存放到self.ids中 [(image_path, label_path), ... ]

    def __getitem__(self, index):
        image, bboxes, class_labels, image_path = self.pull_item(index)  # image:ndarray, label:[(class_id,x,y,h,w),...], image_path:[string jpg,string label]

        # 使用albumentations增强图片
        class_labels_name = [self.labels_dict[str(tmp)] for tmp in class_labels]    # 通过class_id获得class_name
        if self.preproc is not None:
            transformed = self.preproc(image=image, bboxes=bboxes, class_labels=class_labels_name)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
        transformed_image = transformed_image.transpose(2, 0, 1)  # c, h, w
        transformed_bboxes = np.asarray(transformed_bboxes)
        return transformed_image, transformed_bboxes, transformed_class_labels, image_path

    def pull_item(self, index):
        """
        通过index，返回img，bboxes，图片路径
        :param index:
        :return:
        """
        img, bbox, label = self._load_img(index)  # 图片（ndarray）; bbox（[(x,y,w,h)归一化后的,]); label[类别ID,...]
        if self.in_channels == 1:
            img = np.expand_dims(img.copy(), axis=2)
        elif self.in_channels == 3:
            img = img.copy()
        return img, bbox, label, self.ids[index]

    def _load_img(self, index):
        """
        功能：通过index,找到文件名，
             然后获得
                    图片（ndarray）
                    bbox（[(x,y,w,h)归一化后的(yolo格式）,])
                    label[类别ID,...]
        :param index:
        :return:
        """
        image_path, label_path = self.ids[index]

        # get image
        image = None
        if self.in_channels == 1:
            image = Image.open(image_path).convert('L')
        elif self.in_channels == 3:
            image = Image.open(image_path)
        image = np.array(image)

        # get label
        bbox = []
        label = []
        with open(label_path, 'r') as bbox_file:
            for line in bbox_file.readlines():
                bbox.append([eval(tmp) for tmp in line.strip().split()][1:])
                label.append([eval(tmp) for tmp in line.strip().split()][0])
        return image, bbox, label

    def _set_ids(self):
        """
        功能：获取所有文件的文件名
        """
        list_path = os.path.join(self.root, self.image_set)

        # 获得图片路径，设置self.ids
        with open(list_path, 'r', encoding='utf-8') as images_labels:
            for line in images_labels.readlines():
                image_path = os.path.join(self.root, "images", line.strip() + self.image_suffix).replace("\\",'/')
                label_path = os.path.join(self.root, "labels", line.strip() + self.mask_suffix).replace("\\",'/')
                self.ids.append((image_path, label_path))

        # 获得id:name，设置self.labels_dict
        with open(os.path.join(self.root, "labels.txt"), 'r', encoding='utf-8') as labels:
            for label in labels:
                self.labels_dict[label.split()[0]] = label.split()[1]

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    # Root: {}".format(self.root)
        return fmt_str


def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == "__main__":
    from dao.dataloaders.augments import get_transformerYOLO
    from dotmap import DotMap
    dataset_d = {
        "type": "Det2Dataset",
        "kwargs": {
            "data_dir": "/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/coco2014",
            "image_set": "train.txt",
            "in_channels": 3,
            "input_size": [416, 416],
            "image_suffix": ".jpg",
            "mask_suffix": ".txt"
        },
        "transforms": {
            "kwargs": {
                "Resize": {"height": 416, "width": 416, "p": 1},
                # "RandomCrop": {"width": 224, "height": 224, "p": 1},
                "Flip":{"p":1},
                "Normalize": {"mean": [0.471, 0.448, 0.408], "std": [0.234, 0.239, 0.242], "p": 1}

            }
        }
    }

    dataset_c = DotMap(dataset_d)
    transforms = get_transformerYOLO(dataset_c.transforms.kwargs)
    seg_d = Det2Dataset(preproc=transforms, **dataset_c.kwargs)
    transformed_image, transformed_bboxes, transformed_class_labels, image_path = seg_d.__getitem__(1)

    image = denormalization(transformed_image, norm_mean=[0.471, 0.448, 0.408], norm_std=[0.234, 0.239, 0.242])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[0], image.shape[1]
    for bbox, labelName in zip(transformed_bboxes, transformed_class_labels):
        # cxcywh2xyxy
        center_x = round(bbox[0] * width)
        center_y = round(bbox[1] * height)
        bbox_width = round(bbox[2] * width)
        bbox_height = round(bbox[3] * height)
        xmin = int(center_x - bbox_width / 2)
        ymin = int(center_y - bbox_height / 2)
        xmax = int(center_x + bbox_width / 2)
        ymax = int(center_y + bbox_height / 2)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, labelName, (xmin, ymin), font, 1, (0, 0, 255), 1)
    cv2.imwrite("/ai/data/t1.jpg", image)
