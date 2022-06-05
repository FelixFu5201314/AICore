# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import os
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from torch.utils.data import Dataset
import torch.nn.functional as F

from dao.register import Registers


def pad_to_square(img, pad_value=0):
    """
    Function：将img图像使用pad_value值填充，填充到（ max(w, h) ， max(w, h))大小
    :param img: ndarray
    :param pad_value:int
    :return:
    """
    h, w, c = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    # pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)  # （w, w, h, h)
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))  # ((h,h),(w,w),(c,c))
    # Add padding
    # img = F.pad(img, pad, "constant", value=pad_value)
    img = np.pad(img, pad, 'constant', constant_values=0)

    return img, [pad[1][0], pad[1][1], pad[0][0],pad[0][1]] # pad:((h,h),(w,w),(c,c))-->（w, w, h, h)


def resize(image, size):
    """
    Function：将image resize 到（size， size）大小
    :param image:
    :param size:
    :return:
    """
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


@Registers.datasets.register
class DetDataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 preproc=None,
                 preproc_pixel=None,
                 image_set="",
                 in_channels=1,
                 input_size=(416, 416),
                 image_suffix=".jpg",
                 mask_suffix=".txt",
                 ):
        """
        Function: 目标检测数据集

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
        input_size:tuple 输入图片的HW, 当需要cache image时需要，此处用不到
        preproc:albumentations.Compose 对图片进行预处理
        preproc_pixel:albumentations.Compose 对图片进行预处理, 针对COCO数据集中无bbox情况
        cache:bool 是否对图片进行内存缓存
        images_suffix:str 可接受的图片后缀
        mask_suffix:str 可接受的图片后缀
        """
        # set attr
        self.root = data_dir    # 数据集路径
        self.preproc = preproc  # albumentations预处理
        self.preproc_pixel = preproc_pixel  # albumentations预处理, 针对COCO数据集中无bbox情况
        self.image_set = image_set  # 训练集、验证集、测试集
        self.in_channels = in_channels  # 图片通道数
        self.img_size = input_size  # 图片宽高
        self.image_suffix = image_suffix    # 图片文件后缀名
        self.mask_suffix = mask_suffix  # 标注文件后缀名

        # 存储数据
        self.ids = []   # 存放图片路径 (image path, mask path)
        self.labels_id_name = dict()  # id:name形式
        self.labels_name_id = dict()  # name:id形式

        self._set_ids()  # 获取所有文件名，存放到self.ids中 [(image_path, label_path), ... ]

    def __getitem__(self, index):
        """
        Function: 通过index, 获取数据
        :param index:
        :return:
            transformed_image: image ndarray
            transformed_bboxes: 返回bboxes [n, 4], norm[x1,y1,x2,y2]
            transformed_class_labels:[n], class_id
            image_path:图片路径
        """

        # 1. 获得原始的图片，bboxes，labels和图片路径
        image, bboxes, class_labels, image_path = self.pull_item(index)  # image:ndarray(h,w,c), label:ndarray[(x1,y1,x2,y2),...], class_labels:ndarray[class_id,...], image_path:[string jpg,string label]

        # 2.将image保持比例，resize到max(height,width)大小
        h_factor, w_factor, _ = image.shape
        image, pad = pad_to_square(image, 0)  # pad:((h,h),(w,w),(c,c))
        padded_h, padded_w, _ = image.shape
        # Extract coordinates for unpadded + unscaled image，对未padding图片 解压 标签坐标
        x1 = w_factor * bboxes[:, 0]  # 左上角x1
        y1 = h_factor * bboxes[:, 1]  # 左上角y1
        x2 = w_factor * bboxes[:, 2]  # 右下角x2
        y2 = h_factor * bboxes[:, 3]  # 右下角y2
        # Adjust for added padding
        x1 += pad[0]  # 扩充左上角x1
        y1 += pad[2]  # 扩充左上角y1
        x2 += pad[1]  # 扩充左上角x2
        y2 += pad[3]  # 扩充左上角y2
        # (x1,y1,x2,y2)->norm(cx,cy,w,h)
        # bboxes[:, 0] = x1 / padded_w
        # bboxes[:, 1] = y1 / padded_h
        # bboxes[:, 2] = x2 / padded_w
        # bboxes[:, 3] = y2 / padded_h
        bboxes[:, 0] = ((x1+x2)/2) / padded_w    # bboxes[:, 0] = x1 / padded_w
        bboxes[:, 0] = ((x1 + x2) / 2) / padded_w  # bboxes[:, 0] = x1 / padded_w
        bboxes[:, 1] = ((y1 + y2) / 2) / padded_h   # bboxes[:, 1] = y1 / padded_h
        bboxes[:, 2] = (x2 - x1) / padded_w         # bboxes[:, 2] = x2 / padded_w
        bboxes[:, 3] = (y2 - y1) / padded_h         # bboxes[:, 3] = y2 / padded_h
        # resize image
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)

        # 3. 使用albumentations增强图片
        if len(bboxes) == 1 and np.sum(bboxes) == 0 and self.preproc_pixel is not None:
            transformed = self.preproc_pixel(image=image)
            transformed_image = transformed['image']
            transformed_bboxes = bboxes
            class_labels = [255]
        elif self.preproc is not None:
                class_labels_name = [self.labels_id_name[str(tmp)] for tmp in class_labels]  # 通过class_id获得class_name
                transformed = self.preproc(image=image, bboxes=bboxes, class_labels=class_labels_name)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']

        # 4. 对albumentations增强的images和bboxes进行处理
        transformed_image = transformed_image.transpose(2, 0, 1)  # c, h, w
        transformed_bboxes = np.asarray(transformed_bboxes)
        labels_bboxes = np.hstack((np.expand_dims(class_labels, axis=1), transformed_bboxes))  # 将bboxes和labels合并
        labels = np.zeros((len(labels_bboxes), 6))  # 添加一维，为了存放此张图片是那个batchsize的
        labels[:, 1:] = labels_bboxes
        return transformed_image, labels, image_path

    def pull_item(self, index):
        """
        Function: 通过index，返回img，bboxes，图片路径
        :param index:
        :return:
        """
        img, bbox, label = self._load_img(index)  # 图片（ndarray）; bbox（[(x,y,w,h)归一化后的,]); label[类别ID,...]
        if self.in_channels == 1:
            img = np.expand_dims(img.copy(), axis=2)
        elif self.in_channels == 3:
            if len(img.shape) == 2:  # COCO2017数据中存在灰度图像，000000559665.jpg，所以需要将这类图像转成BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img.copy()
        return img, np.array(bbox), np.array(label), self.ids[index]

    def _load_img(self, index):
        """
        Function：通过index,找到文件名，
             然后获得
                    图片（ndarray）
                    bbox（[norm(x1,y1,x2,y2), ...])
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
                bbox.append([eval(tmp) for tmp in line.strip().split()][:4])
                label.append([eval(tmp) for tmp in line.strip().split()][4])
        return image, bbox, label

    def _set_ids(self):
        """
        Function：获取所有文件的文件名
        """
        list_path = os.path.join(self.root, self.image_set)

        # 获得图片路径，设置self.ids
        with open(list_path, 'r', encoding='utf-8') as images_labels:
            for line in images_labels.readlines():
                image_path = os.path.join(self.root, "images", line.strip() + self.image_suffix).replace("\\",'/')
                label_path = os.path.join(self.root, "labels", line.strip() + self.mask_suffix).replace("\\",'/')
                self.ids.append((image_path, label_path))

        # 获得id:name，设置self.labels_id_name = dict() , self.labels_name_id
        with open(os.path.join(self.root, "labels.txt"), 'r', encoding='utf-8') as labelsFile:
            for line in labelsFile.readlines():
                line = line.strip()
                self.labels_id_name[line.split(":")[1]] = line.split(":")[0]
                self.labels_name_id[line.split(":")[0]] = line.split(":")[1]

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
    from dao.dataloaders.augments import get_transformerYOLO, get_transformer
    from dotmap import DotMap
    # VOC
    dataset_d = {
        "type": "DetDataset",
        "kwargs": {
            "data_dir": "/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/voc0712",
            "image_set": "train.txt",
            "in_channels": 3,
            "image_suffix": ".jpg",
            "mask_suffix": ".txt"
        },
        "transforms": {
            "kwargs": {
                # "Resize": {"height": 416, "width": 416, "p": 1},
                "Flip": {"p": 1},
                # "SmallestMaxSize":{"max_size":640, "p":1},
                # "PadIfNeeded":{"min_height":416, "min_width":416, "p":1, "border_mode":0, "value":0},
                "Normalize": {"mean": [0.45289162, 0.43158466, 0.3984241], "std": [0.2709828, 0.2679657, 0.28093508], "p": 1}

            }
        }
    }
    dataset_c = DotMap(dataset_d)
    transforms = get_transformerYOLO(dataset_c.transforms.kwargs)
    seg_d = DetDataset(preproc=transforms, **dataset_c.kwargs)
    for i in range(1000):
        transformed_image, transformed_class_labels, image_path = seg_d.__getitem__(i)
        transformed_class_labels = transformed_class_labels[:, 1:]
        # 获得图片
        image = denormalization(transformed_image, norm_mean=[0.45289162, 0.43158466, 0.3984241], norm_std=[0.2709828, 0.2679657, 0.28093508])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image.shape[0], image.shape[1]
        for bbox in transformed_class_labels:
            # dataset返回norm(x1,y1,x2,y2)
            # xmin = round(bbox[1] * width)
            # ymin = round(bbox[2] * height)
            # xmax = round(bbox[3] * width)
            # ymax = round(bbox[4] * height)

            # dataset返回norm(cx,cy,w,h)
            xmin = round((bbox[1] - bbox[3]/2)*width)
            ymin = round((bbox[2] - bbox[4]/2)*height)
            xmax = round((bbox[1] + bbox[3]/2)*width)
            ymax = round((bbox[2] + bbox[4]/2)*height)
            labelName = int(bbox[0])

            if xmax <= xmin or ymax <= ymin:
                logger.error("No bbox")
                continue

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, seg_d.labels_id_name[str(labelName)], (xmin, ymin), font, 1, (0, 0, 255), 1)
        cv2.imwrite("/ai/data/test_voc/{}".format(image_path[0].split('/')[-1]), image)

    # COCO
    dataset_d = {
        "type": "DetDataset",
        "kwargs": {
            "data_dir": "/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/coco2017",
            "image_set": "train.txt",
            "in_channels": 3,
            "image_suffix": ".jpg",
            "mask_suffix": ".txt"
        },
        "transforms": {
            "kwargs": {
                # "Resize": {"height": 416, "width": 416, "p": 1},
                # "Flip": {"p": 1},
                # "PadIfNeeded": {"min_height": 416, "min_width": 416, "p": 1, "border_mode": 0, "value": 0},
                "Normalize": {"mean": [0.47013634, 0.44689935, 0.4076691], "std": [0.27452907, 0.26994488, 0.28498003], "p": 1}

            }
        }
    }
    dataset_c = DotMap(dataset_d)
    transforms = get_transformerYOLO(dataset_c.transforms.kwargs)
    transforms_pixel = get_transformer(dataset_c.transforms.kwargs)
    seg_d = DetDataset(preproc=transforms, preproc_pixel=transforms_pixel, **dataset_c.kwargs)
    for i in range(1000):
        transformed_image, transformed_class_labels, image_path = seg_d.__getitem__(i)
        transformed_class_labels = transformed_class_labels[:, 1:]

        # 获得图片
        image = denormalization(transformed_image,
                                norm_mean=[0.47013634, 0.44689935, 0.4076691],
                                norm_std=[0.27452907, 0.26994488, 0.28498003])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image.shape[0], image.shape[1]
        for bbox in transformed_class_labels:
            # dataset返回norm(x1,y1,x2,y2)
            # xmin = round(bbox[1] * width)
            # ymin = round(bbox[2] * height)
            # xmax = round(bbox[3] * width)
            # ymax = round(bbox[4] * height)

            # dataset返回norm(cx,cy,w,h)
            xmin = round((bbox[1] - bbox[3]/2)*width)
            ymin = round((bbox[2] - bbox[4]/2)*height)
            xmax = round((bbox[1] + bbox[3]/2)*width)
            ymax = round((bbox[2] + bbox[4]/2)*height)
            labelName = int(bbox[0])
            if xmax <= xmin or ymax <= ymin:
                logger.error("{} ... No bbox".format(image_path[0]))
            else:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, seg_d.labels_id_name[str(labelName)], (xmin, ymin), font, 1, (0, 0, 255), 1)
        cv2.imwrite("/ai/data/test_coco/{}".format(image_path[0].split('/')[-1]), image)
