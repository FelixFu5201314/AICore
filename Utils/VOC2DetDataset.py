# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import shutil
import os
import os.path as osp
import cv2
import xml.etree.ElementTree as ET
from loguru import logger


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = round(eval(bbox.find(pt).text))
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOC2DetDataset:
    def __init__(self, data_dir, image_sets, img_suffix="jpg", label_suffix="txt"):
        """
        Function: 将VOC数据格式 转成 AICore所需的DetDataset格式

        :param data_dir: 存放VOC原始数据的位置
            root@880d76488018:/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/VOCdevkit# tree -d
            .
            |-- VOC2007
            |   |-- Annotations
            |   |-- ImageSets
            |   |   |-- Layout
            |   |   |-- Main
            |   |   `-- Segmentation
            |   |-- JPEGImages
            |   |-- SegmentationClass
            |   `-- SegmentationObject
            `-- VOC2012
                |-- Annotations
                |-- ImageSets
                |   |-- Action
                |   |-- Layout
                |   |-- Main
                |   `-- Segmentation
                |-- JPEGImages
                |-- SegmentationClass
                `-- SegmentationObject
        :param image_sets: [('2007', 'trainval'), ('2012', 'trainval')]
        :param img_suffix: jpg
        :param label_suffix: txt
        """
        self.root = data_dir    # 数据集路径
        self.image_set = image_sets  # 数据选择
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        self.target_transform = VOCAnnotationTransform()
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()   # 存放所有数据的路径
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def generateDetDataset(self, dstPath="/root/voc0712", train_val_test="train.txt"):
        """
        Function: 生成images文件夹、labels文件夹、train.txt/val.txt/test.txt

        :param dstPath:
        :param train_val_test:
        :return:
        """
        imgDstPath = osp.join(dstPath, "images")
        os.makedirs(imgDstPath, exist_ok=True)
        labelDstPath = osp.join(dstPath, "labels")
        os.makedirs(labelDstPath, exist_ok=True)

        with open(osp.join(dstPath, train_val_test), 'a') as trainFile:
            logger.info("生成数据集中，共生成 {}".format(str(len(self.ids))))
            for i, (rootPath, img_id) in enumerate(self.ids):
                logger.info("{}...{}".format(str(i), self._imgpath % (rootPath, img_id)))
                # 1. copy image
                imgSrcPath = self._imgpath % (rootPath, img_id)
                shutil.copy(imgSrcPath, osp.join(imgDstPath, img_id+"."+self.img_suffix))
                img = cv2.imread(self._imgpath % (rootPath, img_id))
                height, width, channels = img.shape

                # 2. load a target
                target = ET.parse(self._annopath % (rootPath, img_id)).getroot()
                target = self.target_transform(target, width, height)

                # 3. check target
                if len(target) == 0:
                    logger.error("{} not bbox".format(imgSrcPath))
                    target = [[0, 0, 0, 0, 0]]

                # 4. 写入到labels中
                with open(osp.join(labelDstPath, img_id+"."+self.label_suffix), 'w') as lableFile:
                    for bbox_label in target:
                        lableFile.write(str(bbox_label[0]) + " ")
                        lableFile.write(str(bbox_label[1]) + " ")
                        lableFile.write(str(bbox_label[2]) + " ")
                        lableFile.write(str(bbox_label[3]) + " ")
                        lableFile.write(str(bbox_label[4]) + "\n")

                # 5. 添加train.txt
                trainFile.write("{}".format(img_id) + "\n")

    def generateDetDataset_labels(self, dstPath="/root/voc0712"):
        """
        Function: 生成labels.txt文件

        :param dstPath:
        :return:
        """
        with open(osp.join(dstPath, "labels.txt"), 'w') as labelFile:
            logger.info("生成labels.txt")
            for id_name in self.target_transform.class_to_ind.items():
                labelFile.write(str(id_name[0]) + " ")
                labelFile.write(str(id_name[1]) + "\n")


if __name__ == "__main__":
    trainval_dataset = VOC2DetDataset(
        data_dir='/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/VOCdevkit',
        image_sets=[('2007', 'trainval'), ('2012', 'trainval')]
    )
    test_dataset = VOC2DetDataset(
        data_dir='/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/VOCdevkit',
        image_sets=[('2007', 'test')]
    )
    trainval_dataset.generateDetDataset(dstPath="/root/voc0712", train_val_test="train.txt")    # 生成训练集
    test_dataset.generateDetDataset(dstPath="/root/voc0712", train_val_test="test.txt")  # 生成验证/测试集
    trainval_dataset.generateDetDataset_labels(dstPath="/root/voc0712")    # 生成labels.txt

    """
    结果：
        root@880d76488018:/root/voc0712# tree
        
        |-images  
            - *.jpg
        |-labels
            - *.txt
        |-labels.txt
        |-test.txt
        |-train.txt
    """