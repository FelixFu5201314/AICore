
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/blob/main/datasets/mvtec.py

import os
import numpy as np
from PIL import Image
from loguru import logger

from torch.utils.data import Dataset

from dao.register import Registers


@Registers.datasets.register
class MVTecDataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 preproc=None,
                 image_set="",
                 in_channels=1,
                 cache=False,
                 image_suffix=".png",
                 mask_suffix=".png",
                 **kwargs
                 ):
        """
        异常检测数据集，（MVTecDataset类型）

        data_dir:str  数据集文件夹路径，文件夹要求是
            📂datasets 数据集名称
              ┣ 📂 ground_truth  test测试文件夹对应的mask
              ┃     ┣ 📂 defective_type_1    异常类别1 mask（0，255）
              ┃     ┗ 📂 defective_type_2    异常类别2 mask
              ┣ 📂 test  测试文件夹
              ┃     ┣ 📂 defective_type_1    异常类别1 图片
              ┃     ┣ 📂 defective_type_2    异常类别2 图片
              ┃     ┗ 📂 good
              ┗ 📂 train 训练文件夹
              ┃     ┗ 📂 good

        preproc:albumentations.Compose 对图片进行预处理
        image_set:str "train.txt or val.txt or test.txt"； train.txt是训练，其余是测试
        in_channels:int  输入图片的通道数，目前只支持1和3通道
        cache:bool 是否对图片进行内存缓存
        image_suffix:str 可接受的图片后缀
        mask_suffix:str 可接受的图片后缀
        """
        # set attr
        self.root = data_dir    # 数据集路径
        self.preproc = preproc  # 预处理 transform
        self.is_train = True if image_set == "train.txt" else False  # 是否是训练
        self.in_channels = in_channels  # 输入图片通道数
        self.image_suffix = image_suffix    # 图片后缀
        self.mask_suffix = mask_suffix      # mask后缀

        # 存储image-mask pair
        self.x, self.y, self.mask = self.load_dataset_folder()  # x存放图片的路径；y标志此图片是否是good，good为0，非good为1；mask存放mask图片路径，good为空；

        if cache:
            logger.warning("MVTecDataset not supported cache !")

    def __getitem__(self, index):
        image, mask, label, image_path = self.pull_item(index)  # image:ndarray, 图片；mask:ndarray,掩码；label:int，是否有mask； image_path:string，图片路径
        if self.preproc is not None:
            transformed = self.preproc(image=image, mask=mask)
            image, mask = transformed['image'], transformed["mask"]
        image = image.transpose(2, 0, 1)  # c, h, w
        mask = mask.astype(np.int64)
        return image, mask, label, image_path  # image:ndarray, 图片；mask:ndarray,掩码；label:int，是否有mask； image_path:string，图片路径

    def pull_item(self, index):
        img, mask = self._load_img(index)
        if self.in_channels == 1:
            img = np.expand_dims(img.copy(), axis=2)
        elif self.in_channels == 3:
            img = img.copy()
        return img, mask, self.y[index], self.x[index]

    def _load_img(self, index):
        image_path = self.x[index]
        has_mask = self.y[index]
        mask_path = self.mask[index]

        # get image
        image = None
        if self.in_channels == 1:
            image = Image.open(image_path).convert('L')
        elif self.in_channels == 3:
            image = Image.open(image_path)
        image = np.array(image)

        # get mask
        if has_mask == 0:
            mask = np.zeros(image.shape[:2])
        else:
            mask = Image.open(mask_path)
            mask = np.array(mask)

        return image, mask

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []     # x存放图片的路径，y标志此图片是否是good（0），mask存放mask图片路径

        # 获得dataset目录下的所有文件夹，即train、test、ground_truth
        img_dir = os.path.join(self.root, phase)    # 训练集或测试集文件夹
        gt_dir = os.path.join(self.root, 'ground_truth')    # 真实mask文件夹

        # 如果是train，则只有good
        # 如果是test，则有good、其他异常类别
        img_types = sorted(os.listdir(img_dir))  # good、其他异常类别
        for img_type in img_types:  # 处理每个异常类别（包括good），train和test情况。
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            # 遍历其中一个类别下的所有文件
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith(self.image_suffix)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask' + self.mask_suffix)
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        fmt_str = "Dataset:" + self.__class__.__name__
        fmt_str += "; Length:{}".format(self.__len__())
        fmt_str += "; Data_dir:{}".format(self.root)
        return fmt_str


if __name__ == "__main__":
    from dao.dataloaders.augments import get_transformer
    from dotmap import DotMap
    dataset_c = {
        "type": "MVTecDataset",
        "kwargs": {
            "data_dir": "/ai/data/AIDatasets/AnomalyDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/zipper",
            "image_set": "test.txt",
            "in_channels": 1,
            "cache": False,
            "image_suffix": ".png",
            "mask_suffix": ".png"
        },
        "transforms": {
            "kwargs": {
                "Resize": {"height": 224, "width": 224, "p": 1, "interpolation": 0},
                "Normalize": {"mean": 0, "std": 1, "p": 1}
            }
        }
    }
    dataset_c = DotMap(dataset_c)
    transformer = get_transformer(dataset_c.transforms.kwargs)
    dataset = MVTecDataset(preproc=transformer, **dataset_c.kwargs)

    for i in range(len(dataset)):
        img, mask, label, img_p = dataset.__getitem__(i)
        print("image path:{}-->mask unique:{}".format(img_p, np.unique(mask)))
