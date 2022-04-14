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
class SegDataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 preproc=None,
                 image_set="",
                 in_channels=1,
                 input_size=(224, 224),
                 cache=False,
                 image_suffix=".jpg",
                 mask_suffix=".png",
                 ):
        """
        分割数据集

        data_dir:str  数据集文件夹路径，文件夹要求是
            |-dataset
                |- images
                    |-图片
                |- masks
                    |-图片
                |- train.txt
                |- val.txt
                |- test.txt
                |- labels.txt

        image_set:str "train.txt or val.txt or test.txt"
        in_channels:int  输入图片的通道数，目前只支持1和3通道
        input_size:tuple 输入图片的HW
        preproc:albumentations.Compose 对图片进行预处理
        cache:bool 是否对图片进行内存缓存
        images_suffix:str 可接受的图片后缀
        mask_suffix:str 可接受的图片后缀
        """
        # set attr
        self.root = data_dir
        self.preproc = preproc
        self.image_set = image_set
        self.in_channels = in_channels
        self.img_size = input_size
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        # 存储数据
        self.ids = []   # 存放图片路径 (image path, mask path)
        self.labels_dict = dict()  # id:name形式

        self._set_ids()  # 获取所有文件名，存放到self.ids中 [(image_path, mask_path), ... ]
        self.imgs = None
        self.masks = None
        if cache:
            self._cache_images()
            self._cache_masks()

    def __getitem__(self, index):
        image, mask, image_path = self.pull_item(index)  # image:ndarray, label:ndarray, image_path:string
        if self.preproc is not None:
            transformed = self.preproc(image=image, mask=mask)
            image, mask = transformed['image'], transformed["mask"]
        image = image.transpose(2, 0, 1)  # c, h, w
        mask = mask.astype(np.int64)  # c, h, w
        return image, mask, image_path

    def pull_item(self, index):
        if self.imgs is not None and self.masks is not None:
            img = np.array(self.imgs[index])
            mask = np.array(self.masks[index])
        else:
            img, mask = self._load_img(index)
            if self.in_channels == 1:
                img = np.expand_dims(img.copy(), axis=2)
            elif self.in_channels == 3:
                img = img.copy()
        return img, mask, self.ids[index]

    def _load_img(self, index):
        """
        功能：通过文件名获得，图片
        :param index:
        :return:
        """
        image_path, mask_path = self.ids[index]

        # get image
        image = None
        if self.in_channels == 1:
            image = Image.open(image_path).convert('L')
        elif self.in_channels == 3:
            image = Image.open(image_path)
        image = np.array(image)

        # get mask
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
        else:
            mask = np.zeros(image.shape[:2])

        return image, mask

    def _load_resize_img(self, index):
        image, mask = self._load_img(index)  # ndarray, ndarray
        image = cv2.resize(image, self.img_size, interpolation=0)
        mask = cv2.resize(mask, self.img_size, interpolation=0)
        return image, mask

    def _cache_images(self):
        """
        预加载所有图片到RAM中
        """
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have Enough RAM and Available disk space.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.root, "img_resized_cache_{}.array".format(self.image_set[:-4]))

        if not os.path.exists(cache_file):
            logger.info("Caching images for the frist time. This might take sometime")
            # np.memmap为存储在磁盘上的二进制文件中的数组创建内存映射。
            # 内存映射文件用于访问磁盘上的大段文件，而无需将整个文件读入内存。
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, self.in_channels),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self._load_resize_img(x)[0],
                range(len(self.ids)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.ids))
            for k, out in pbar:
                if self.in_channels == 1:
                    self.imgs[k][: out.shape[0], : out.shape[1], :] = np.expand_dims(out.copy(), axis=2)
                elif self.in_channels == 3:
                    self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, self.in_channels),
            dtype=np.uint8,
            mode="r+",
        )

    def _cache_masks(self):
        """
        预加载所有图片到RAM中
        """
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached masks in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have Enough RAM and Available disk space.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.root, "mask_resized_cache_{}.array".format(self.image_set[:-4]))

        if not os.path.exists(cache_file):
            logger.info("Caching images for the frist time. This might take sometime")
            # np.memmap为存储在磁盘上的二进制文件中的数组创建内存映射。
            # 内存映射文件用于访问磁盘上的大段文件，而无需将整个文件读入内存。
            self.masks = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_masks = ThreadPool(NUM_THREADs).imap(
                lambda x: self._load_resize_img(x)[1],
                range(len(self.ids)),
            )
            pbar = tqdm(enumerate(loaded_masks), total=len(self.ids))
            for k, out in pbar:
                self.masks[k][: out.shape[0], : out.shape[1]] = out.copy()
            self.masks.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.masks = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w),
            dtype=np.uint8,
            mode="r+",
        )

    def _set_ids(self):
        """
        功能：获取所有文件的文件名
        """
        list_path = os.path.join(self.root, self.image_set)

        with open(list_path, 'r', encoding='utf-8') as images_labels:
            for image_label in images_labels:
                image_path = os.path.join(self.root, "images", image_label.strip() + self.image_suffix)
                mask_path = os.path.join(self.root, "masks", image_label.strip() + self.mask_suffix)
                self.ids.append((image_path, mask_path))

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


if __name__ == "__main__":
    from dao.dataloaders.augments import get_transformer
    from dotmap import DotMap

    dataset_c = {
        "type": "SegDataset",
        "kwargs": {
            "data_dir": "/ai/data/AIDatasets/SemanticSegmentation/4AR6N-L546S-DQSM9-424ZM-N4DZ2/VOC2012_Seg_Aug",
            "image_set": "train.txt",
            "in_channels": 3,
            "input_size": [380, 380],
            "cache": True,
            "image_suffix": ".jpg",
            "mask_suffix": ".png"
        },
        "transforms": {
            "kwargs": {
                "Resize": {"height": 224, "width": 224, "p": 1},
                "Normalize": {"mean": [0.398993, 0.431193, 0.452234], "std": [0.285205, 0.273126, 0.276610], "p": 1}

            }
        }
    }
    dataset_c = DotMap(dataset_c)
    transforms = get_transformer(dataset_c.transforms.kwargs)
    seg_d = SegDataset(preproc=transforms, **dataset_c.kwargs)
    a, b, c = seg_d.__getitem__(1)
    # print(a, np.unique(b), c)
    print(c)
    print(np.unique(b))

    print(np.unique(np.array(Image.open("/ai/data/AIDatasets/SemanticSegmentation/4AR6N-L546S-DQSM9-424ZM-N4DZ2/VOC2012_Seg_Aug/masks/2007_000039.png"))))
