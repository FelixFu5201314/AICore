# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import cv2
import random
import numpy as np
from PIL import Image
from loguru import logger

import torch
from torch.utils.data import Dataset

from dao.register import Registers


@Registers.datasets.register
class ClsDataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 image_set="",
                 in_channels=1,
                 input_size=(224, 224),
                 preproc=None,
                 cache=False,
                 separator=":",
                 images_suffix=None):
        """
        分类数据集

        data_dir:str  数据集文件夹路径，文件夹要求是
            |-dataset
                |- 类别1
                    |-图片
                |- 类别2
                |- ......
                |- train.txt
                |- val.txt
                |- test.txt
                |- labels.txt

        image_set:str "train.txt", "val.txt" or "test.txt"
        in_channels:int  输入图片的通道数，目前只支持1和3通道
        input_size:tuple 输入图片的HW
        preproc:albumentations.Compose 对图片进行预处理
        cache:bool 是否对图片进行内存缓存
        separator:str labels.txt, train.txt, val.txt, test.txt 的分割符（name与id）
        images_suffix:list[str] 可接受的图片后缀
        """
        # 属性赋值
        self.root = data_dir
        self.image_set = image_set
        self.in_channels = in_channels
        self.img_size = input_size
        self.preproc = preproc
        if images_suffix is None:
            self.images_suffix = [".bmp"]

        # 设置ids，获得所有文件列表
        self.ids = []
        self.labels_dict = dict()   # name:id形式

        # 获取ids和label dict
        self._set_ids(separator=separator)  # 获取所有文件的路径和标签，存放到files中. (img_path, label_id)
        self._get_label_dict(data_dir, separator=separator)  # 设置labels， name:id

        # cache 过程
        self.imgs = None
        if cache:
            self._cache_images()

    def __getitem__(self, index):
        image, label, image_path = self.pull_item(index)  # image:mem, label:tensor(long), image_path:string
        image = np.asarray(image)
        if self.preproc is not None:
            image = self.preproc(image=image)['image']
        image = image.transpose(2, 0, 1)  # c, h, w
        return image, label, image_path

    @logger.catch
    def pull_item(self, index):
        if self.imgs is not None:
            img = self.imgs[index]
        else:
            img = self._load_resize_img(index)
            if self.in_channels == 1:
                img = np.expand_dims(img.copy(), axis=2)
            elif self.in_channels == 3:
                img = img.copy()
        label = torch.from_numpy(np.array(self.ids[index][1], dtype=np.int32)).long()   # torch.array形式
        # label = np.array(self.labels[index], dtype=np.int32)  # numpy.array形式
        return img, label, self.ids[index][0]

    @logger.catch
    def _load_resize_img(self, index):
        img = self._load_img(index)
        return np.asarray(cv2.resize(img, self.img_size))

    @logger.catch
    def _load_img(self, index):
        """
        功能：通过文件名获得，图片
        :param index:
        :return:
        """
        image_path = self.ids[index][0]
        img = None
        if self.in_channels == 1:
            img = Image.open(image_path).convert('L')
        elif self.in_channels == 3:
            img = Image.open(image_path)
        return np.asarray(img)

    @logger.catch
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
                lambda x: self._load_resize_img(x),
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

    @logger.catch
    def _set_ids(self, separator=" "):
        """
        功能：获取所有文件的文件名和标签, 放到self.ids中（img_path, label_id)
        """
        list_path = os.path.join(self.root, self.image_set)

        with open(list_path, 'r', encoding='utf-8') as images_labels:
            for image_label in images_labels:
                img_name = os.path.join(self.root, image_label.strip().split(separator)[0])
                label_id = image_label.strip().split(separator)[1]
                self.ids.append((img_name, label_id))

    @logger.catch
    def _get_label_dict(self, data_dir, separator=":"):
        """
        获得label dict数组， name:id形式
        """
        label_txt = os.path.join(data_dir, "labels.txt")
        with open(label_txt, "r", encoding='utf-8') as labels_file:
            for name_id in labels_file.readlines():
                self.labels_dict[name_id.strip().split(separator)[0]] = name_id.strip().split(separator)[1]

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
    from PIL import Image
    import cv2


    def denormalization(x, norm_mean, norm_std):
        mean = np.array(norm_mean)
        std = np.array(norm_std)
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
        return x

    dataset_c = {
        "type": "ClsDataset",
        "kwargs": {
            "data_dir": "/ai/data/AIDatasets/ImageClassification/4AR6N-L546S-DQSM9-424ZM-N4DZ2/screen",
            "image_set": "train.txt",
            "in_channels": 1,
            "input_size": [224, 224],
            "cache": False,
            "images_suffix": [".bmp"]
        },
        "transforms": {
            "kwargs": {
                "Normalize": {"mean": 0, "std": 1, "p": 1}
            }
        }
    }
    dataset_c = DotMap(dataset_c)
    transforms = get_transformer(dataset_c.transforms.kwargs)
    seg_d = ClsDataset(preproc=transforms, **dataset_c.kwargs)
    a, b, c = seg_d.__getitem__(200)
    cv2.imwrite("/ai/data/t1.jpg", denormalization(a, [0], [1]))
    print(c)
    print(b)

