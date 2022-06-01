# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
# coding=utf-8
"""
计算图片数据集所有图片的均值和方差
"""
import numpy as np
import cv2
import os
from tqdm import tqdm


def calMeanStd(
        imgs_path="/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/voc0712/images",

):
    img_h, img_w = 32, 32  # 根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []

    imgs_path_list = os.listdir(imgs_path)
    for item in tqdm(imgs_path_list):
        img = cv2.imread(os.path.join(imgs_path, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))


if __name__ == "__main__":
    calMeanStd(imgs_path="/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/voc0712/images")
    calMeanStd(imgs_path="/ai/data/AIDatasets/ObjectDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/coco2017/images")
    """
    VOC2007_train & VOC2007_test & VOC2012_train
        normMean = [0.45289162, 0.43158466, 0.3984241]
        normStd = [0.2709828, 0.2679657, 0.28093508]
    
    COCO2017
        normMean = [0.47013634, 0.44689935, 0.4076691]
        normStd = [0.27452907, 0.26994488, 0.28498003]
    """