# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def calMeanRGB(filePath=None, imgH=512, imgW=512):
    pathDir = os.listdir(filePath)
    R_channel = 0
    G_channel = 0
    B_channel = 0
    num = len(pathDir) * imgH * imgW  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样

    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = np.asarray(Image.open(os.path.join(filepath, filename))) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = np.asarray(Image.open(os.path.join(filepath, filename))) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_var = np.sqrt(R_channel / num)
    G_var = np.sqrt(G_channel / num)
    B_var = np.sqrt(B_channel / num)
    print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))


def calMeanGray(filePath=None, imgH=512, imgW=512):
    pathDir = os.listdir(filePath)
    Gray_channel = 0
    num = len(pathDir) * imgH * imgW  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样

    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = np.asarray(Image.open(os.path.join(filepath, filename))) / 255.0

        Gray_channel = Gray_channel + np.sum(img[:, :])

    Gray_mean = Gray_channel / num

    Gray_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = np.asarray(Image.open(os.path.join(filepath, filename))) / 255.0

        Gray_channel = Gray_channel + np.sum((img[:, :] - Gray_mean) ** 2)

    Gray_var = np.sqrt(Gray_channel / num)
    print("mean is %f, var is %f" % (Gray_mean, Gray_var))


if __name__ == "__main__":
    filepath = r'/ai/data/AIDatasets/AnomalyDetection/4AR6N-L546S-DQSM9-424ZM-N4DZ2/light/train/good'  # 数据集目录
    # calMeanGray(filepath, 2448, 2048)
    calMeanRGB(filepath, 2448, 2048)
    # light mean:       std:
