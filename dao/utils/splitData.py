# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import os
import random
from loguru import logger


def split_ImageClassification(srcPath, valper, testper):
    log = ""
    train_list, val_list, test_list, labels_dict = [], [], [], {}

    # 获得所有类别名称
    classes_folder = [folder for folder in os.listdir(srcPath) if os.path.isdir(os.path.join(srcPath, folder))]
    for i, cl in enumerate(classes_folder):
        labels_dict[cl] = str(i)

    # --------- 处理每一类
    # 获得后缀
    all_files = os.listdir(os.path.join(srcPath, classes_folder[0]))
    img_suffixs = set([temp.split(".")[-1] for temp in all_files])  # 所有后缀
    img_suffixs = list(img_suffixs & set(("png", "jpg", "bmp", "jpeg")))  # 图片后缀
    if len(img_suffixs) != 1:
        logger.error("{} no Image".format(srcPath))
        log += "{} no Image".format(srcPath)
        return
    suffix = img_suffixs[0]  # 本文件夹的图片后缀
    # 处理每一类
    for folder in classes_folder:
        # 此类下所有图片
        all_images = [img for img in os.listdir(os.path.join(srcPath, folder)) if img.endswith(suffix)]
        select_ids = list(range(0, len(all_images)))  # 选取下标
        random.shuffle(select_ids)
        train_num = int(round((1 - valper - testper) * len(all_images)))  # 训练集个数
        val_num = int(round(valper * len(all_images)))  # 验证集个数
        test_num = int(round(testper * len(all_images)))  # 测试集个数
        train_index = select_ids[0: train_num]  # 训练集下标
        val_index = select_ids[train_num: train_num + val_num]  # 验证集下标
        test_index = select_ids[train_num + val_num:]  # 测试集下标

        # 选取训练集
        for sel in train_index:
            train_list.append(os.path.join(folder, all_images[sel]) + ":" + labels_dict[folder])

        # 选取验证集
        for sel in val_index:
            val_list.append(os.path.join(folder, all_images[sel]) + ":" + labels_dict[folder])

        # 选取训练集
        for sel in test_index:
            test_list.append(os.path.join(folder, all_images[sel]) + ":" + labels_dict[folder])

    # 生成txt
    with open(os.path.join(srcPath, "labels.txt"), 'w') as f:
        for line in labels_dict:
            f.write(line + ":" + labels_dict[line] + "\n")
    logger.info("write labels.txt")
    log += "write labels.txt"

    with open(os.path.join(srcPath, "train.txt"), 'w') as f:
        for line in train_list:
            f.write(line + "\n")
    logger.info("write train.txt")
    log += "\nwrite train.txt"

    with open(os.path.join(srcPath, "val.txt"), 'w') as f:
        for line in val_list:
            f.write(line + "\n")
    logger.info("write val.txt")
    log += "\nwrite val.txt"

    with open(os.path.join(srcPath, "test.txt"), 'w') as f:
        for line in test_list:
            f.write(line + "\n")
    logger.info("write test.txt")
    log += "\nwrite test.txt"

    return log


if __name__ == "__main__":
    split_ImageClassification(r"D:/钢41夹3D", 0.1, 0.1)