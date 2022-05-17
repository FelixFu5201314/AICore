# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:https://github.com/AICoreRef/ind_knn_ad
from typing import Tuple
from tqdm import tqdm
from loguru import logger
import os
import pickle

import torch
from torch import tensor
from torch.utils.data import DataLoader
import timm

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
from loguru import logger
from random import sample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
from skimage import morphology
from skimage.segmentation import mark_boundaries

from .SPADE_PaDiM_PatchCore_Utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params
from dao.register import Registers


class KNNExtractor(torch.nn.Module):
    def __init__(
            self,
            backbone_name: str = "resnet50",
            out_indices: Tuple = None,
            pool_last: bool = False,
            device=None
    ):
        super().__init__()

        self.feature_extractor = timm.create_model(
            backbone_name,
            out_indices=out_indices,
            features_only=True,
            pretrained=True,
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone_name  # for results metadata
        self.out_indices = out_indices

        self.device = device
        self.feature_extractor = self.feature_extractor.to(self.device)

    def __call__(self, x: tensor):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x.to(self.device))
        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        if self.pool:
            # spit into fmaps and z
            return feature_maps[:-1], self.pool(feature_maps[-1])
        else:
            return feature_maps


@Registers.anomaly_models.register
class PaDiM2(torch.nn.Module):
    def __init__(self,
                 backbone, device=None, pool_last=False,
                 d_reduced: int = 100,
                 image_size=224, feature_size=56, beta=1):
        super(PaDiM2, self).__init__()
        # 定义网络结构
        self.feature_extractor = timm.create_model(
            backbone.type,
            out_indices=(1, 2, 3),
            features_only=True,
            pretrained=True
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_extractor.to(device)

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.device = device
        self.resize = torch.nn.AdaptiveAvgPool2d(feature_size)

        # 定义网络需要的其他信息
        self.image_size = image_size
        self.d_reduced = d_reduced  # your RAM will thank you
        self.epsilon = 0.04  # cov regularization
        self.patch_lib = []
        self.beta = beta

    def fit(self, train_dataloader, output_dir=None):
        # extract train set features 提取特征
        train_feature_filepath = os.path.join(output_dir, 'features.pkl')  # 特征存放路径
        if not os.path.exists(train_feature_filepath):  # 如果特征不存在
            # 提取特征
            logger.info("1.1 extract train set features")
            for i, (image, mask, label, image_path) in enumerate(train_dataloader):
                logger.info("extract feature iter {}/{}".format(i, len(train_dataloader)))
                with torch.no_grad():
                    feature_maps = self.feature_extractor(image.to(self.device))
                resized_maps = [self.resize(fmap) for fmap in feature_maps]
                self.patch_lib.append(torch.cat(resized_maps, 1))  # self.patch_lib = [ torch.Size([32, 1792, 56, 56]), ...]
            self.patch_lib = torch.cat(self.patch_lib, 0)   # 合并特征 torch.Size([240, 1792, 56, 56])

            # 随机选取特征
            logger.info("1.2 select randomly features")
            if self.patch_lib.shape[1] > self.d_reduced:
                logger.info(f"PaDiM: (randomly) reducing {self.patch_lib.shape[1]} dimensions to {self.d_reduced}.")
                self.r_indices = torch.randperm(self.patch_lib.shape[1])[:self.d_reduced]   # 550
                self.patch_lib_reduced = self.patch_lib[:, self.r_indices, ...]     # torch.Size([240, 550, 56, 56])
            else:
                logger.info(f"PaDiM: d_reduced is higher than the actual number of dimensions, copying self.patch_lib ...")
                self.patch_lib_reduced = self.patch_lib

            # 计算mean
            logger.info("1.3 calculate mean")
            self.means = torch.mean(self.patch_lib, dim=0, keepdim=True)    # torch.Size([1, 1792, 56, 56])
            self.means_reduced = self.means[:, self.r_indices, ...]     # torch.Size([1, 550, 56, 56])
            x_ = self.patch_lib_reduced - self.means_reduced    # torch.Size([240, 550, 56, 56])

            # 计算协方差
            logger.info("1.4 calculate cov")
            self.E = torch.einsum(
                'abkl,bckl->ackl',
                x_.permute([1, 0, 2, 3]),  # transpose first two dims
                x_,
            ) * 1 / (self.patch_lib.shape[0] - 1)  # torch.Size([550, 550, 56, 56])

            self.E += self.epsilon * torch.eye(self.d_reduced).unsqueeze(-1).unsqueeze(-1).to(self.device)  # torch.Size([550, 550, 1, 1])
            self.E_inv = torch.linalg.inv(self.E.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])   # torch.Size([550, 550, 56, 56])

            # 存储结果
            logger.info("1.5 save learned distribution")
            train_outputs = [self.means_reduced.cpu(), self.E_inv.cpu(), self.r_indices.cpu()]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            logger.info('1.1 load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        self.train_output = train_outputs

    def evaluate(self, test_dataloader, output_dir=None):
        """Calls predict step for each test sample."""
        gt_list = []
        gt_mask_list = []
        test_imgs = []
        test_imgs_path = []

        logger.info("2.1 extract test set features, and cal mean&cov&dist")
        maps = []   # 存放
        for i, (image, y, mask, image_path) in enumerate(test_dataloader):
            test_imgs.extend(image.cpu().detach().numpy())  # 所有图片
            gt_list.extend(y.cpu().detach().numpy())    # 是否是good
            gt_mask_list.extend(mask.cpu().detach().numpy())    # 所有mask
            test_imgs_path.extend(image_path)   # 图片路径

            logger.info("extract feature iter {}/{}".format(i, len(test_dataloader)))
            with torch.no_grad():
                feature_maps = self.feature_extractor(image.to(self.device))
            resized_maps = [self.resize(fmap) for fmap in feature_maps]
            fmap = torch.cat(resized_maps, 1)  # torch.Size([32, 1792, 56, 56])
            fmap = fmap.cpu()

            # reduce
            x_ = fmap[:, self.train_output[2], ...] - self.train_output[0]  # torch.Size([32, 550, 56, 56])

            # left = torch.einsum('abkl,bckl->ackl', x_, self.train_output[1])  # torch.Size([32, 550, 56, 56])
            left = torch.sum(x_.unsqueeze(1) * self.train_output[1].unsqueeze(0), dim=2)
            # s_map = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))  # torch.Size([32, 56, 56])
            s_map = torch.sqrt(torch.sum(x_ * left, dim=1))
            # score_map = torch.nn.functional.interpolate(
            #     s_map.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear'
            # )  # torch.Size([1, 32, 224, 224])
            score_map = torch.nn.Upsample(scale_factor=4, mode='bilinear')(s_map.unsqueeze(0)).squeeze(0)
            maps.append(score_map)

        # score_map = torch.cat(maps, dim=1).squeeze().numpy()
        score_map = torch.cat(maps, dim=0).squeeze().numpy()
        logger.info("2.6 apply gaussian smoothing on the score map")
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        score_map = torch.tensor(score_map)
        # Normalization
        logger.info("2.6 Normalization")
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)  # (B, 224, 224) scores是均值化后的结果

        # 以下是显示结果内容
        logger.info("2.7 result .......")
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))  # 绘制ROC曲线
        fig_img_rocauc = ax[0]
        fig_pixel_rocauc = ax[1]

        # calculate image-level ROC AUC score
        logger.info("calculate image-level ROC AUC score")
        # img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)  # shape B
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)[0]  # shape B
        gt_list = np.asarray(gt_list)  # shape B
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        logger.info('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='img_ROCAUC: %.3f' % (img_roc_auc))

        # calculate per-pixel level ROCAUC
        logger.info("calculate per-pixel level ROCAUC")
        gt_mask = np.where(np.asarray(gt_mask_list) != 0, 1, 0)  # (49, 1, 224, 224)
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        logger.info('pixel ROCAUC: %.3f' % (per_pixel_rocauc))
        fig_pixel_rocauc.plot(fpr, tpr, label='ROCAUC: %.3f' % (per_pixel_rocauc))

        # 绘制ROC曲线，image-level&pixel-level
        save_dir = os.path.join(output_dir, "pictures")
        os.makedirs(save_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=100)

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = (1 + self.beta ** 2) * precision * recall
        b = self.beta ** 2 * precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # 绘制每张test图片预测信息
        mean = test_dataloader.dataset.mean
        std = test_dataloader.dataset.std
        self.plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, test_imgs_path, mean, std)

        threshold_txt = os.path.join(output_dir, 'threshold.txt')
        with open(threshold_txt, 'w') as f:
            f.write(str(threshold) + "\n")
            f.write(str(max_score.numpy().item()) + "\n")
            f.write(str(min_score.numpy().item()) + "\n")

    def plot_fig(self, test_img, scores, gts, threshold, save_dir, test_imgs_path, mean, std):
        """
        将test_img,scores,gts根据threshold绘制成图像，并保存到save_dir中
        :param test_img: test_imgs:[(3, 224, 224), ..., batchsize]
        :param scores:  scores: (batchsize, 224, 224)
        :param gts: gt_mask_list: [(1, 224, 224), ..., batchsize]
        :param threshold: float
        :param save_dir: str
        :param class_name: [img_path, ..., batchsize]
        :return:
        """
        num = len(scores)
        logger.info("number:{}".format(num))
        vmax = scores.max() * 255.
        vmin = scores.min() * 255.
        for i in range(num):
            img = test_img[i]
            img = self.denormalization(img, mean=mean, std=std)
            gt = gts[i].transpose(1, 2, 0).squeeze()  # .transpose(1, 2, 0)
            heat_map = scores[i] * 255
            mask = scores[i]
            mask[mask > threshold] = 1
            mask[mask <= threshold] = 0
            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
            fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
            fig_img.subplots_adjust(right=0.9)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image')
            ax_img[1].imshow(gt, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
            ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[2].imshow(img, cmap='gray', interpolation='none')
            ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax_img[2].title.set_text('Predicted heat map')
            ax_img[3].imshow(mask, cmap='gray')
            ax_img[3].title.set_text('Predicted mask')
            ax_img[4].imshow(vis_img)
            ax_img[4].title.set_text('Segmentation result')
            left = 0.92
            bottom = 0.15
            width = 0.015
            height = 1 - 2 * bottom
            rect = [left, bottom, width, height]
            cbar_ax = fig_img.add_axes(rect)
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            font = {
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 8,
            }
            cb.set_label('Anomaly Score', fontdict=font)

            img_name = test_imgs_path[i].split("/")[-1][:-4] + ".png"
            ngtype = test_imgs_path[i].split("/")[-2]
            fig_img.savefig(os.path.join(save_dir, "{}_".format(ngtype) + img_name), dpi=100)
            plt.close()

    def denormalization(self, x, mean=[0.335782, 0.335782, 0.335782], std=[0.256730, 0.256730, 0.256730]):
        mean = np.array(mean)
        std = np.array(std)
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

        return x


@Registers.anomaly_models.register
class PaDiM2_demo(torch.nn.Module):
    def __init__(self,
                 backbone, device=None, pool_last=False,
                 image_size=224, feature_size=56,
                 select_index=None, features_mean=None, features_cov=None,
                 threshold=None, max_score=None, min_score=None,
                 output_dir=None, **kwargs):
        super(PaDiM2_demo, self).__init__()
        # 定义网络结构
        self.feature_extractor = timm.create_model(
            backbone.type,
            out_indices=(1, 2, 3),
            features_only=True,
            pretrained=True
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_extractor.to(device)

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.device = device
        self.resize = torch.nn.AdaptiveAvgPool2d(feature_size)

        # 定义其他权重信息
        self.image_size = image_size
        self.patch_lib = []
        # features.pkl
        self.mean = features_mean
        self.cov = features_cov
        self.select_index = select_index
        # threshold.txt
        self.threshold = threshold
        self.max_score = max_score
        self.min_score = min_score

        self.output_dir = output_dir

    def forward(self, x):
        fmaps = []
        with torch.no_grad():
            for img in x:
                feature_maps = self.feature_extractor(img[0].unsqueeze(0).to(self.device))
                resized_maps = [self.resize(fmap) for fmap in feature_maps]
                fmap = torch.cat(resized_maps, 1)  # torch.Size([32, 1792, 56, 56])
                fmaps.append(fmap)
        fmaps = torch.cat(fmaps, dim=0)  # torch.Size([36, 1792, 56, 56])

        # reduce
        x_ = fmaps[:, self.select_index, ...] - self.mean.to(self.device)  # torch.Size([32, 550, 56, 56])
        # left = torch.einsum('abkl,bckl->ackl', x_, self.cov.to(self.device))  # torch.Size([32, 550, 56, 56])
        left = torch.sum(x_.unsqueeze(1) * self.cov.to(self.device).unsqueeze(0), dim=2)
        # s_map = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))  # torch.Size([32, 56, 56])
        s_map = torch.sqrt(torch.sum(x_ * left, dim=1))
        # score_map = torch.nn.functional.interpolate(
        #     s_map.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear'
        # )  # torch.Size([1, 32, 224, 224])
        score_map = torch.nn.Upsample(scale_factor=4, mode='bilinear')(s_map.unsqueeze(0)).squeeze(0)

        # Normalization
        # logger.info("2.6 Normalization")
        scores = ((score_map - self.min_score) / (
                self.max_score - self.min_score)).squeeze()  # (B, 224, 224) scores是均值化后的结果
        return scores


@Registers.anomaly_models.register
class PaDiM2_export(torch.nn.Module):
    def __init__(self,
                 backbone, device=None, pool_last=False,
                 image_size=224, feature_size=56,
                 select_index=None, features_mean=None, features_cov=None,
                 threshold=None, max_score=None, min_score=None,
                 output_dir=None, **kwargs):
        super(PaDiM2_export, self).__init__()
        # 定义网络结构
        self.feature_extractor = timm.create_model(
            backbone.type,
            out_indices=(1, 2, 3),
            features_only=True,
            pretrained=True,
            exportable=True
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.device = device
        self.resize = torch.nn.AdaptiveAvgPool2d(feature_size)

        # 定义其他权重信息
        self.image_size = image_size
        self.patch_lib = []
        # features.pkl
        self.mean = features_mean
        self.cov = features_cov
        self.select_index = select_index
        # threshold.txt
        self.threshold = threshold
        self.max_score = max_score
        self.min_score = min_score

        self.output_dir = output_dir

    def forward(self, x):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x.to(self.device))
            resized_maps1 = feature_maps[0]
            resized_maps2 = torch.nn.Upsample(scale_factor=2, mode='nearest')(feature_maps[1])
            resized_maps3 = torch.nn.Upsample(scale_factor=4, mode='nearest')(feature_maps[2])
        fmaps = torch.cat((resized_maps1, resized_maps2, resized_maps3), 1)  # torch.Size([32, 1792, 56, 56])

        # reduce
        x_ = fmaps[:, self.select_index, ...] - self.mean.to(self.device)  # torch.Size([32, 550, 56, 56])
        # left = torch.einsum('abkl,bckl->ackl', x_, self.cov.to(self.device))  # torch.Size([32, 550, 56, 56])
        left = torch.sum(x_.unsqueeze(1) * self.cov.to(self.device).unsqueeze(0), dim=2)
        # s_map = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))  # torch.Size([32, 56, 56])
        s_map = torch.sqrt(torch.sum(x_ * left, dim=1))
        # score_map = torch.nn.functional.interpolate(
        #     s_map.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear'
        # )  # torch.Size([1, 32, 224, 224])
        score_map = torch.nn.Upsample(scale_factor=4, mode='bilinear')(s_map.unsqueeze(0)).squeeze(0)

        # Normalization
        # logger.info("2.6 Normalization")
        scores = ((score_map - self.min_score) / (
                self.max_score - self.min_score)).squeeze()  # (B, 224, 224) scores是均值化后的结果
        return scores


