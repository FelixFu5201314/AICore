# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/blob/main/main.py

import os
import pickle
import random
from random import sample
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
from loguru import logger

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
from skimage import morphology
from skimage.segmentation import mark_boundaries

import torch
import timm
import torch.nn.functional as F

from dao.register import Registers


@Registers.anomaly_models.register
class PaDiM:
    def __init__(self,
                 backbone,
                 device=None,
                 d_reduced: int = 100,
                 image_size=224,
                 beta=1,
                 ):
        self.feature_extractor = timm.create_model(
            backbone.type,
            **backbone.kwargs
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_extractor = self.feature_extractor.to(device)

        self.image_size = image_size
        self.d_reduced = d_reduced  # your RAM will thank you
        self.beta = beta
        self.device = device

        # random.seed(1024)
        # torch.manual_seed(1024)
        # if self.device is not None:
        #     torch.cuda.manual_seed_all(1024)

    def fit(self, train_dl, output_dir=None):
        # extract train set features
        train_feature_filepath = os.path.join(output_dir, 'features.pkl')
        if not os.path.exists(train_feature_filepath):
            feature_maps = []
            for image, mask, label, image_path in tqdm(train_dl, desc="Train: "):
                # model prediction
                feature_maps.append([fmap.to("cpu") for fmap in self.feature_extractor(image.to(self.device))])

            # 将feature_maps 转为 embedding_vectors: torch.Size([200, 1792, 56, 56])
            embedding_vectors = []
            for feature_map in feature_maps:
                embedding_vector = feature_map[0]
                for layer in feature_map[1:]:
                    embedding_vector = embedding_concat(embedding_vector, layer)
                embedding_vectors.append(embedding_vector)
            embedding_vectors = torch.cat(embedding_vectors, dim=0)

            # randomly select d dimension
            idx = torch.tensor(sample(range(0, embedding_vectors.shape[1]), self.d_reduced))
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)

            for i in tqdm(range(H * W), desc="Train calculate cov:"):
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov, idx]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            logger.info('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        self.train_output = train_outputs

    def evaluate(self, test_dl, output_dir=None):
        gt_list = []
        gt_mask_list = []
        test_imgs = []
        test_imgs_path = []
        self.test_outputs = []

        feature_maps = []
        # extract test set features
        for image, mask, label, image_path in tqdm(test_dl, desc="Evaluate: "):
            test_imgs.extend(image.cpu().detach().numpy())
            gt_list.extend(label.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            test_imgs_path.extend(image_path)

            feature_maps.append([fmap.to("cpu") for fmap in self.feature_extractor(image.to(self.device))])

        # 将feature_maps 转为 embedding_vectors: torch.Size([200, 1792, 56, 56])
        embedding_vectors = []
        for feature_map in feature_maps:
            embedding_vector = feature_map[0]
            for layer in feature_map[1:]:
                embedding_vector = embedding_concat(embedding_vector, layer)
            embedding_vectors.append(embedding_vector)
        embedding_vectors = torch.cat(embedding_vectors, dim=0)

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.train_output[2])

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors =embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in tqdm(range(H * W), desc="Evaluate calculate cov::"):
            mean = self.train_output[0][:, i]
            conv_inv = np.linalg.inv(self.train_output[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list) # torch.Size([49, 56, 56])
        score_map = F.interpolate(dist_list.unsqueeze(1), size=self.image_size, mode='bilinear',
                                  align_corners=False).squeeze().numpy()    # (49, 224, 224)

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)  # (49, 224, 224)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_img_rocauc = ax[0]
        fig_pixel_rocauc = ax[1]

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)    # shape 49
        gt_list = np.asarray(gt_list)   # shape 49
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        logger.info('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='img_ROCAUC: %.3f' % (img_roc_auc))

        # calculate per-pixel level ROCAUC
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
        # test_imgs:[(3, 224, 224), ..., batchsize]
        # scores: (batchsize, 224, 224)
        # gt_mask_list: [(1, 224, 224), ..., batchsize]
        # threshold: float
        # save_dir: str
        # test_imgs_path: [img_path, ..., batchsize]
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, test_imgs_path)

        train_feature_filepath = os.path.join(output_dir, 'features.pkl')
        train_outputs = [self.train_output[0], self.train_output[1], self.train_output[2], threshold]
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].squeeze()  # .transpose(1, 2, 0)
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

        img_name = class_name[i].split("/")[-1]
        fig_img.savefig(os.path.join(save_dir, img_name), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()  # (209,256,56,56)
    _, C2, H2, W2 = y.size()  # (209,512,28,28)
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)  # (209, 1024, 784)
    x = x.view(B, C1, -1, H2, W2)  # torch.Size([209, 256, 4, 28, 28])
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)  # torch.Size([209, 768, 4, 28, 28])
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)  # torch.Size([209, 3072, 784])
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)  # torch.Size([209, 768, 56, 56])

    return z