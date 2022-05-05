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

    def fit(self, _: DataLoader):
        raise NotImplementedError

    def predict(self, _: tensor):
        raise NotImplementedError

    def evaluate(self, test_dl: DataLoader, output_dir=None) -> Tuple[float, float]:
        """Calls predict step for each test sample."""
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for i, (image, y, mask, image_path) in enumerate(test_dl):
            z_score, fmap = self.predict(image)

            # Normalization
            fmap = fmap.numpy()
            max_score = fmap.max()
            min_score = fmap.min()
            scores = (fmap - min_score) / (max_score - min_score)  # (49, 224, 224)
            self.plot_fig(image[0], scores[0], 0.98, image_path[0])
        #     image_preds.append(z_score.numpy())
        #     image_labels.append(y.numpy())
        #
        #     pixel_preds.extend(fmap.flatten().numpy())
        #     pixel_labels.extend(mask.flatten().numpy())
        #
        # image_preds = np.stack(image_preds)
        #
        # image_rocauc = roc_auc_score(image_labels, image_preds)
        # pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)
        #
        # return image_rocauc, pixel_rocauc
    def denormalization(self, x, mean=[0.335782, 0.335782, 0.335782], std=[0.256730, 0.256730, 0.256730]):
        mean = np.array(mean)
        std = np.array(std)
        x = (((x.numpy().transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
        return x
    def plot_fig(self, test_img, scores, threshold, img_p):
        import matplotlib.pyplot as plt
        import matplotlib
        from skimage import morphology
        from skimage.segmentation import mark_boundaries

        vmax = scores.max() * 255.
        vmin = scores.min() * 255.

        img = self.denormalization(test_img)
        heat_map = scores * 255
        mask = scores
        threshold = np.median(scores) if threshold is None else threshold

        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image')

            ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[1].imshow(img, cmap='gray', interpolation='none')
            ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax_img[1].title.set_text('Predicted heat map')

            ax_img[2].imshow(mask, cmap='gray')
            ax_img[2].title.set_text('Predicted mask')

            ax_img[3].imshow(vis_img)
            ax_img[3].title.set_text('Segmentation result')
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

            dstPath = os.path.join("/ai/data/test", "pictures")
            os.makedirs(dstPath, exist_ok=True)
            ngtype = img_p.split("/")[-2]
            image_name = ngtype + "_" + img_p.split("/")[-1][:-4]+".png"
            fig_img.savefig(os.path.join(dstPath, image_name), dpi=100)
            plt.close()
    def get_parameters(self, extra_params: dict = None) -> dict:
        return {
            "backbone_name": self.backbone_name,
            "out_indices": self.out_indices,
            **extra_params,
        }


@Registers.anomaly_models.register
class PaDiM2(KNNExtractor):
    def __init__(self, backbone, device=None, d_reduced: int = 100, total_dim=None, image_size=224, beta=1):
        super().__init__(
            backbone_name=backbone.type,
            out_indices=(1, 2, 3),
            device=device
        )
        self.image_size = 224
        self.d_reduced = d_reduced  # your RAM will thank you
        self.epsilon = 0.04  # cov regularization
        self.patch_lib = []
        self.resize = None

    def fit(self, train_dataloader, output_dir=None):
        # extract train set features 提取特征
        train_feature_filepath = os.path.join(output_dir, 'features.pkl')  # 特征存放路径
        if not os.path.exists(train_feature_filepath):  # 如果特征不存在
            # 提取特征
            logger.info("1.1 extract train set features")
            for i, (image, mask, label, image_path) in enumerate(train_dataloader):
                logger.info("extract feature iter {}/{}".format(i, len(train_dataloader)))
                feature_maps = self(image)
                if self.resize is None:
                    largest_fmap_size = feature_maps[0].shape[-2:]
                    self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
                resized_maps = [self.resize(fmap) for fmap in feature_maps]
                self.patch_lib.append(torch.cat(resized_maps, 1))  # self.patch_lib = [ torch.Size([32, 1792, 56, 56]), ...]
            self.patch_lib = torch.cat(self.patch_lib, 0)   # 合并特征 torch.Size([240, 1792, 56, 56])

            # random projection
            if self.patch_lib.shape[1] > self.d_reduced:
                logger.info(f"PaDiM: (randomly) reducing {self.patch_lib.shape[1]} dimensions to {self.d_reduced}.")
                self.r_indices = torch.randperm(self.patch_lib.shape[1])[:self.d_reduced]   # 550
                self.patch_lib_reduced = self.patch_lib[:, self.r_indices, ...]     # torch.Size([240, 550, 56, 56])
            else:
                logger.info(f"PaDiM: d_reduced is higher than the actual number of dimensions, copying self.patch_lib ...")
                self.patch_lib_reduced = self.patch_lib

            # calcs
            self.means = torch.mean(self.patch_lib, dim=0, keepdim=True)    # torch.Size([1, 1792, 56, 56])
            self.means_reduced = self.means[:, self.r_indices, ...]     # torch.Size([1, 550, 56, 56])
            x_ = self.patch_lib_reduced - self.means_reduced    # torch.Size([240, 550, 56, 56])

            # cov calc
            self.E = torch.einsum(
                'abkl,bckl->ackl',
                x_.permute([1, 0, 2, 3]),  # transpose first two dims
                x_,
            ) * 1 / (self.patch_lib.shape[0] - 1)  # torch.Size([550, 550, 56, 56])
            self.E += self.epsilon * torch.eye(self.d_reduced).unsqueeze(-1).unsqueeze(-1)  # torch.Size([550, 550, 1, 1])
            self.E_inv = torch.linalg.inv(self.E.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])   # torch.Size([550, 550, 56, 56])

            # save learned distribution
            logger.info("save learned distribution")
            train_outputs = [self.means_reduced, self.E_inv, self.r_indices]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            logger.info('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        self.train_output = train_outputs

    def predict(self, sample):
        feature_maps = self(sample)
        resized_maps = [self.resize(fmap) for fmap in feature_maps]
        fmap = torch.cat(resized_maps, 1)   # torch.Size([32, 1792, 56, 56])

        # reduce
        x_ = fmap[:, self.r_indices, ...] - self.means_reduced  # torch.Size([32, 550, 56, 56])

        left = torch.einsum('abkl,bckl->ackl', x_, self.E_inv)  # torch.Size([32, 550, 56, 56])
        s_map = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))    # torch.Size([32, 56, 56])
        scaled_s_map = torch.nn.functional.interpolate(
            s_map.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear'
        )   # torch.Size([1, 32, 224, 224])

        return torch.max(s_map), scaled_s_map[0, ...]
        # return torch.max(s_map.reshape(s_map.shape[0],-1), dim=1)[0], scaled_s_map[0, ...]

    def get_parameters(self):
        return super().get_parameters({
            "d_reduced": self.d_reduced,
            "epsilon": self.epsilon,
        })


# class SPADE(KNNExtractor):
#     def __init__(
#             self,
#             k: int = 5,
#             backbone_name: str = "resnet18",
#     ):
#         super().__init__(
#             backbone_name=backbone_name,
#             out_indices=(1, 2, 3, -1),
#             pool_last=True,
#         )
#         self.k = k
#         self.image_size = 224
#         self.z_lib = []
#         self.feature_maps = []
#         self.threshold_z = None
#         self.threshold_fmaps = None
#         self.blur = GaussianBlur(4)
#
#     def fit(self, train_dl):
#         for sample, _ in tqdm(train_dl, **get_tqdm_params()):
#             feature_maps, z = self(sample)
#
#             # z vector
#             self.z_lib.append(z)
#
#             # feature maps
#             if len(self.feature_maps) == 0:
#                 for fmap in feature_maps:
#                     self.feature_maps.append([fmap])
#             else:
#                 for idx, fmap in enumerate(feature_maps):
#                     self.feature_maps[idx].append(fmap)
#
#         self.z_lib = torch.vstack(self.z_lib)
#
#         for idx, fmap in enumerate(self.feature_maps):
#             self.feature_maps[idx] = torch.vstack(fmap)
#
#     def predict(self, sample):
#         feature_maps, z = self(sample)
#
#         distances = torch.linalg.norm(self.z_lib - z, dim=1)
#         values, indices = torch.topk(distances.squeeze(), self.k, largest=False)
#
#         z_score = values.mean()
#
#         # Build the feature gallery out of the k nearest neighbours.
#         # The authors migh have concatenated all features maps first, then check the minimum norm per pixel.
#         # Here, we check for the minimum norm first, then concatenate (sum) in the final layer.
#         scaled_s_map = torch.zeros(1, 1, self.image_size, self.image_size)
#         for idx, fmap in enumerate(feature_maps):
#             nearest_fmaps = torch.index_select(self.feature_maps[idx], 0, indices)
#             # min() because kappa=1 in the paper
#             s_map, _ = torch.min(torch.linalg.norm(nearest_fmaps - fmap, dim=1), 0, keepdims=True)
#             scaled_s_map += torch.nn.functional.interpolate(
#                 s_map.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear'
#             )
#
#         scaled_s_map = self.blur(scaled_s_map)
#
#         return z_score, scaled_s_map
#
#     def get_parameters(self):
#         return super().get_parameters({
#             "k": self.k,
#         })



# class PatchCore(KNNExtractor):
#     def __init__(
#             self,
#             f_coreset: float = 0.01,  # fraction the number of training samples
#             backbone_name: str = "resnet18",
#             coreset_eps: float = 0.90,  # sparse projection parameter
#     ):
#         super().__init__(
#             backbone_name=backbone_name,
#             out_indices=(2, 3),
#         )
#         self.f_coreset = f_coreset
#         self.coreset_eps = coreset_eps
#         self.image_size = 224
#         self.average = torch.nn.AvgPool2d(3, stride=1)
#         self.blur = GaussianBlur(4)
#         self.n_reweight = 3
#
#         self.patch_lib = []
#         self.resize = None
#
#     def fit(self, train_dl):
#         for sample, _ in tqdm(train_dl, **get_tqdm_params()):
#             feature_maps = self(sample)
#
#             if self.resize is None:
#                 largest_fmap_size = feature_maps[0].shape[-2:]
#                 self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
#             resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
#             patch = torch.cat(resized_maps, 1)
#             patch = patch.reshape(patch.shape[1], -1).T
#
#             self.patch_lib.append(patch)
#
#         self.patch_lib = torch.cat(self.patch_lib, 0)
#
#         if self.f_coreset < 1:
#             self.coreset_idx = get_coreset_idx_randomp(
#                 self.patch_lib,
#                 n=int(self.f_coreset * self.patch_lib.shape[0]),
#                 eps=self.coreset_eps,
#             )
#             self.patch_lib = self.patch_lib[self.coreset_idx]
#
#     def predict(self, sample):
#         feature_maps = self(sample)
#         resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
#         patch = torch.cat(resized_maps, 1)
#         patch = patch.reshape(patch.shape[1], -1).T
#
#         dist = torch.cdist(patch, self.patch_lib)
#         min_val, min_idx = torch.min(dist, dim=1)
#         s_idx = torch.argmax(min_val)
#         s_star = torch.max(min_val)
#
#         # reweighting
#         m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
#         m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
#         w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
#         _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
#         # equation 7 from the paper
#         m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
#         # Softmax normalization trick as in transformers.
#         # As the patch vectors grow larger, their norm might differ a lot.
#         # exp(norm) can give infinities.
#         D = torch.sqrt(torch.tensor(patch.shape[1]))
#         w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
#         s = w * s_star
#
#         # segmentation map
#         s_map = min_val.view(1, 1, *feature_maps[0].shape[-2:])
#         s_map = torch.nn.functional.interpolate(
#             s_map, size=(self.image_size, self.image_size), mode='bilinear'
#         )
#         s_map = self.blur(s_map)
#
#         return s, s_map
#
#     def get_parameters(self):
#         return super().get_parameters({
#             "f_coreset": self.f_coreset,
#             "n_reweight": self.n_reweight,
#         })
