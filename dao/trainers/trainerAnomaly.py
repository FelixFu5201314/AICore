
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Base On:

import os
import datetime
import numpy as np
import shutil
import json
import pickle
from PIL import Image
from loguru import logger

import onnx
import torch.onnx
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms as T

from dao import Registers
from dao.utils import setup_logger
from dao.dataloaders.augments import get_transformer
from dao.utils import get_rank, get_local_rank, get_world_size  # 导入分布式库


@Registers.trainers.register
class AnomalyTrainer:
    def __init__(self, exp, parser):
        self.exp = exp   # DotMap 格式 的配置文件
        self.parser = parser  # 命令行配置文件

        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')  # 此次trainer的开始时间
        self.data_type = torch.float16 if self.parser.fp16 else torch.float32  # 使用的数据类型
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.parser.amp)  # 在训练开始之前实例化一个Grad Scaler对象

        # anomaly只支持单机单卡
        assert self.parser.devices == 1, "exp.envs.gpus.devices must 1, please set again "
        assert self.parser.num_machines == 1, "exp.envs.gpus.devices must 1, please set again "
        assert self.parser.machine_rank == 0, "exp.envs.gpus.devices must 0, please set again "

    def run(self):
        self._before_train()
        self._train()
        self._after_train()

    def _before_train(self):
        """
        1.Logger Setting
        2.Model Setting;    包含fit和evaluate
        3.DataLoader Setting;
        """
        if self.parser.record:
            self.output_dir = os.path.join(self.exp.trainer.log_dir, self.exp.name, self.start_time)  # 日志目录
        else:
            self.output_dir = os.path.join(self.exp.trainer.log_dir, self.exp.name)  # 日志目录
            if get_rank() == 0:
                if os.path.exists(self.output_dir):  # 如果存在self.output_dir删除
                    try:
                        shutil.rmtree(self.output_dir)
                        pass
                    except Exception as e:
                        logger.info("global rank {} can't remove tree {}".format(get_rank(), self.output_dir))
        setup_logger(self.output_dir, distributed_rank=get_rank(), filename=f"train_log.txt", mode="a")  # 设置只有rank=0输出日志，并重定向

        logger.warning("Anomaly Detection only supported Single Machine and Single GPU !!!!")
        logger.info("....... Train Before, Setting something ...... ")

        logger.info("1. Logging Setting ...")
        logger.info(f"create log file {self.output_dir}/train_log.txt")  # log txt
        self.exp.pprint(pformat='json') if self.parser.detail else None  # 根据parser.detail来决定日志输出的详细
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:    # 将配置文件写到self.output_dir
            json.dump(dict(self.exp), f)
        logger.info(f"create Tensorboard log {self.output_dir}")
        self.tblogger = SummaryWriter(self.output_dir) if get_rank() == 0 else None  # log tensorboard

        logger.info("2. Model Setting ...")
        self.device = torch.device("cuda:{}".format(self.parser.gpu))
        self.model = Registers.anomaly_models.get(self.exp.model.type)(
            self.exp.model.backbone,
            device=self.device,
            **self.exp.model.kwargs)  # get model from register

        logger.info("3. Dataloader Setting ...")
        self.train_loader = Registers.dataloaders.get(self.exp.dataloader.type)(
            dataset=self.exp.dataloader.dataset,
            **self.exp.dataloader.kwargs)
        self.val_loader = Registers.dataloaders.get(self.exp.evaluator.type)(
            dataset=self.exp.evaluator.dataset,
            **self.exp.evaluator.kwargs)

        logger.info("train start now .......")

    def _train(self):
        self.model.fit(self.train_loader, output_dir=self.output_dir)

    def _after_train(self):
        self.model.evaluate(self.val_loader, output_dir=self.output_dir)


@Registers.trainers.register
class AnomalyDemo:
    def __init__(self, exp, parser):
        self.exp = exp  # DotMap 格式 的配置文件
        self.parser = parser  # 命令行配置文件

        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')  # 此次trainer的开始时间
        self.data_type = torch.float16 if self.parser.fp16 else torch.float32  # 使用的数据类型
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.parser.amp)  # 在训练开始之前实例化一个Grad Scaler对象

        # anomaly只支持单机单卡
        assert self.parser.devices == 1, "exp.envs.gpus.devices must 1, please set again "
        assert self.parser.num_machines == 1, "exp.envs.gpus.devices must 1, please set again "
        assert self.parser.machine_rank == 0, "exp.envs.gpus.devices must 0, please set again "

    def run(self):
        self._before_demo()
        self._demo()

    def _before_demo(self):
        """
        1.Logger Setting
        2.Model Setting;    包含fit和evaluate
        3.DataLoader Setting;
        """
        if self.parser.record:
            self.output_dir = os.path.join(self.exp.trainer.log_dir, self.exp.name, self.start_time)  # 日志目录
        else:
            self.output_dir = os.path.join(self.exp.trainer.log_dir, self.exp.name)  # 日志目录
            if get_rank() == 0:
                if os.path.exists(self.output_dir):  # 如果存在self.output_dir删除
                    try:
                        shutil.rmtree(self.output_dir)
                    except Exception as e:
                        logger.info("global rank {} can't remove tree {}".format(get_rank(), self.output_dir))
        setup_logger(self.output_dir, distributed_rank=get_rank(), filename=f"demo_log.txt",
                     mode="a")  # 设置只有rank=0输出日志，并重定向

        logger.warning("Anomaly Detection only supported Single Machine and Single GPU !!!!")
        logger.info("....... Demo Before, Setting something ...... ")

        logger.info("1. Logging Setting ...")
        logger.info(f"create log file {self.output_dir}/demo_log.txt")  # log txt
        self.exp.pprint(pformat='json') if self.parser.detail else None  # 根据parser.detail来决定日志输出的详细
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:  # 将配置文件写到self.output_dir
            json.dump(dict(self.exp), f)
        logger.info(f"create Tensorboard log {self.output_dir}")
        self.tblogger = SummaryWriter(self.output_dir) if get_rank() == 0 else None  # log tensorboard

        logger.info("2. Model Setting ...")
        self.device = torch.device("cuda:{}".format(self.parser.gpu))
        # 读取训练好的模型
        with open(self.exp.trainer.ckpt, 'rb') as f:
            train_output = pickle.load(f)
        self.model = Registers.anomaly_models.get(self.exp.model.type)(
            self.exp.model.backbone,
            device=self.device,
            select_index=train_output[2],
            **self.exp.model.kwargs)  # get model from register

        logger.info("3. Dataloader Setting ...")
        # 存放所有测试图片路径
        all_paths = [os.path.join(self.exp.images.path, p) for p in os.listdir(self.exp.images.path) if self._img_ok(p)]
        self.images = []
        transform_x = T.Compose([T.Resize(self.exp.images.resize, Image.ANTIALIAS),
                                 T.CenterCrop(self.exp.images.cropsize),
                                 T.ToTensor(),
                                 T.Normalize(mean=self.exp.images.mean,
                                             std=self.exp.images.std)])
        for img_p in sorted(all_paths):
            image = transform_x(Image.open(img_p).convert('RGB'))
            self.images.append((image, image.shape, img_p))

        logger.info("demo start now .......")

    def _img_ok(self, img_p):
        flag = False
        for m in self.exp.images.image_ext:
            if img_p.endswith(m):
                flag = True
        return flag

    def _demo(self):
        import pickle
        from tqdm import tqdm
        from scipy.spatial.distance import mahalanobis
        from scipy.ndimage import gaussian_filter

        # 读取训练好的模型
        with open(self.exp.trainer.ckpt, 'rb') as f:
            train_output = pickle.load(f)

        embedding_vectors = self.model(self.images)

        logger.info("calculate multivariate Gaussian distribution, this will take a minute ......")
        logger.info("this operate will use cpu, please Reserve sufficient resources ......")
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        embedding_vectors = embedding_vectors.transpose(0, 2, 1)    # (550,550,3136)->(3136,550,550)
        dist_list = []
        logger.info("Evaluate calculate cov:")
        # for i in tqdm(range(H * W), desc="Evaluate calculate cov::"):
        for i in range(H * W):
            if i % 10 == 0:
                logger.info("{}/{}".format(i, len(range(H * W))))
            mean = train_output[0][i, :]
            conv_inv = np.linalg.inv(train_output[1][i, :, :])
            dist = [mahalanobis(sample[i, :], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)  # torch.Size([49, 56, 56])
        score_map = F.interpolate(dist_list.unsqueeze(1), size=224, mode='bilinear',
                                  align_corners=False).squeeze().numpy()  # (49, 224, 224)

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # 读取阈值信息
        with open(self.exp.trainer.threshold, 'r') as threshold_file:
            threshold = eval(threshold_file.readline())
            logger.info("threshold is {}".format(str(threshold)))
            max_score = eval(threshold_file.readline())
            logger.info("max_score is {}".format(str(max_score)))
            min_score = eval(threshold_file.readline())
            logger.info("min_score is {}".format(str(min_score)))

        # Normalization
        # max_score = score_map.max()
        # min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)  # (49, 224, 224)



        # 绘制每张test图片预测信息
        # test_imgs:(3, 224, 224)
        # scores: (224, 224)
        # threshold: float
        # test_imgs_path: str
        for i in range(len(self.images)):
            image = torch.tensor(self.images[i][0])
            score = scores[i]
            print(type(score))
            print(score.shape)
            img_p = self.images[i][2]
            self.plot_fig(image, score, threshold, img_p)

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

        img = self.denormalization(test_img, self.exp.images.mean, self.exp.images.std)
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

            dstPath = os.path.join(self.output_dir, "pictures")
            os.makedirs(dstPath, exist_ok=True)
            ngtype = img_p.split("/")[-2]
            image_name = ngtype + "_" + img_p.split("/")[-1][:-4]+".png"
            fig_img.savefig(os.path.join(dstPath, image_name), dpi=100)
            plt.close()


@Registers.trainers.register
class AnomalyExport:
    def __init__(self, exp, parser):
        self.exp = exp  # DotMap 格式 的配置文件
        self.parser = parser  # 命令行配置文件

        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')  # 此次trainer的开始时间
        self.data_type = torch.float16 if self.parser.fp16 else torch.float32  # 使用的数据类型
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.parser.amp)  # 在训练开始之前实例化一个Grad Scaler对象

        # anomaly只支持单机单卡
        assert self.parser.devices == 1, "exp.envs.gpus.devices must 1, please set again "
        assert self.parser.num_machines == 1, "exp.envs.gpus.devices must 1, please set again "
        assert self.parser.machine_rank == 0, "exp.envs.gpus.devices must 0, please set again "

    def _before_export(self):
        """
        1.Logger Setting
        2.Model Setting;    包含fit和evaluate
        3.DataLoader Setting;
        """
        if self.parser.record:
            self.output_dir = os.path.join(self.exp.trainer.log_dir, self.exp.name, self.start_time)  # 日志目录
        else:
            self.output_dir = os.path.join(self.exp.trainer.log_dir, self.exp.name)  # 日志目录
            if get_rank() == 0:
                if os.path.exists(self.output_dir):  # 如果存在self.output_dir删除
                    try:
                        shutil.rmtree(self.output_dir)
                    except Exception as e:
                        logger.info("global rank {} can't remove tree {}".format(get_rank(), self.output_dir))
        setup_logger(self.output_dir, distributed_rank=get_rank(), filename=f"export_log.txt",
                     mode="a")  # 设置只有rank=0输出日志，并重定向

        logger.warning("Anomaly Detection only supported Single Machine and Single GPU !!!!")
        logger.info("....... Export Before, Setting something ...... ")

        logger.info("1. Logging Setting ...")
        logger.info(f"create log file {self.output_dir}/export_log.txt")  # log txt
        self.exp.pprint(pformat='json') if self.parser.detail else None  # 根据parser.detail来决定日志输出的详细
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:  # 将配置文件写到self.output_dir
            json.dump(dict(self.exp), f)
        logger.info(f"create Tensorboard log {self.output_dir}")
        self.tblogger = SummaryWriter(self.output_dir) if get_rank() == 0 else None  # log tensorboard

        logger.info("2. Model Setting ...")
        # 读取训练好的模型
        with open(self.exp.trainer.ckpt, 'rb') as f:
            train_output = pickle.load(f)
        self.device = torch.device("cpu")
        self.model = Registers.anomaly_models.get(self.exp.model.type)(
            self.exp.model.backbone,
            device=self.device,
            select_index=train_output[2],
            **self.exp.model.kwargs)  # get model from register

    def run(self):

        self._before_export()
        x = torch.randn(self.exp.onnx.x_size)

        onnx_path = os.path.join(self.output_dir, "export.onnx")
        torch.onnx.export(self.model,
                          x,
                          onnx_path,
                          **self.exp.onnx.kwargs)
