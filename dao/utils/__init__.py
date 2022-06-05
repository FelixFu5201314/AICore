# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:


# 1.分类,分割 评价指标  —— evaluator,trainer组件会引用
from .metricCls import MeterClsTrain
from .metricCls import MeterClsEval
from .metricCls import plot_confusion_matrix
from .metricSeg import MeterSegTrain, MeterSegEval
from .metricDet import MeterDetEval, MeterDetTrain


# 2.多GPU工具    —— dataloader组件会引用
from .dist import get_num_devices
from .dist import wait_for_the_master
from .dist import is_main_process
from .dist import synchronize
from .dist import get_world_size
from .dist import get_rank
from .dist import get_local_rank
from .dist import get_local_size
from .dist import time_synchronized
from .dist import gather
from .dist import all_gather
from .dist import find_free_port  # 查找空闲端口
from .dist import synchronize  # 当所有进程都到barrier时，才继续执行


# 3.环境设置IB、NCCL、OpenCV等
from .setup_env import configure_nccl
from .setup_env import configure_module
from .setup_env import configure_omp


# 4. all reduce norm
from .allreduce_norm import all_reduce_norm


# 5.logger 配置
from .logger import setup_logger


# 6.模型保存、模型加载
from .checkpoint import save_checkpoint, load_ckpt


# 7.数据预读取
from .data_prefetcher import DataPrefetcherCls, DataPrefetcherSeg, DataPrefetcherDet


# 8.占据显存、显存剩余大小
from .metrics import occupy_mem, gpu_mem_usage


# 9.EMA 指数平均
from .ema import EMA, ModelEMA, is_parallel


# 10.显示设置
from .visualize import denormalization  # 反归一化
from .palette import get_palette, colorize_mask  # 分割可视化

# 11.YOLOX 目标检测boxes.py文件定义的同居
from .boxes import (
    filter_box,
    postprocess,
    bboxes_iou,
    matrix_iou,
    adjust_box_anns,
    xyxy2xywh,
    xyxy2cxcywh,
)

# 12. YoloV3所需要函数
from .detUtils import multi_gt_creator
