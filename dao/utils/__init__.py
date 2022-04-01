# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:


# 分类评价指标  —— evaluator,trainer组件会引用
from .metricCls import MeterClsTrain
from .metricCls import MeterClsEval
from .metricCls import plot_confusion_matrix


# 多GPU工具    —— dataloader组件会引用
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


# 环境设置IB、NCCL、OpenCV等
from .setup_env import configure_nccl
from .setup_env import configure_module
from .setup_env import configure_omp


# all reduce norm
from .allreduce_norm import all_reduce_norm


# logger 配置
from .logger import setup_logger


# 模型保存、模型加载
from .checkpoint import save_checkpoint, load_ckpt


# 数据预读取
from .data_prefetcher import DataPrefetcherCls, DataPrefetcherSeg, DataPrefetcherDet


# 占据显存、显存剩余大小
from .metrics import occupy_mem, gpu_mem_usage


# EMA 指数平均
from .ema import EMA, ModelEMA, is_parallel
