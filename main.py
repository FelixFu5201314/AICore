# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

import os
import sys
import argparse
import random
import shutil
import time
import json
import warnings
from dotmap import DotMap
from datetime import timedelta
from loguru import logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 添加dao库到sys.path中
from dao import Registers, import_all_modules_for_register     # 获得所有组件的Register, 可以从Registers中获得组件
import dao.utils.dist as comm
from dao.utils import configure_nccl, configure_module, configure_omp

DEFAULT_TIMEOUT = timedelta(minutes=30)


def make_parser():
    parser = argparse.ArgumentParser(description='AICore Arguments')
    # 1.实验配置 & 实验优化
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')  # 随机数
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision training.")    # 是否使用fp16训练
    parser.add_argument("--cache", dest="cache", default=False, action="store_true",
                        help="Caching imgs to RAM for fast training.")  # 是否对数据进行缓存
    parser.add_argument("--occupy", dest="occupy", default=False, action="store_true",
                        help="occupy GPU memory first for training.")   # 是否占据GPU显存
    parser.add_argument("--detail", dest="detail", default=False, action="store_true",
                        help="detail log info.")  # 是否显示详细的log信息
    parser.add_argument("--amp", dest="amp", default=False, action="store_true",
                        help="automatic mixed precision.")  # 是否使用混合精度
    parser.add_argument("--ema", dest="ema", default=False, action="store_true",
                        help="Exponential Moving Average.")  # 是否使用指数移动平均

    # 2.分布式
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')  # 底层协议
    parser.add_argument('--dist-url', default=None, type=str,
                        help='url used to set up distributed training')  # url
    parser.add_argument("--num_machines", default=1, type=int,
                        help="num of node for training")  # 主机数
    parser.add_argument("--machine_rank", default=0, type=int,
                        help="node rank for multi-node training")  # 主机rank
    parser.add_argument('--devices', default=None, type=int,
                        help='devices GPU number to use.')  # 每台机器GPU数量, 多台机器此参数需一致

    # 3.Modules-组件
    parser.add_argument("-c", "--exp_file", default=None, type=str,
                        help="please input your experiment description file")   # 组件配置文件
    parser.add_argument("-m", "--cus_file", default=None, type=str,
                        help="please input your dataloaders description file")

    return parser


@logger.catch
def main():
    parsers = make_parser().parse_args()

    # ---------------- 分布式 or 单卡 ---------------
    num_machines = parsers.num_machines    # 1. 机器数量
    num_gpus_per_machine = comm.get_num_devices() if parsers.devices is None else parsers.devices  # 2. 获得GPU数，每台机器数上的gpu数量应该相等
    assert num_gpus_per_machine <= comm.get_num_devices()
    world_size = num_machines * num_gpus_per_machine  # 3. world size 等于机器数 * 每台机器的GPU数
    if world_size > 1:  # 分布式
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        # 设置dist url
        dist_url = "auto" if parsers.dist_url is None else parsers.dist_url
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto cannot work with distributed training."
            port = comm.find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        # 设置启动方法
        start_method = "spawn"  # torch.multiprocessing 启动方法

        # 设置cache
        # To use numpy memmap for caching image into RAM, we have to use fork method
        if parsers.cache:
            assert sys.platform != "win32", logger.error(
                "As Windows platform doesn't support fork method, "
                "do not add --cache in your training command."
            )
            start_method = "fork"

        # 启动多进程训练
        machine_rank = parsers.machine_rank    # 本机rank
        dist_backend = parsers.dist_backend    # dist backend:nccl
        mp.start_processes(
            _distributed_worker,  # 执行函数
            nprocs=num_gpus_per_machine,  # 启动进程数
            args=(  # 执行函数的参数
                main_worker,    # 启动进程后执行主要函数，即trainer函数
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_backend,
                dist_url,
                (parsers.exp_file, parsers.cus_file, parsers),   # main_worker函数的参数
            ),
            daemon=False,
            start_method=start_method,
        )
    else:   # 单卡
        # Simply call main_worker function
        main_worker(parsers.exp_file, parsers.cus_file, parsers)


def _distributed_worker(
        local_rank,  # 进程id
        main_func,  # 执行函数
        world_size,  # 所有进程数(GPU数)
        num_gpus_per_machine,   # 每台机器的GPU数(进程数)
        machine_rank,   # 当前机器的rank
        backend,    # nccl
        dist_url,   # master主机的ip:port
        args,   # main_func的参数
        timeout=DEFAULT_TIMEOUT,
):
    assert torch.cuda.is_available(), logger.error("cuda is not available. Please check your installation.")
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    main_func(*args)


@logger.catch
def main_worker(modules_file, custom_file, parsers):
    """
    每个进程(GPU)运行的函数
    :param modules_file:    组件文件
    :param custom_file:     自定义组件文件
    :param parsers:         从命令行中获取的内容
    :return:
    """
    # 1.设置实验随机数
    if parsers.seed != 0:
        random.seed(parsers.seed)
        torch.manual_seed(parsers.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # 2.设置环境变量
    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    torch.backends.cudnn.benchmark = True

    # 3.注册所有组件
    if custom_file is not None:
        custom_modules = json.load(open(custom_file))  # load custom modules
    else:
        custom_modules = None
    logger.info("global rank-{}, local rank-{}, register all modules ......".format(
        comm.get_rank(), comm.get_local_rank()))
    import_all_modules_for_register(custom_modules=custom_modules)

    # 4.启动Trainer, trainer自动使用Registers中的组件
    exp = DotMap(json.load(open(modules_file)))   # load config.json
    # 判断status是否满足要求
    status = exp['fullName'].split("-")[-2]
    assert status in ("trainval", "eval", "demo", "export"), \
        logger.error("This status {} is not supported, now supported trainval, eval, demo, export".format(status))
    # 初始化trainer类，并开始训练
    trainer = Registers.trainers.get(exp.trainer.type)(exp, parsers)    # exp modules组件配置字典;parsers 命令行参数
    trainer.run()


if __name__ == '__main__':
    main()

