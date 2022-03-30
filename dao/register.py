#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
# @Ref: https://github.com/Delta-ML/delta/blob/master/delta/utils/register.py
# @Ref: https://applenob.github.io/python/register/

"""Module register."""

__all__ = ['import_all_modules_for_register', 'Registers']

import os
import sys
import importlib
from loguru import logger


class Register(object):
    """Module register"""
    def __init__(self, register_name):
        self._dict = {}
        self._name = register_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logger.warning("Key {} already in registry {}.".format(key, self._name))
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            self[key] = value
            return value

        if callable(param):
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')
        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logger.error(f"module {key} not found: {e}")
            raise e

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

    def get(self, key):
        return self.__getitem__(key)


class Registers:
    """All module registers."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    # 1. model
    backbones = Register("backbones")       # 主干网络
    cls_models = Register("cls_models")     # 分类模型
    seg_models = Register("seg_models")     # 分割模型
    det_models = Register("det_models")     # 目标检测模型
    anomaly_models = Register("anomaly_models")     # 异常检测模型

    # 2. optim
    optims = Register("optim")              # 优化器

    # 3. dataset&dataloader
    datasets = Register("datasets")         # 数据集
    dataloaders = Register("dataloaders")   # 数据加载器

    # 4. loss
    losses = Register("losses")             # 损失函数

    # 5. scheduler
    schedulers = Register("schedulers")     # 学习调整策略

    # 6. evaluator
    evaluators = Register("evaluators")     # 验证器

    # 7. trainer
    trainers = Register("trainers")         # 训练过程


def import_all_modules_for_register(custom_modules=None):
    """Import all modules for register."""
    _register_core()    # 注册核心组件
    _register_custom(custom_modules)    # 注册自定义组件


@logger.catch()
def _register_core():
    all_modules = [
        "dao.dataloaders",
        "dao.evaluators",
        "dao.losses",
        "dao.models",
        "dao.optimizers",
        "dao.schedulers",
        "dao.trainers",
    ]

    for modules in all_modules:
        importlib.import_module(modules)
        logger.info(" modules {}(core) loaded ! ".format(modules))


@logger.catch
def _register_custom(custom_modules):
    all_modules = []
    if custom_modules is not None and "custom_modules" in custom_modules:
        custom_modules = custom_modules["custom_modules"]
        if not isinstance(custom_modules, list):
            custom_modules = [custom_modules]
        for module in custom_modules:
            custom_modules_path = os.path.join(module[:module.rfind("/")])
            sys.path.append(custom_modules_path)
            all_modules.append(_path_to_module_format(module))

        for modules in all_modules:
            importlib.import_module(modules)
            logger.info("modules {}(custom) loaded ! ".format(modules))


def _path_to_module_format(py_path):
    """Transform a python file path to module format."""
    return py_path.split("/")[-1].rstrip(".py")








