# AI 核心库

#### **功能添加说明**

- 

## 一、项目由来

在**工业流程**中，深度学习应用过程包括：

1. TrainVal(针对特定场景，特定数据集训练一个模型)
2. Eval(使用验证/测试集测试，得到工业上的性能指标)
3. Demo(将模型做个demo给客户展示)
4. Export(将模型转成其他格式)
5. Deploy(将模型部署到具体的设备上，并权衡速度与准确率)
6. APP(将整个工业流程封装成带界面的APP)

在**深度学习训练**中，训练过程包括以下组件：

- dataloaders(数据加载)
- models(模型：分类、分割、检测、异常检测等[Task](https://paperswithcode.com/sota))
- losses(损失函数)
- optims(优化器)
- schedulers(学习率调整策略)
- evaluator(训练过程中的性能评价)
- trainers(训练过程, 将上述组件联系起来, 并且包含一些调优trick)
    - resume
    - fune turning
    - 日志监控(训练过程日志输出，tensorboard，...)
    - 权重输出
    - multigpus(是否使用多GPU形式训练)
    - mixedprecisions(是否使用混合精度进行训练)
    - ......

目前，我所遇到的深度学习项目基本都能用这两个维度概括，为了方便以后使用，在此将两个维度整理成这个项目.

## 二、项目结构
### 1. 目录介绍
```shell
|-- configs # 分类、分割、目标检测、异常检测配置文件
|   |-- SemanticSegmentation
|   |-- ObejctDetection
|   |-- AnomalyDetection
|   `-- ImageClassification
|-- custom_modules  # 自定义组件
|   `-- __init__.py
|-- dao   # 核心库
|   |-- __init__.py
|   |-- dataloaders # 数据集、数据加载器、数据增强等
|   |-- evaluators  # 验证器
|   |-- losses      # 损失函数
|   |-- models      # 模型
|   |   |-- ImageClassification   # 分类
|   |   |-- ObejctDetection       # 目标检测
|   |   |-- SemanticSegmentation  # 分割
|   |   |-- anomaly               # 异常检测
|   |   `-- backbone              # 主干网络
|   |-- optimizers  # 优化器
|   |-- schedulers  # 学习率调整策略
|   |-- trainers    # 训练器
|   |-- register.py # 注册, dao中所有组件都注册在这里
|   `-- utils       # 工具库
|-- main.py         # 启动文件
|-- notes           # 笔记
|   |-- SemanticSegmentation
|   |-- ObejctDetection
|   |-- AnomalyDetection
|   `-- ImageClassification
```
### 2. Train流程
```shell
|- trainer
|	|- before_train(train之前的操作，eg. dataloader,model,optim,... setting)
|	|	|- 1.logger setting：日志路径，tensorboard日志路径，日志重定向等
|	|	|- 2.model setting：获取模型
|	|	|- 3.optimizer setting：获取优化器，不同权重层，优化器参数不同的设置
|	|	|- 4.resume setting：resume，fune turning等设置
|	|	|- 5.dataloader setting：数据集dataset定义-->Transformer(数据增强）-->Dataloader（数据加载)--> ...
|	|   |- 6.loss setting: 损失函数选择，有的实验可以略掉，因为在model中定义了
|	|   |- 7.scheduler setting：学习率调整策略选择
|	|   |- 8.other setting: 补充2model setting，EMA，DDP模型等设置
|	|   |- 9.evaluator setting：验证器设置，包括读取验证集，计算评价指标等
|	|- train_in_epoch(训练一个epoch的操作)
|	|	|- before_epoch(一个epoch之前的操作)
|	|	|	|- 判断此次epoch使用进行马赛克增强；
|	|	|	|- 修改此次epoch的损失函数；
|	|	|	|- 修改此次epoch的日志信息格式;
|	|	|	|- ...
|	|	|- train_in_iter(训练一次iter的操作，一次完整的forward&backwards)
|	|	|	|- before_iter(一次iter之前的操作)
|	|	|	|	|- nothing todo
|	|	|	|- train_one_iter(一次iter的操作)
|	|	|	|	|- 1.记录data time和iter time
|	|	|	|	|- 2.(预)读取数据
|	|	|	|	|- 3.forward
|	|	|	|	|- 4.计算loss
|	|	|	|	|- 5.backwards
|	|	|	|	|- 6.optimizer 更新网络权重
|	|	|	|	|- 7.是否进行EMA操作
|	|	|	|	|- 8.lr_scheduler修改学习率
|	|	|	|	|- 9.记录日志信息(datatime,itertime,各种loss,...)
|	|	|	|- after_iter(一次iter之后的操作)
|	|	|	|	|- 1.打印一个iter的日志信息(epoch,iter,losses,gpu mem,lr,eta,...)
|	|	|	|	|- 2.是否进行图片的resize，即多尺度训练
|	|	|- after_epoch(一个epoch之后的操作)
|	|	|	|- 1.保存模型
|	|	|	|- 2.是否进行，evaluator
|	|- after_train(训练之后的操作)
|	|	|- 输出最优的结果
```

### 3. Eval/Demo/Export流程

Eval/Demo/Export是Train中的子集，或者过程比较简单，此处不再赘述，看代码即可

## 三、环境搭建

### 1. 容器制作

本项目的实验环境全部是在Docker中完成的，docker镜像的制作过程如下

```
sudo docker pull nvcr.io/nvidia/pytorch:22.02-py3
sudo docker run -it -v /home/user/data:/root/temp nvcr.io/nvidia/pytorch:22.02-py3 bash

apt-get install net-tools
apt-get install inetutils-ping
apt install ssh
vim /etc/ssh/sshd_config
修改root可登录
把AIServer, AICore, setup, start.sh文件放到/ai目录下
ln -s /ai/data/AITorch/hub/ /root/.cache/torch/
ln -s /opt/conda/bin/python /usr/bin/python
ln -s /opt/conda/bin/pip /usr/bin/pip
pip install python-socketio eventlet nvgpu xlsxwriter # AIServer
pip install dotmap loguru albumentations torchcam timm torchsummary     # AICore
```

## 四、如何使用
### 1. ImageClassification

```
1. trainval
# 单机单卡
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# 单机多卡
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# 多机多卡
# use master ip 10.1.130.111
CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json
CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# use docker ip 172.17.0.2
CUDA_VISIBLE_DEVICES=0,1  python main.py  --dist-url 'tcp://172.17.0.2:1234' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json
CUDA_VISIBLE_DEVICES=0,1  python main.py  --dist-url 'tcp://172.17.0.2:1234' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

2. eval
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-ClsDataloader-ClsEvaluator-eval-linux.json

3. demo
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-demo-linux.json

4. export
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-export-linux.json

```

### 2. SemanticSegmentation

```
1. trainval
# 单机单卡
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-sgdWarmupBiasBnWeight-segDataset-crossEntropyLoss-warmCosLr-segEvaluator-gpus-trainval-linux.json

# 单机多卡
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 2 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-sgdWarmupBiasBnWeight-segDataset-crossEntropyLoss-warmCosLr-segEvaluator-gpus-trainval-linux.json

# 多机多卡
# use master ip 10.1.130.111
CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-sgdWarmupBiasBnWeight-segDataset-crossEntropyLoss-warmCosLr-segEvaluator-gpus-trainval-linux.json
CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-sgdWarmupBiasBnWeight-segDataset-crossEntropyLoss-warmCosLr-segEvaluator-gpus-trainval-linux.json

# use docker ip 172.17.0.2
CUDA_VISIBLE_DEVICES=0,1  python main.py  --dist-url 'tcp://172.17.0.2:1234' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-sgdWarmupBiasBnWeight-segDataset-crossEntropyLoss-warmCosLr-segEvaluator-gpus-trainval-linux.json
CUDA_VISIBLE_DEVICES=0,1  python main.py  --dist-url 'tcp://172.17.0.2:1234' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-sgdWarmupBiasBnWeight-segDataset-crossEntropyLoss-warmCosLr-segEvaluator-gpus-trainval-linux.json

2. eval
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-segDataset-segEvaluator-eval-linux.json

3. demo
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-demo-linux.json

4. export
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/SemanticSegmentation/seg-UNet_resnet50-export-linux.json

```

### 3. ObjectDetection

```
1. trainval
# 单机单卡
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# 单机多卡
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# 多机多卡
# use master ip 10.1.130.111
CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json
CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# use docker ip 172.17.0.2
CUDA_VISIBLE_DEVICES=0,1  python main.py  --dist-url 'tcp://172.17.0.2:1234' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json
CUDA_VISIBLE_DEVICES=0,1  python main.py  --dist-url 'tcp://172.17.0.2:1234' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

2. eval
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-ClsDataloader-ClsEvaluator-eval-linux.json

3. demo
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-demo-linux.json

4. export
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-export-linux.json

```

### 4. AnomalyDetection

```
1. trainval
# 单机单卡
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# 单机多卡
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# 多机多卡
# use master ip 10.1.130.111
CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json
CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

# use docker ip 172.17.0.2
CUDA_VISIBLE_DEVICES=0,1  python main.py  --dist-url 'tcp://172.17.0.2:1234' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json
CUDA_VISIBLE_DEVICES=0,1  python main.py  --dist-url 'tcp://172.17.0.2:1234' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-sgdWarmupBiasBnWeight-clsDataloader-crossEntropyLoss-warmCosLr-clsEvaluator-gpus-trainval-linux.json

2. eval
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-ClsDataloader-ClsEvaluator-eval-linux.json

3. demo
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-demo-linux.json

4. export
CUDA_VISIBLE_DEVICES=0,1  python main.py --num_machines 1 --machine_rank 0 --devices 1 -c /ai/AICore/configs/ImageClassification/cls-efficientnetb0-export-linux.json

```



## 五、支持模型

### 1. BackBone:[notes](notes/BackBone/README.md)
- timm:[code ref](https://github.com/AICoreRef/pytorch-image-models)
### 2. ImageClassification:[notes](notes/ImageClassification/README.md)
分类全部在timm基础上修改的, 因此[code ref](https://github.com/AICoreRef/pytorch-image-models)同上
- efficientnetb0
- efficientnetb1
- resnet18
- resnet34
- resnet50
- resnet101

#### 效果对比

| 模型 | 外观数据PZ-验证集top1 | 外观数据PZ-测试集top1 | 权重 |
| ---- | --------------------- | --------------------- | ---- |
|      |                       |                       |      |

### 3. SemanticSegmentation:[notes](notes/SemanticSegmentation/README.md)
- Unet:[code ref](https://github.com/AICoreRef/segmentation_models.pytorch)
- Unet++:[code ref](https://github.com/AICoreRef/segmentation_models.pytorch)
    - https://github.com/yassouali/pytorch-segmentation

#### 效果对比

注意：以下实验均是跑通即可，均未调优

| 模型   | 主干网络 | PascalVoc-验证集mIoU | PascalVoc -测试集mIoU | 权重 |
| ------ | -------- | -------------------- | --------------------- | ---- |
| UNet   | resnet50 | 0.69                 | 未测                  |      |
| UNet++ | resnet50 |                      |                       |      |


### 4. ObjectDetection:[notes](notes/ObjectDetection/README.md)
    - https://github.com/Megvii-BaseDetection/YOLOX
    - https://zhuanlan.zhihu.com/p/391396921
    - YOLOX课程: https://edu.51cto.com/center/course/lesson/index?id=774242
    - FPN: https://cloud.tencent.com/developer/article/1546594
    - PAFPN: https://zhuanlan.zhihu.com/p/397020975
    - YOLOX结构图: https://blog.csdn.net/nan355655600/article/details/119329727

### 5. AnomalyDetection:[notes](notes/AnomalyDetection/README.md)
- PaDim [code ref](https://github.com/AICoreRef/PaDiM-Anomaly-Detection-Localization-master)

## 六、组件说明

整个项目是由trainer(train,eval,demo,export), dataloader, dataset,model,optimizer,loss,scheduler,evaluator组件组成。其中，trainer负责将其他组件拼接起来，接下来会说明一些组件，但是不是每个都说下，详细请看代码。

- 分布式组件
  - Pytorch原生分布式
    - DDP原理1: https://zhuanlan.zhihu.com/p/76638962 
    - DDP原理2: https://zhuanlan.zhihu.com/p/343951042
    - DDP随机种子: https://bbs.cvmart.net/articles/5491
- Tricks
  - 

## 七、部署说明

深度学习模型训练完毕后需要部署在嵌入式、主机、手机等设备上，部署过程请参考我的另一个项目[AIDeploy](https://github.com/FelixFu520/AIDeploy)

