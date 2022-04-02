# AICore

## 1. 容器制作
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
## 2. 如何部署

## 3. 如何启动
```
一、 分类
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