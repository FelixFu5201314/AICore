CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 0 --devices 2

CUDA_VISIBLE_DEVICES=0,1  NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO  python main.py  --dist-url 'tcp://10.1.130.111:803' --dist-backend 'nccl' --num_machines 2 --machine_rank 1 --devices 2
