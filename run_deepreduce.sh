#!/usr/bin/env bash

### tf_cnn_benchmarks ResNet20 ResNet50
./grace/env-tf1.14/bin/mpirun \
-x CUDA_VISIBLE_DEVICES=0 \
-x NCCL_IB_DISABLE=0 \
-np 8 -H 11.0.0.233:1,11.0.0.234:1,11.0.0.235:1,11.0.0.236:1,11.0.0.237:1,11.0.0.238:1,11.0.0.239:1,11.0.0.240:1, \
--display-allocation -map-by slot -bind-to none -nooversubscribe \
-mca pml ob1 -mca btl ^openib --tag-output --mca btl_tcp_if_include enp1s0f1 -x NCCL_SOCKET_IFNAME=enp1s0f1 \
python \
./grace-benchmarks/tensorflow/Classification/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet20_v2 --data_name=cifar10 --batch_size=256 --weight_decay=0.0001 --optimizer=momentum --piecewise_learning_rate_schedule=0.1;163;0.01;245;0.001 --variable_update=horovod --train_dir={log_dir}/ckpts --summary_verbosity=1 --save_summaries_steps=10 --num_epochs=328 --eval_during_training_every_n_steps=200 --num_eval_epochs=8 --data_dir=/mnt/scratch/cifar-10-batches-py/

./grace/env-tf1.14/bin/mpirun \
-x CUDA_VISIBLE_DEVICES=0 \
-x NCCL_IB_DISABLE=0 \
-np 8 -H 11.0.0.233:1,11.0.0.234:1,11.0.0.235:1,11.0.0.236:1,11.0.0.237:1,11.0.0.238:1,11.0.0.239:1,11.0.0.240:1, \
--display-allocation -map-by slot -bind-to none -nooversubscribe \
-mca pml ob1 -mca btl ^openib --tag-output --mca btl_tcp_if_include enp1s0f1 -x NCCL_SOCKET_IFNAME=enp1s0f1 \
./grace/env-tf1.14/bin/python \
./grace-benchmarks/tensorflow/Classification/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --batch_size=256 --weight_decay=1e-4 --optimizer=momentum --nodistortions --variable_update=horovod --train_dir={log_dir}/ckpts --summary_verbosity=1 --save_summaries_steps=100 --num_epochs=90 --eval_during_training_every_n_steps=625 --num_eval_epochs=8 --data_dir=/mnt/scratch/imagenet18/data/imagenet



### pytorch resnet20 cifar10 data volume and micro-benchmark

./grace/env-tf1.14/bin/mpirun \
-x CUDA_VISIBLE_DEVICES=0 \
-x NCCL_IB_DISABLE=0 \
-np 8 -H 11.0.0.233:1,11.0.0.234:1,11.0.0.235:1,11.0.0.236:1,11.0.0.237:1,11.0.0.238:1,11.0.0.239:1,11.0.0.240:1, \
--display-allocation -map-by slot -bind-to none -nooversubscribe \
-mca pml ob1 -mca btl ^openib --tag-output --mca btl_tcp_if_include enp1s0f1 -x NCCL_SOCKET_IFNAME=enp1s0f1 \
./grace/env-tf1.14/bin/python \
./grace-benchmarks/torch/cifar10/trainer_grace.py \
-a resnet20 --data=/mnt/scratch/cifar-10-batches-py/ --log_volume \
--grace_config="{'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather', 'compress_ratio': 0.01, 'deepreduce':'index', 'index':'bloom', 'micro-benchmark':True}"


### NCF sparse/dense/deepreduce accuracy

./grace/env-tf1.14/bin/mpirun \
-x CUDA_VISIBLE_DEVICES=0 \
-x NCCL_IB_DISABLE=0 \
-np 2 -H 11.0.0.203:1,11.0.0.204:1 \
--display-allocation -map-by slot -bind-to none -nooversubscribe \
-mca pml ob1 -mca btl ^openib --tag-output --mca btl_tcp_if_include ens1f0 -x NCCL_SOCKET_IFNAME=ens1f0 \
./grace/env-tf1.14/bin/python -W ignore::UserWarning \
./grace-benchmarks/torch/Recommendation/NCF/ncf_grace.py \
--data ./data/ml-20m/torch/cache/ml-20m \
--load_checkpoint_path ./grace/results/NCF/checkpoints/model_init.pth --seed 44 \
 --weak_scaling --extra_wandb_tags=accuracy,10G --log_volume \
--grace_config="{'compressor': 'none', 'memory': 'none', 'communicator': 'allreduce'}"



./grace/env-tf1.14/bin/mpirun \
-x CUDA_VISIBLE_DEVICES=1 \
-x NCCL_IB_DISABLE=0 \
-np 8 -H 11.0.0.233:1,11.0.0.234:1,11.0.0.235:1,11.0.0.236:1,11.0.0.237:1,11.0.0.238:1,11.0.0.239:1,11.0.0.240:1, \
--display-allocation -map-by slot -bind-to none -nooversubscribe \
-mca pml ob1 -mca btl ^openib --tag-output --mca btl_tcp_if_include enp136s0f0 -x NCCL_SOCKET_IFNAME=enp136s0f0 \
./grace/env-tf1.14/bin/python -W ignore::UserWarning \
./grace/src/grace-benchmarks/torch/Recommendation/NCF/ncf_grace.py \
--data ./grace/data/ml-20m/torch/cache/ml-20m \
--load_checkpoint_path ./grace/results/NCF/checkpoints/model_init.pth --seed 44 \
 --weak_scaling --extra_wandb_tags=accuracy,10G --log_volume \
--grace_config="{'compressor': 'threshold', 'memory': 'none', 'communicator': 'allgather', 'threshold': 0.0}"

# "{'compressor': 'threshold', 'memory': 'none', 'communicator': 'allgather', 'threshold': 0.0, 'deepreduce':'index', 'index':'bloom'}"
# "{'compressor': 'threshold', 'memory': 'none', 'communicator': 'allgather', 'threshold': 0.0, 'deepreduce':'value', 'value':'polyfit'}"
# "{'compressor': 'threshold', 'memory': 'none', 'communicator': 'allgather', 'threshold': 0.0, 'deepreduce':'value', 'value':'qsgd', 'buckets_num':128, 'quantum_num': 32}"
# "{'compressor': 'threshold', 'memory': 'none', 'communicator': 'allgather', 'threshold': 0.0, 'deepreduce':'both'}"
# "{'compressor': 'threshold', 'memory': 'none', 'communicator': 'allgather', 'threshold': 0.0, 'deepreduce':'index', 'index':'bloom', 'policy':'p0', 'fpr':0.01}"
# "{'compressor': 'threshold', 'memory': 'none', 'communicator': 'allgather', 'threshold': 0.0, 'deepreduce':'both', 'index':'bloom', 'policy':'random', 'fpr':0.01, 'value':'qsgd'}"
# "{'compressor': 'threshold', 'memory': 'none', 'communicator': 'allgather', 'threshold': 0.0, 'deepreduce':'both', 'index':'bloom', 'policy':'random', 'fpr':0.01, 'value':'qsgd'}"


# SKCompress
./grace/env-tf1.14/bin/mpirun \
-x CUDA_VISIBLE_DEVICES=1 \
-x NCCL_IB_DISABLE=0 \
-np 8 -H 11.0.0.233:1,11.0.0.234:1,11.0.0.235:1,11.0.0.236:1,11.0.0.237:1,11.0.0.238:1,11.0.0.239:1,11.0.0.240:1, \
--display-allocation -map-by slot -bind-to none -nooversubscribe \
-mca pml ob1 -mca btl ^openib --tag-output --mca btl_tcp_if_include enp136s0f0 -x NCCL_SOCKET_IFNAME=enp136s0f0 \
./grace/env-tf1.14/bin/python -W ignore::UserWarning \
./grace/src/grace-benchmarks/torch/Recommendation/NCF/ncf_grace.py \
--data ./grace/data/ml-20m/torch/cache/ml-20m \
--load_checkpoint_path ./grace/results/NCF/checkpoints/model_init.pth --seed 44 \
 --weak_scaling --extra_wandb_tags=accuracy,10G \
--grace_config="{'compressor': 'SKCompressCPU','num_quantiles':128, 'sparsifier': 'threshold', 'threshold': 0.0, \
'memory': 'none', 'communicator': 'allgather'}" --micro_benchmark  --log_volume



### NCF time breakdown

./grace/env-tf1.14/bin/mpirun \
-x CUDA_VISIBLE_DEVICES=0 \
-x NCCL_IB_DISABLE=1 \
-np 4 -H 10.0.0.133:1,10.0.0.134:1,10.0.0.135:1,10.0.0.136:1 \
--display-allocation -map-by slot -bind-to none -nooversubscribe \
-mca pml ob1 -mca btl ^openib --tag-output --mca btl_tcp_if_include enp1s0f1 -x NCCL_SOCKET_IFNAME=enp1s0f1 \
./grace/env-tf1.14/bin/python -W ignore::UserWarning \
./grace/src/grace-benchmarks/torch/Recommendation/NCF/ncf_grace.py \
--data ./grace/data/ml-20m/torch/cache/ml-20m \
--load_checkpoint_path ./grace/results/NCF/checkpoints/model_init.pth --seed 44 \
 --weak_scaling --extra_wandb_tags=timer,10G -e=3 --grads_accumulated=10 --log_time \
--grace_config="{'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather', 'compress_ratio': 0.1, 'deepreduce':'index', 'index':'bloom'}"