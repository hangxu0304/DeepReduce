# DeepReduce

A Sparse-tensor Communication Framework for Distributed Deep Learning

**Abstrat**: Sparse tensors appear frequently in distributed deep learning, either as a direct artifact of the deep neural network’s gradients, or as a result of an explicit sparsification process. Existing communication primitives are agnostic to the peculiarities of deep learning; consequently, they impose unnecessary communication overhead. We introduce DeepReduce, a versatile framework for the compressed communication of sparse tensors, tailored for distributed deep learning. DeepReduce decomposes sparse tensors in two sets, values and indices, and allows both independent and combined compression of these sets. We support a variety of common compressors, such as Deflate for values, or run-length encoding for indices. We also propose two novel compression schemes that achieve superior results: curve fitting-based for values and bloom filter-based for indices. DeepReduce is orthogonal to existing gradient sparsifiers and can be applied in conjunction with them, transparently to the end-user, to significantly lower the communication overhead. As proof of concept, we implement our approach on TensorFlow and PyTorch. Our experiments with large real models demonstrate that DeepReduce transmits fewer data and imposes lower computational overhead than existing methods, without affecting the training accuracy.

## Prerequisites

The code is built with following libraries:

- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.4
- [TensorFlow](https://www.tensorflow.org/) >= 1.14
- [GRACE](https://github.com/sands-lab/grace) >=1.0

## Benchmarks

We use the following benchmarks to run our experiments:

- [Image Classification/tf_cnn_benchmarks](https://github.com/sands-lab/grace-benchmarks/tree/master/tensorflow/Classification/tf_cnn_benchmarks) [TensorFlow] ResNet-20, ResNet-50
- [Image_Classification/Cifar10](https://github.com/sands-lab/grace-benchmarks/tree/master/torch/cifar10) [PyTorch] ResNet-20
- [Recommendation/NCF](https://github.com/sands-lab/grace-benchmarks/tree/master/torch/Recommendation/NCF) [PyTorch] NCF

## Usage

For the usage of GRACE and environment setup etc., please check the guides [here](https://github.com/sands-lab/grace).

First, create a GRACE instance from `params`. `params` should include parameters for both GRACE and DeepReduce. The valid parameter options for DeepReduce is listed as below:

```python
'''
'deepreduce': None, 'value', 'index', 'both'
'value': None, 'polyfit', ...(other custom methods)
'index': None, 'bloom', ...(other custom methods)
'''
from grace_dl.dist.helper import grace_from_params
params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'allgather', 'compress_ratio': 0.01, 'deepreduce':'index', 'index':'bloom'}
grc = grace_from_params(params)
```

Once you get a desired GRACE instance, warp the compressor by DeepReduce. After that, you can use DeepReduce in the same way as GRACE.

```python
deepreduce_wrapper = {'value': ValueCompressor,
                      'index': IndexCompressor,
                      'both': DeepReduce}
DReduce = deepreduce_wrapper[deepreduce](grc.compressor, params)
grc.compressor = DReduce
```

## Scripts & Data

We provide the bash scripts to reproduce the experiments in our paper. All experiment results are also available at [WANDB](https://wandb.ai/sands-lab/deepreduce/reports/DeepReduce--VmlldzoxODM5NTU) database.

## Citation

If you find this useful, please cite our work as:

Kostopoulou K, Xu H, Dutta A, et al. DeepReduce: A Sparse-tensor Communication Framework for Distributed Deep Learning[J]. arXiv preprint arXiv:2102.03112, 2021.

Here's a BibTeX blurb:

```
@misc{kostopoulou2021deepreduce,
      title={DeepReduce: A Sparse-tensor Communication Framework for Distributed Deep Learning}, 
      author={Kelly Kostopoulou and Hang Xu and Aritra Dutta and Xin Li and Alexandros Ntoulas and Panos Kalnis},
      year={2021},
      eprint={2102.03112},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Publications

* DeepReduce: A Sparse-tensor Communication Framework for Distributed Deep Learning [[arXiv]](https://arxiv.org/abs/2102.03112)<br>
