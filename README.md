# Residual Attention Network for Image Classification

##  Introduction
This repository contains a re-implementation of Residual Attention Network based on the paper [Residual Attention Network for Image Classification](https://arxiv.org/pdf/1704.06904.pdf).

The Residual Attention Network adopts mixed attention mechanism into very deep structure for image classification tasks. It is built by stacking Attention Modules, which generate attention-aware features from low resolution and mapping back to original feature maps.

## Dataset
- Used **CIFAR-10** and **CIFAR-100** which consist of 50,000 training set and 10,000 test set with 32 x 32 RGB images, representing 10 or 100 different image labels [here](https://www.cs.toronto.edu/~kriz/cifar.html). 
- Data augmentation techniques are also applied, which generate image shifting and horizontal flip.


## Implemented network structure
| Layer       | Output Size | Attention-56 Detail | Attention-92 Detail |
| ----------- | ----------- | ------------------- | ------------------- |
| Conv2D | 32x32 | 5x5, stride=1 | 5x5, stride=1 |
| Max pooling | 16x16 | 2x2, stride=2 | 2x2, stride=2 |
| Residual Unit | 16x16 | x1 | x1 |
| Attention Stage 1 | 16x16 | attention x1 | attention x1 |
| Residual Unit | 8x8  | x1 | x1 |
| Attention Stage 2 | 8x8 | attention x1 | attention x2 |
| Residual Units | 4x4 | x1 | x1 |
| Attention Stage 3 | 4x4 | attention x1 | attention x3 |
| Residual Unit | 4x4 | x1 | x1 |
| Residual Unit | 4x4 | x3 | x3 |
| AvgPooling2D | 1x1 | 4x4, stride=1 | 4x4, stride=1 |
| Dense (FC, softmax) | 10 | x1 | x1 |


## Code and system dependencies
* Python 3.8+
* Tensorflow, Matplotlib
* Tensorflow-gpu 2.0

## Usage

### Customize training attention model

`utils/model_utils.py`: contains the class ResidualAttentionNetwork

### Customize training parameters

Jupyter notebooks are also provided, as mentioned in the directory organization below, for each configuration that we have trained the model for. For example, `CIFAR10_attn56_arl_spatial.ipynb` generates work for Attention-56 attention residual learning with spatial attention.


## Organization of the directory

```./
├── CIFAR100_attn56_arl_spatial.ipynb
├── CIFAR100_attn56_nal_mixed.ipynb
├── CIFAR10_attn56_arl_mixed.ipynb
├── CIFAR10_attn56_arl_spatial.ipynb
├── CIFAR10_attn56_nal_mixed.ipynb
├── CIFAR10_attn92_arl_spatial.ipynb
├── E4040_2021Fall_RANC_report_rl3164_so2639_st3425.pdf
├── README.md
└── utils
    ├── learning_mech.py
    ├── model_utils.py
    └── res_unit.py
