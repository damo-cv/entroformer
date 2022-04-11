![Python >=3.7](https://img.shields.io/badge/Python->=3.7-yellow.svg)
![PyTorch >=1.7](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)

# [ICLR2022] Entroformer: A Transformer-based Entropy Model for Learned Image Compression [[pdf]](https://arxiv.org/abs/2202.05492)

The official repository for [Entroformer: A Transformer-based Entropy Model for Learned Image Compression](https://arxiv.org/abs/2202.05492).

## Pipeline

![framework](figs/framework.jpg)

## Evaluation on [Kodak](http://r0k.us/graphics/kodak/) Dataset

![result](figs/result.jpg)

## Requirements

### Prerequisites

Clone the repo and create a conda environment as follows:

```bash
conda create --name entroformer python=3.7
conda activate entroformer
conda install pytorch=1.7 torchvision cudatoolkit=10.1
pip install torchac
```

(We use PyTorch 1.7, CUDA 10.1. We use torchac for arithmetic coding.)

### Test Dataset

[Kodak](http://r0k.us/graphics/kodak/) Dataset

```
kodak
├── image1.jpg 
├── image2.jpg
└── ...
```

## Evaluation & Comress & Decompress

**Evaluation:**

```bash
# Kodak
sh test.sh [/path/to/kodak] [model_path]
(sh test_parallel.sh [/path/to/kodak] [model_path])
```

**Compress:**

```bash
sh compress.sh original.png [model_path]
(sh compress_parallel.sh original.png [model_path])
```

**Decompress:**

```bash
sh decompress.sh original.bin [model_path]
(sh decompress_parallel.sh original.bin [model_path])
```

## Trained Models

Download the pre-trained [models]()(coming soon) optimized by MSE.

Note: We reorganize code and the performances are slightly different from the paper's.

## Acknowledgement

Codebase from [L3C-image-compression](https://github.com/fab-jul/L3C-PyTorch) , [torchac](https://github.com/fab-jul/torchac)

## Citation

If you find this code useful for your research, please cite our paper

```
@InProceedings{Yichen_2022_ICLR,
    author    = {Qian, Yichen and Lin, Ming and Sun, Xiuyu and Tan, Zhiyu and Jin, Rong},
    title     = {Entroformer: A Transformer-based Entropy Model for Learned Image Compression},
    booktitle = {International Conference on Learning Representations},
    month     = {May},
    year      = {2022},
}
```

