# SFRP
SFRP: Fine-Grained Point Cloud Classification via Interaction of Spatial and Feature Representation Points

## Install
```
source install.sh
```
Please change the version of the relevant package in `install.sh` according to your CUDA version.

## Dataset
### FG3DPoint
Download the file from *Google Drive* or *Baidu Netdisk*.

**Because of double-blind review, we will release the link after accept.**

Google Drive Link: 

Baidu Netdisk Link: 

### ModelNet40
Please refer to [PointNeXt tutorial](https://guochengqian.github.io/PointNeXt/) to download the datasets. 

## Usage
### Classification
#### Train
```
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/fg3d_air/SFRP.yaml
```
#### Test
```
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/fg3d_air/SFRP.yaml mode=test --pretrained_path /path/to/your/pretrained_model
```
#### Profile Parameters, FLOPs, and Throughput
```
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/fg3d_air/SFRP.yaml flops=True timing=True
```

## Acknowledgment
This repository is built on reusing codes of [OpenPoints](https://github.com/guochengqian/openpoints)， [PointNeXt](https://github.com/guochengqian/PointNeXt) and [PointMetaBase](https://github.com/linhaojia13/PointMetaBase.git)
