# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## HBN Boosting Training
```
python -m HBN.boost_train --dataset cifar100 --basemodel resnet18 --stage-epochs 50,50,50,50
```

### HBN basemodel 扩展（backbone）
`--basemodel` 现在支持更多 CIFAR 常用 backbone（大多来自本仓库 `models/`，ConvNeXt/ViT 为 CIFAR 适配实现），并提供两类用法：

- split 版本：固定 4 个 stage（你需要提供 4 个 `--stage-epochs`）
  - `preactresnet18`
  - `resnext29_32x4d`
  - `regnetx_200mf`
  - `regnety_400mf`
  - `efficientnetb0`
  - `densenet121`
  - `convnext_tiny_cifar`
  - `vit_tiny_cifar`
- original* 版本：第 1 个 stage 是完整 backbone feature extractor，后面追加 `--empty-stage-num` 个 identity stage（用于 stage-wise boosting；你需要提供 `1 + empty_stage_num` 个 `--stage-epochs`）
  - `originalpreactresnet18`
  - `originalresnext29_32x4d`
  - `originalregnetx_200mf`
  - `originalregnety_400mf`
  - `originalefficientnetb0`
  - `originaldensenet121`
  - `originalconvnexttinycifar`
  - `originalvittinycifar`

示例（保持你现有命令结构不变，只替换 basemodel）：
```
NO_PROGRESS_BAR=1 nohup python3 -u -m HBN.boost_train \
  --dataset cifar100 \
  --data-dir ./data \
  --basemodel originalregnety_400mf \
  --empty-stage-num 16 \
  --batch-size 512 \
  --stage-epochs 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100 \
  > ./run.log 2>&1 &
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |
