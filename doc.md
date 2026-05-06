# (week4) 0131～0213 HBN 实验结果

本周实验对分类头计算权重在 logits space 进行加权集成。
一共有 12 个后续阶段的弱训练，每个阶段都训练 100epoch。
本周 还使用了 不同 loss 的配置进行对比实验，对两类 loss 定义如下：
- loss 1 ：'weighted_cross_entropy(logits_prev.detach() + logits_t, targets, w_batch)'
- loss 2 ：'weighted_cross_entropy(logits_t, targets, w_batch)'

## 分阶段结果（包含 epsilon 和分类头权重 alpha）

### 本周实验设置&&和上周实验的设置对比
week4实验设置 diff：
- 实验 1：使用 loss1+loss2
- 实验 2：仅使用 loss2


week3实验设置 diff：
- 放弃了分类头的权重；loss 采用 loss1+loss2


week2实验设置 diff：
- 使用分类头权重；仅仅使用 loss1


### week4 实验 1 结果
|Stage|Head weight (alpha)|Epsilon|Train acc (%)|Test acc (%)|**Test acc (ensemble, %)**|Test (week3, %)|Test (week2, %)|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|3.887418|0.000420|99.968|75.640|75.640|75.380|74.82|
|2|2.647307|0.004993|99.956|75.170|75.570|75.320|73.38|
|3|1.736587|0.030085|99.952|75.200|75.550|75.080|67.51|
|4|1.544529|0.043561|99.938|74.990|75.450|74.920|73.10|
|5|1.351636|0.062781|99.942|74.830|75.430|74.700|50.72|
|6|1.070658|0.105145|99.894|74.860|75.410|74.540|68.77|
|7|1.228449|0.078936|99.928|74.470|75.310|74.560|70.77|
|8|0.505718|0.266699|99.908|74.730|75.280|74.430|71.54|
|9|0.097779|0.451266|99.934|74.610|75.300|74.470|71.92|
|10|0.258161|0.373713|99.926|74.760|75.260|NA|71.95|
|11|1.342374|0.063879|99.916|74.660|75.160|NA|72.14|
|12|0.100000|0.555752|99.904|74.440|75.150|NA|NA|
|13|0.578198|0.239323|99.916|74.420|75.170|NA|NA|

|stage|分类器权重|epsilon|训练 acc|测试 acc|**测试 acc（集成）**|test(week3结果)|test(week2 结果）|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|3.887418|0.000420|99.968|75.640|75.640|75.380|74.82|
|2|2.647307|0.004993|99.956|75.170|75.570|75.320|73.38|
|3|1.736587|0.030085|99.952|75.200|75.550|75.080|67.51|
|4|1.544529|0.043561|99.938|74.990|75.450|74.920|73.10|
|5|1.351636|0.062781|99.942|74.830|75.430|74.700|50.72|
|6|1.070658|0.105145|99.894|74.860|75.410|74.540|68.77|
|7|1.228449|0.078936|99.928|74.470|75.310|74.560|70.77|
|8|0.505718|0.266699|99.908|74.730|75.280|74.430|71.54|
|9|0.097779|0.451266|99.934|74.610|75.300|74.470|71.92|
|10|0.258161|0.373713|99.926|74.760|75.260|NA|71.95|
|11|1.342374|0.063879|99.916|74.660|75.160|NA|72.14|
|12|0.100000|0.555752|99.904|74.440|75.150|NA|NA|
|13|0.578198|0.239323|99.916|74.420|75.170|NA|NA|


### week4 实验 2 结果
|Stage|Head weight (alpha)|Epsilon|Train acc (%)|Test acc (%)|Test acc (ensemble, %)|Test (Setup 1 %)|Test (Setup2, %)|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|4.167316|0.000240|99.972|77.740|**77.740**|75.380|74.82|
|2|2.759812|0.003991|99.968|77.670|**77.780**|75.320|73.38|
|3|1.874677|0.022992|99.970|77.790|**77.860**|75.080|67.51|
|4|1.015501|0.115986|99.968|77.570|**77.850**|74.920|73.10|
|5|0.856436|0.152792|99.966|77.720|**77.870**|74.700|50.72|
|6|1.314214|0.067331|99.970|77.510|**77.940**|74.540|68.77|
|7|0.370417|0.322822|99.966|77.500|**77.950**|74.560|70.77|
|8|0.680096|0.204209|99.966|77.510|**77.950**|74.430|71.54|
|9|0.448539|0.289651|99.970|77.480|**77.970**|74.470|71.92|
|10|0.032041|0.483985|99.962|77.460|**77.960**|NA|71.95|
|11|0.345904|0.333631|99.970|77.460|**77.990**|NA|72.14|
|12|0.465868|0.282573|99.972|77.410|**77.970**|NA|NA|
|13|0.120395|0.440092|99.968|77.390|**77.970**|NA|NA|

|stage|分类器权重|epsilon|训练 acc|测试 acc|测试 acc（集成）|test(week3结果)|test(week2 结果）|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|4.167316|0.000240|99.972|77.740|**77.740**|75.380|74.82|
|2|2.759812|0.003991|99.968|77.670|**77.780**|75.320|73.38|
|3|1.874677|0.022992|99.970|77.790|**77.860**|75.080|67.51|
|4|1.015501|0.115986|99.968|77.570|**77.850**|74.920|73.10|
|5|0.856436|0.152792|99.966|77.720|**77.870**|74.700|50.72|
|6|1.314214|0.067331|99.970|77.510|**77.940**|74.540|68.77|
|7|0.370417|0.322822|99.966|77.500|**77.950**|74.560|70.77|
|8|0.680096|0.204209|99.966|77.510|**77.950**|74.430|71.54|
|9|0.448539|0.289651|99.970|77.480|**77.970**|74.470|71.92|
|10|0.032041|0.483985|99.962|77.460|**77.960**|NA|71.95|
|11|0.345904|0.333631|99.970|77.460|**77.990**|NA|72.14|
|12|0.465868|0.282573|99.972|77.410|**77.970**|NA|NA|
|13|0.120395|0.440092|99.968|77.390|**77.970**|NA|NA|

## 训练指标

### week4 实验 1 训练指标

![all-stage acc](runs/20260210_184826_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![all-stage loss](runs/20260210_184826_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)

### week4 实验 2 训练指标

![all-stage acc](runs/20260213_104246_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![all-stage loss](runs/20260213_104246_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)



# (week3)0123~0130 HBN 实验结果

本周实验主要聚焦弱分类直接监督 loss 的有效性

前两周实验损失函数为
weighted_cross_entropy(logits_prev.detach() + logits_t, targets, w_batch) 

现在的损失函数是
weighted_cross_entropy(logits_prev.detach() + logits_t, targets, w_batch) + weighted_cross_entropy(logits_t, targets, w_batch)

其中logits_prev.detach()是冻结权重部分的前阶段预测 logits，logits_t 是当前处于训练阶段的预测 logits。

本次实验的一些变化
 - 后续 boost 阶段都采用 100 epoch 训练,boosting 8阶段，总的集合权重大小为44.84MB（第一阶段为43.26MB）
 - 发现后续阶段，样本权重分布集中在少数样本上，损失函数震荡严重，采用不同程度的 alpha clip
 - 单独统计了每个阶段弱分类器的准确率
 
总的来说比之前的训练稳定一些，不会有显著的性能退化。但是还是没有看到稳定的 boosting 提升。

## 实验结果

### 不设置 alpha 上下限

由于 train acc 在第一阶段已经 99%+，因此样本加权分布会越来越集中在少数样本上，导致后续阶段损失函数震荡严重。

![all-stage acc](runs/20260127_141645_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![all-stage loss](runs/20260127_141645_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)

### alpha上限设置为 1

![all-stage acc](runs/20260128_142403_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![all-stage loss](runs/20260128_142403_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)

### alpha 上限设置为 0.1 
![all-stage acc](runs/20260127_212225_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![all-stage loss](runs/20260127_212225_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)

### 三种设定的acc 比较
|Stage|不设上限 Train|不设上限 Test|alpha<=1 Train|alpha<=1 Test|alpha<=0.1 Train|alpha<=0.1 Test|
|---|---:|---:|---:|---:|---:|---:|
|1|99.96|75.38|99.97|75.36|99.97|75.63|
|2|99.97|75.32|99.97|75.24|99.97|75.66|
|3|99.97|75.08|99.97|75.30|99.98|75.58|
|4|99.97|74.92|99.97|75.28|99.98|75.56|
|5|99.97|74.70|99.97|75.30|99.97|75.54|
|6|99.96|74.54|99.97|75.20|99.97|75.52|
|7|99.96|74.56|99.98|75.27|99.96|75.56|
|8|99.96|74.43|99.97|75.20|99.98|75.58|
|9|99.96|74.47|99.96|75.08|99.96|75.58|

#### 各阶段分类器准确率（eval_stage_acc，非集成）

|Stage|不设上限 Train|不设上限 Test|alpha<=1 Train|alpha<=1 Test|alpha<=0.1 Train|alpha<=0.1 Test|
|---|---:|---:|---:|---:|---:|---:|
|1|99.968|75.380|99.964|75.360|99.970|75.630|
|2|99.952|74.830|99.962|75.250|99.972|75.470|
|3|99.934|74.590|99.964|75.210|99.972|75.480|
|4|99.938|74.180|99.960|75.240|99.972|75.570|
|5|99.902|73.760|99.960|75.290|99.974|75.470|
|6|99.860|73.750|99.962|75.020|99.974|75.520|
|7|99.906|73.870|99.962|74.800|99.972|75.560|
|8|99.906|73.920|99.958|74.840|99.972|75.490|
|9|99.892|73.740|99.954|74.700|99.972|75.540|

## Todo
- 也许 boosting 参数量太少了，考虑加大后续分类器参数量（目前训练 epoch 拉满了）
- 目前后续 boosting 显然受到了第一个 stage 特征提取的影响，很难有进一步提升。也许应该从训练集采样分阶段，给后续学习空间？
ß


# （week2）ResNet18 作为第一个分类器训练结果

## 实验设置变更

将第一个分类器设置为标准 ResNet18，最终特征维度来到 `512×4×4`

带来的变化
- boosting起点从强分类器 test acc> 74% 开始
- 特征通道数的加倍带来 incremental block 参数量显著增加，单个 incremental block 权重大小 18M。
- 后续 boosting 效果不是很理想。从几个角度出发做了改进尝试
  - boosting 阶段训练是否充分：incremental block 的训练 epoch 从 20 增加到 100
  - 第一阶段分类器是否过拟合：first classifier lr 增大 epoch 减小， 保证 train acc 尽量小的同时有一个高的 test acc
  - incremental block 是否不够 lightweight： 进一步轻量化 从两层 conv 改为一层
  - 分类器权重 alpha 机制的去除：阶段切换时训练不稳定

## 实验结果

### 1) 将第一个分类器设置为标准 Resnet18：

Shared Trunk：`ResNet-18`； 特征维度： `512×4×4` ；incremental block 训练 epoch：`20`；分类头加权：保留；incremental block 结构：两层

![layer1 all-stage acc](runs/20260123_104938_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![layer1 all-stage loss](runs/20260123_104938_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)

#### 各 stage boosting 结果（Acc）

|Stage|Train Acc (%)|Test Acc (%)|
|---|---:|---:|
|1|99.97|74.74|
|2|99.98|73.98|
|3|99.97|56.02|
|4|99.95|48.51|
|5|99.15|15.99|
|6|99.18|12.75|
|7|97.98|9.17|
|8|77.93|13.46|
|9|77.87|31.61|
|10|79.22|36.41|
|11|78.58|52.81|

现象：发现阶段切换时集成模型性能骤减，损失函数跳变。
我的理解：后续阶段的弱分类器事实上在强分类器之上没有学习到有效信息，却由于高 acc 被加权很大。

### 2) 减少 incremental 层数

 Shared Trunk：`ResNet-18`； 特征维度： `512×4×4` ; incremental block 训练 epoch：`20` ；分类头加权：保留；incremental block 结构：一层

![layer1 all-stage acc](runs/20260122_185411_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![layer1 all-stage loss](runs/20260122_185411_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)

#### 各 stage boosting 结果（Acc）

|Stage|Train Acc (%)|Test Acc (%)|
|---|---:|---:|
|1|99.97|74.82|
|2|99.97|73.38|
|3|99.97|67.51|
|4|98.92|73.10|
|5|98.60|50.72|
|6|94.48|68.77|
|7|98.44|70.77|
|8|99.08|71.54|
|9|99.22|71.92|
|10|99.21|71.95|
|11|99.31|72.14|

现象：减少 incremental block 层数后，还是无法稳定训练。
我的理解：incremental block 层数不是关键。


### 3)  集成和训练阶段 head 权重保持统一为 1，验证在强分类器上能否提升。（样本加权保持不变）
从实验 1,2看出，阶段切换带来 boosting 训练的不稳定十分严重。为了排除这一干扰，后续实验均去除分类头的 alpha 权重更新，集成和训练阶段 head 权重保持统一为 1，验证是否在强分类器后有提升空间。

 Shared Trunk：`ResNet-18`； 特征维度： `512×4×4` ; incremental block 训练 epoch：`100` ；分类头加权：全为 1；incremental block 结构：两层

![layer1 all-stage loss](runs/20260122_140536_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![layer1 all-stage loss](runs/20260122_140536_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)

#### 各 stage boosting 结果（Acc）

|Stage|Train Acc (%)|Test Acc (%)|
|---|---:|---:|
|1|99.97|74.80|
|2|99.97|74.80|
|3|99.97|74.68|

### 4) 在强分类器 train acc 未达到较高水平的时候早停。
具体做法是强分类器 lr 提高 5 倍，epoch 减少到 20，最终效果如下

 Shared Trunk：`ResNet-18`； 特征维度： `512×4×4` ; incremental block 训练 epoch：`100` ；分类头加权：全为 1；incremental block 结构：两层

 ![layer1 all-stage loss](runs/20260122_165917_cifar100_HBNBoost_originalresnet/curves_all_acc.svg)

![layer1 all-stage loss](runs/20260122_165917_cifar100_HBNBoost_originalresnet/curves_all_loss.svg)

#### 各 stage boosting 结果（Acc）

|Stage|Train Acc (%)|Test Acc (%)|
|---|---:|---:|
|1|82.78|66.93|
|2|98.34|66.80|
|3|98.32|66.83|



## 实验分析
- boosting 阶段 train acc 持续增高而 test acc 基本不变。一种猜测：
  - 强分类器已经不存在 class level 的系统性错误，而是只有 instance level 的错误。
  - 后续 incremental block 训练时，主要集中在 instance level 上。
  - train set 和 test set 在 instance level 上不一致，所以 train acc 在 boosting 阶段的提升无法在 test 上看到。
- logits 空间的加权可能的问题
  - 由于 epsilon 是由包含强分类器的集成模型计算出来的，后续弱分类器贡献被高估，权重虚高。
  - 训练时 当前阶段 logits权重为 1，而阶段训练结束后权重马上变为2～3的值集成到 previous logits 上，导致阶段切换的时候模型训练不稳定。如`| epsilon 0.000280 | alpha 4.090220 | test acc 74.89%`

## Todo
- 尝试 alpha 在训练阶段更平滑的接入 logit 的加权中
- 如果强分类器已经没有 class level 的系统性分类错误，可能 incremental learning 的任务设定更符合 HBN（？）。


# （week1）HBN 在 CIFAR100 上的实验结果（ResNet18 前 l 层作为 Shared Backbone）



## 统一的训练/评估参数

- Dataset：CIFAR100
- Incremental Block：lightweight的残差 block（没有改变特征 shape，不知道t>=2 阶段之后需不需要each stage reduces spatial resolution and increases semantic abstraction）
- Boost 设置：第一个为 shared trunk 训练阶段，后续十个阶段为弱分类器 boost 训练
- Stage epochs：第一阶段对于 Shared Trunk 的训练做了多种不同实验，后续每个弱分类器的训练 epoch 目前均为 10
- Batch size：128（train）；test eval batch size：100
- Optimizer：SGD（momentum=0.9，weight_decay=5e-4）
- LR：0.1（每个阶段相同）；
- Alpha：按照公式计算
- Loss：对 merged logits 做 sample-weighted cross entropy（每 batch 按样本权重加权平均）
- 数据增强：RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize；测试仅 Normalize

## 图例说明
- train：训练集ACC/Loss
- test_prev：上一阶段的冻结权重模型在 test 上 ACC/Loss
- test_cand: 上阶段冻结的 logits+本阶段弱分类器 logits 在 test 上的 ACC/Loss




## 1.Shared Trunk 深度对比

三组实验的分阶段训练 stage 数一致，不同的是 shared trunk 深度

- Shared Trunk： `layer1`，输出特征约shape `64×32×32`
- Shared Trunk： `layer1+layer2`，输出特征约shape `128×16×16`
- Shared Trunk： `layer1+layer2+layer3`，输出特征约shape `256×8×8`



## Acc/Loss 曲线图

### 1) Shared Trunk：`ResNet-18 layer1`； 特征维度： `64×32×32`

![layer1 all-stage acc](runs/20260115_004429_cifar100_HBNBoost_resnet_shared/curves_all_acc.svg)

![layer1 all-stage loss](runs/20260115_004429_cifar100_HBNBoost_resnet_shared/curves_all_loss.svg)

### 2) Shared Trunk：`ResNet-18 layer1+layer2`； 特征维度： `128×16×16`

![layer2 all-stage acc](runs/20260115_113905_cifar100_HBNBoost_resnetlargeshared/curves_all_acc.svg)

![layer2 all-stage loss](runs/20260115_113905_cifar100_HBNBoost_resnetlargeshared/curves_all_loss.svg)

### 3) Shared Trunk：`ResNet-18 layer1+layer2+layer3`； 特征维度： `256×8×8`

![3-stage all-stage acc](runs/20260115_112632_cifar100_HBNBoost_resnetXLshared/curves_all_acc.svg)

![3-stage all-stage loss](runs/20260115_112632_cifar100_HBNBoost_resnetXLshared/curves_all_loss.svg)

## 各 stage boosting 结果（Acc）

说明：下表取自各 run 的 `metrics.csv` 中 `split=stage_done` 行（每个 stage 训练结束后，用当前 ensemble 在 test 上评估得到）。

|Stage|64×32×32|128×16×16|256×8×8|
|---|---:|---:|---:|
|1|44.77|47.60|53.89|
|2|47.70|51.76|56.48|
|3|49.38|52.80|57.02|
|4|50.20|52.92|57.52|
|5|50.83|53.21|57.67|
|6|50.97|53.34|57.81|
|7|51.31|53.68|57.84|
|8|51.38|53.70|57.86|
|9|51.58|53.84|57.77|
|10|51.79|53.74|57.81|
|11|51.87|53.81|57.81|

## 各 stage boosting 结果（Loss）

|Stage|64×32×32|128×16×16|256×8×8|
|---|---:|---:|---:|
|1|3.9002|4.5141|3.2306|
|2|3.5003|3.4144|2.1205|
|3|3.1034|2.7286|1.7673|
|4|2.8024|2.3568|1.6320|
|5|2.6093|2.1518|1.5719|
|6|2.4852|2.0283|1.5410|
|7|2.3986|1.9489|1.5236|
|8|2.3364|1.8944|1.5140|
|9|2.2833|1.8592|1.5088|
|10|2.2470|1.8339|1.5064|
|11|2.2158|1.8152|1.5058|

## 简要结论（当前 3 组对比）

- 最终（Stage 11）test acc：
  - 保留 backbone 到 `64×32×32`：51.87%
  - 保留 backbone 到 `128×16×16`：53.81%
  - 保留 backbone 到 `256×8×8`：57.81%

## 2.Shared Trunk 训练 Epoch 数量对比
由于观察到最终集成表现受shared trunk 第一阶段训练结果影响较大，所以采取不同 epoch 数量进行实验
以 shared backbone 保存到第三个 layer，特征 shape 为 `256×8×8`的实验为例子

### 1）Shared Trunk 训练 Epoch = 10（Boost 正常提升 ACC）
![Shared Trunk 训练 Epoch = 10](runs/20260115_004429_cifar100_HBNBoost_resnet_shared/curves_all_acc.svg)

![Shared Trunk 训练 Epoch = 10](runs/20260115_004429_cifar100_HBNBoost_resnet_shared/curves_all_loss.svg)

### 2) Shared Trunk 训练 Epoch = 20
![Shared Trunk 训练 Epoch = 20](runs/20260115_160747_cifar100_HBNBoost_resnetXLshared/curves_all_acc.svg)

![Shared Trunk 训练 Epoch = 20](runs/20260115_160747_cifar100_HBNBoost_resnetXLshared/curves_all_loss.svg)

### 3) Shared Trunk 训练 Epoch = 30
![Shared Trunk 训练 Epoch = 30](runs/20260115_110426_cifar100_HBNBoost_resnetXLshared/curves_all_acc.svg)

![Shared Trunk 训练 Epoch = 30](runs/20260115_110426_cifar100_HBNBoost_resnetXLshared/curves_all_loss.svg)

### 4) Shared Trunk 训练 Epoch = 100

![Shared Trunk 训练 Epoch = 100](runs/20260115_171624_cifar100_HBNBoost_resnetXLshared/curves_all_acc.svg)

![Shared Trunk 训练 Epoch = 100](runs/20260115_171624_cifar100_HBNBoost_resnetXLshared/curves_all_loss.svg)


## 各 stage boosting 结果（Acc）

|Stage|10epoch|20epoch|30epoch|100epoch|
|---|---:|---:|---:|---:|
|1|44.77|65.82|70.07|73.54|
|2|47.70|66.44|69.49|73.70|
|3|49.38|66.48|68.95|73.66|
|4|50.20|66.26|68.42|73.63|
|5|50.83|66.11|68.07|73.70|
|6|50.97|65.90|67.91|73.74|
|7|51.31|65.85|68.02|73.69|
|8|51.38|65.45|67.92|73.68|
|9|51.58|65.43|67.95|73.68|
|10|51.79|65.24|67.95|73.68|
|11|51.87|65.11|67.96|73.71|

### 观察到现象
- shared trunk 训练 epoch 越少，后续集成提升效果更明显
- shared trunk过拟合之后，之后的集成很难提高 acc
- 30 个 epoch 的 case 里面，后续集成 logits 反而降低了 acc
