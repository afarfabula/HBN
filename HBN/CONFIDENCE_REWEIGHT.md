# Confidence-Based Sample Reweight for Shared-Backbone HBN

## 中文

### 背景与动机

在当前的 shared backbone HBN / boosting 链路中，样本权重的更新逻辑本质上是一个二值规则：

- 预测错误的样本被视为困难样本，获得更高权重
- 预测正确的样本不被额外强调

这一规则在弱分类器场景中是自然的，但在 **shared backbone 已经拟合得很好的强分类器** 场景下，会暴露一个明显问题：

- 训练集准确率非常高时，真正被判定为“错误样本”的比例会变得极低
- 例如训练准确率达到 `99.970%` 时，只有约 `0.03%` 的样本会被显式加权
- 随着 boosting stage 持续增加，sample weight 会越来越集中到极少数样本上
- 最终导致样本权重分布非常极端，训练会被少量“顽固样本”主导

这种现象不仅可以从样本权重本身的分布观察到，也可以从 **loss 随 boosting stage 增长的行为** 中间接看出来：随着训练进入后期，loss 越来越容易被少量高权重样本牵引。

因此，我们提出一个更适合强分类器场景的改进方向：

- 不再只关注“预测错误样本”
- 同时关注“预测正确但不够自信的样本”

核心思想是：

- **错误样本** 仍然是困难样本
- **低置信正确样本** 也被视为值得继续强化学习的边界样本

这样可以让样本重加权机制不再只依赖极少数错误样本，而是同时覆盖一部分仍然靠近决策边界的正确样本，从而缓解样本权重极化问题。

### 设计原则

本次改动遵循以下原则：

- 默认 `binary` 模式保持现有行为不变，保证老链路完全兼容
- 新逻辑只在显式开启 confidence 模式时生效
- confidence 统一基于 **集成 sum logits**
- confidence 统一使用 **真实类别概率** `p_y`
- 训练 batch 内的 weighted CE 暂不修改
- 仅修改 stage 结束后的 sample reweight 逻辑
- alpha 仍沿用当前公式

即：

- `epsilon = weighted_mean(difficulty)`
- `alpha = 0.5 * log((1 - epsilon) / epsilon)`

这里的关键变化是：把原来的二值错误率 `errors01` 扩展为更一般的 **difficulty score**。

### 基线方案：Binary

这是当前已有逻辑，作为默认模式保留：

- 若样本预测错误，则 `difficulty = 1`
- 若样本预测正确，则 `difficulty = 0`

也就是：

```text
difficulty_i = 1[y_pred_i != y_i]
```

这个方案的优点是简单直接，但在高训练准确率场景下，difficulty 非零的样本会非常少，从而使权重越来越集中。

### 方案一：Top-k Correct Confidence Reweight

#### 核心想法

除了错误样本之外，再从“预测正确样本”中挑出一小部分 **最不自信** 的样本，作为额外的困难样本。

#### 具体定义

设：

- `p_y` 为模型对真实类别的 softmax 概率
- logits 使用当前集成的 sum logits

则：

- 所有错误样本：`difficulty = 1`
- 所有正确样本中，按 `p_y` 从低到高排序
- 从中选择最低置信的 top-k%
- 这些样本同样记为：`difficulty = 1`
- 其他正确样本：`difficulty = 0`

也就是：

```text
difficulty = 1
  if sample is misclassified
  or sample is correctly classified but belongs to the lowest-confidence top-k%

difficulty = 0
  otherwise
```

#### 直觉解释

该方案仍然保留了“困难样本二值化”的风格，但扩展了困难样本集合：

- 错误样本代表明显没学会的样本
- 低置信正确样本代表靠近决策边界、尚不稳定的样本

因此，它比原始 binary 更适合训练后期的强分类器情形。

#### 默认配置

- 模式名：`topk`
- 默认比例：`0.05`

即默认从 **预测正确样本中选最低置信的 5%**。

#### 优点

- 与原始 binary 逻辑最接近
- 实现简单，解释性强
- 不会一次性把所有正确样本都纳入加权
- 更适合作为第一版稳定替代方案

#### 可能的风险

- 属于硬阈值策略
- 即使整体 confidence 已经很高，也会固定挑出一部分正确样本
- 在某些阶段可能会引入少量“并不真正困难”的样本

### 方案二：Confidence-All Reweight

#### 核心想法

不再用 0/1 的方式定义困难样本，而是给每个样本一个连续的 difficulty 分数。

#### 具体定义

仍然基于真实类别概率 `p_y`，定义：

```text
difficulty = (1 - p_y)^gamma
```

其中：

- `p_y` 越高，说明对真实类别越自信，difficulty 越小
- `p_y` 越低，说明越不自信，difficulty 越大
- 错误样本通常会自动得到较高 difficulty

#### Gamma 的作用

`gamma` 用来控制 difficulty 的非线性强度：

- `gamma = 1`：线性版本，最直接
- `gamma > 1`：更强调低置信样本
- `gamma < 1`：更平缓，更多样本会获得非零 difficulty

默认设置为：

- 模式名：`confall`
- 默认 `gamma = 2.0`

#### 直觉解释

该方案把“样本难度”从二值判断扩展成连续刻画：

- 高置信正确样本：difficulty 接近 0
- 低置信正确样本：difficulty 中等
- 错误样本或极低置信样本：difficulty 接近 1

这样可以显著缓解“只有极少数错样本被不断放大”的问题。

#### 优点

- 比 top-k 更平滑
- 不依赖硬阈值
- 更适合观察样本权重是否从极端尖峰分布回到更平滑的分布

#### 可能的风险

- 会让更多样本参与 reweight
- 若 gamma 不合适，可能削弱 boosting 对真正错误样本的聚焦
- 属于更偏工程启发式的方法，而不是严格的二值 boosting 推导

### 推荐实验顺序

建议优先按以下顺序做对比实验：

1. `binary`
2. `topk`
3. `confall`

原因是：

- `binary` 是当前基线
- `topk` 改动最小，更容易稳定对比
- `confall` 更平滑、更强，但也更容易改变整体行为

### 推荐观察指标

为判断是否缓解了权重极化，建议重点观察：

- 每个 stage 结束后的样本权重分位数
  - `min`
  - `p50`
  - `p90`
  - `p99`
  - `max`

如果后续需要进一步量化权重集中程度，也可以考虑加入：

- ESS（effective sample size）

### 当前命令行接口

新增参数如下：

```bash
--sample-weight-mode {binary,topk,confall}
--sample-weight-topk-ratio 0.05
--sample-weight-gamma 2.0
```

示例：

```bash
python3 -u -m HBN.boost_train \
  --dataset cifar100 \
  --basemodel originalresnet \
  --empty-stage-num 12 \
  --stage-epochs 100,100,100,100,100,100,100,100,100,100,100,100,100 \
  --head-mode hbn \
  --stage0load \
  --loss-mode stage \
  --sample-weight-mode topk \
  --sample-weight-topk-ratio 0.05
```

或：

```bash
python3 -u -m HBN.boost_train \
  --dataset cifar100 \
  --basemodel originalresnet \
  --empty-stage-num 12 \
  --stage-epochs 100,100,100,100,100,100,100,100,100,100,100,100,100 \
  --head-mode hbn \
  --stage0load \
  --loss-mode stage \
  --sample-weight-mode confall \
  --sample-weight-gamma 2.0
```

---

## English

### Background and Motivation

In the current shared-backbone boosting setting, sample reweighting is driven by a binary notion of difficulty: a sample is emphasized if it is misclassified, and it is ignored by the reweighting mechanism if it is classified correctly. This rule is reasonable when the base learner is weak and the training error remains substantial. However, once the shared backbone has already become a well-fitted strong classifier, the same rule becomes increasingly problematic. When training accuracy is extremely high, the set of misclassified samples becomes vanishingly small. For example, when the training accuracy reaches 99.970%, only about 0.03% of the samples are explicitly up-weighted. As boosting proceeds, the weight mass becomes concentrated on this tiny subset of residual mistakes, and the sample-weight distribution becomes highly extreme. In practice, this effect can be observed not only by inspecting the weights themselves, but also indirectly from the stage-wise loss dynamics, which become more and more dominated by a few heavily weighted samples.

This observation motivates a confidence-aware alternative. The main idea is that difficulty should not be defined purely by correctness. Misclassified samples should certainly remain important, but correctly classified samples that are still assigned low confidence should also be treated as informative training signals, because they are often close to the decision boundary and remain unstable under perturbation. A confidence-based strategy therefore extends the focus of reweighting from “wrong samples only” to “wrong samples plus uncertain correct samples,” with the goal of reducing weight collapse while preserving pressure on genuinely hard cases.

### Design Principles

The proposal is intentionally conservative. The original binary scheme is kept as the default behavior so that the existing training pipeline remains fully unchanged unless the confidence-based mode is explicitly enabled. Confidence is defined using the ground-truth class probability under the model’s aggregated prediction, because this quantity directly measures how strongly the model supports the correct answer. The batch-level weighted training objective is left untouched in the first version, and only the stage-end reweighting rule is modified. The alpha update is also kept in the original form, with epsilon interpreted as the weighted mean difficulty. In other words, the main conceptual change is to replace the binary error indicator with a more general difficulty score while leaving the surrounding boosting structure as intact as possible.

### Baseline: Binary

The baseline remains the standard binary rule in which a sample receives unit difficulty if it is misclassified and zero difficulty otherwise. Its strength is its simplicity and its direct connection to classical boosting intuition. Its weakness, in the present setting, is that it becomes too sparse when the backbone is already highly accurate, which is exactly what leads to extreme concentration of sample weights in later stages.

### Method 1: Top-k Correct Confidence Reweight

The first confidence-based option is a top-k strategy. It keeps the binary character of the original rule, but enlarges the set of hard samples. All misclassified samples continue to receive full difficulty. In addition, among the correctly classified samples, one ranks examples by the ground-truth class probability and selects the lowest-confidence fraction, such as the bottom 5%, and assigns them the same difficulty as misclassified samples. The intuition is straightforward: wrong samples represent clearly unresolved cases, while low-confidence correct samples represent boundary cases that are formally correct but still fragile. This method is attractive because it stays close to the original boosting logic, is easy to interpret, and introduces only a limited number of extra emphasized samples. At the same time, it is still a threshold-based mechanism, so it may occasionally select examples that are not truly difficult simply because a fixed fraction must always be chosen.

### Method 2: Confidence-All Reweight

The second option removes the hard threshold altogether and assigns every sample a continuous difficulty score according to its confidence on the true class. A natural choice is to define difficulty as \((1-p_y)^\gamma\), where \(p_y\) is the probability of the ground-truth class and \(\gamma\) controls how sharply the method focuses on low-confidence samples. Under this formulation, highly confident correct samples receive difficulty values close to zero, uncertain correct samples receive intermediate difficulty, and misclassified or very low-confidence samples receive values close to one. Compared with the top-k method, this variant is smoother and does not require choosing a strict selection boundary. It is therefore more suitable when the goal is to soften the weight distribution rather than to enlarge the hard set in a discrete way. Its main limitation is that it departs further from the classic binary boosting view and may spread attention more broadly, so the choice of \(\gamma\) becomes important.

### Recommended Evaluation Order

For empirical study, it is most sensible to compare the methods in the order binary, then top-k, then confidence-all. The binary rule serves as the existing baseline. The top-k strategy introduces the smallest conceptual deviation and is therefore the easiest first replacement to interpret. The confidence-all strategy is the smoothest and potentially the most effective at mitigating weight collapse, but it also changes the training dynamics more substantially and should therefore be evaluated after the simpler alternative.

### Recommended Monitoring

To determine whether confidence-based reweighting actually improves the behavior of the algorithm, the most informative first diagnostic is the stage-wise sample-weight distribution itself. In particular, one should track quantiles such as the minimum, median, upper quantiles, and maximum weight to see whether the distribution remains highly concentrated or becomes more balanced over time. If a more quantitative summary of concentration is needed in later analysis, effective sample size can also be added as a complementary statistic.
