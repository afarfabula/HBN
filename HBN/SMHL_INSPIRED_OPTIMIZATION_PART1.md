## SMHL inspired optimization

Inspired by SMHL, we conduct experiments on the ResNet-18 + CIFAR-100 setup with two modifications: enabling parameter updates for the shared backbone, and feeding multi-scale features from ResNet into multiple classification heads respectively.

The attempt is divided into two parts:

1. Enabling parameter updates for the shared backbone.
2. Using multi-scale features of ResNet as inputs for multiple classification heads.

This document summarizes the first direction. The goal here is to keep the HBN architecture and the stage-wise boosting structure conceptually unchanged, while allowing the shared backbone to be updated during training. At the same time, we keep the SMHL-style intuition that only the current head is actively optimized at each stage, and the loss is still formed stage by stage rather than turning the problem into a fully unconstrained joint training setup. In this sense, the experiment is intended to test a minimal modification: whether simply opening the shared backbone for optimization is already enough to improve the stage-wise training behavior.

### Experimental setup

Two runs are considered here and are referred to as Run 1 and Run 2. Both runs use CIFAR-100, the originalresnet family, batch size 128, and stage-wise fulltrain optimization with shared backbone updates enabled. The only difference between the two runs is the stage schedule.

| Experiment | Initialization | Stage schedule | Batch size | Final acc |
| --- | --- | --- | --- | --- |
| Run 1 | from scratch | 50,50,50,50 | 128 | 65.92 |
| Run 2 | from scratch | 50,100,100,100 | 128 | 65.20 |

### Overall result

The logs show a consistent pattern in both runs. The first stage already learns the strongest classifier, reaching about 75% test accuracy, while the following stages do not continue to improve the overall result. Instead, the final stage-wise accuracy drops into the mid-60% range. Extending the later stages from 50 epochs to 100 epochs does not change this pattern. Therefore, simply enabling shared-backbone training is not sufficient to turn the stage-wise procedure into a stronger optimization strategy in this setting.

### Stage-wise results

#### Run 1

| Boosting stage | Epochs | Final stage acc | Final stage loss | Best candidate acc within stage | Final prev-logit acc |
| --- | --- | --- | --- | --- | --- |
| Stage 1 | 50 | 75.75 | 0.9688 | 75.80 | 75.75 |
| Stage 2 | 50 | 67.90 | 1.5685 | 61.59 | 0.05 |
| Stage 3 | 50 | 70.52 | 1.8850 | 67.68 | 1.13 |
| Stage 4 | 50 | 65.92 | 3.0967 | 63.50 | 0.06 |

#### Run 2

| Boosting stage | Epochs | Final stage acc | Final stage loss | Best candidate acc within stage | Final prev-logit acc |
| --- | --- | --- | --- | --- | --- |
| Stage 1 | 50 | 74.80 | 1.0171 | 75.11 | 74.80 |
| Stage 2 | 100 | 66.63 | 1.4825 | 64.01 | 0.10 |
| Stage 3 | 100 | 66.86 | 1.5654 | 63.88 | 0.03 |
| Stage 4 | 100 | 65.20 | 2.3595 | 61.42 | 0.08 |

These tables show that the degradation is not a one-stage accident. In both runs, Stage 1 provides the best overall result, while the later stages fail to improve on top of it. Stage 2 causes a sharp drop, Stage 3 recovers part of the loss but still stays below Stage 1, and Stage 4 ends lower again. More importantly, the previous-logit path also deteriorates severely once later stages begin. In Run 1, the final previous-logit accuracy falls from 75.75 at Stage 1 to 0.05, 1.13, and 0.06 in Stages 2–4. In Run 2, it similarly collapses from 74.80 to 0.10, 0.03, and 0.08. The longer schedule in Run 2 makes the later stages more stable numerically, but it does not reverse the qualitative trend.

### Interpretation

As the result for the first SMHL-inspired direction, the conclusion is that opening the shared backbone alone does not solve the optimization problem. The first stage can still build a reasonably strong classifier from scratch, but later boosting stages do not successfully leverage the additional freedom of the shared backbone to produce consistent gains. In other words, preserving the original stage-wise structure while simply allowing backbone updates is not enough to obtain a stronger final model in this setup. This suggests that the bottleneck is not merely whether the backbone is trainable, but how stage-wise objectives, head freezing, and inter-stage coordination interact during optimization.
