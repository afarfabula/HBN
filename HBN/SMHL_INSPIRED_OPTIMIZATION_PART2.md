## SMHL inspired optimization

Inspired by SMHL, we conduct experiments on the ResNet-18 + CIFAR-100 setup with two modifications: enabling parameter updates for the shared backbone, and feeding multi-scale features from ResNet into multiple classification heads respectively.

The attempt is divided into two parts:

1. Enabling parameter updates for the shared backbone.
2. Using multi-scale features of ResNet as inputs for multiple classification heads.

This document summarizes the experimental result for the second direction in a code-agnostic way. The central question is not whether a strong deep classifier can solve CIFAR-100, because that is already known, but whether shallow and intermediate features in the same backbone also carry useful class-discriminative information when they are exposed to their own heads. In that sense, the experiment is designed as a fast validation of shallow-feature classification ability on CIFAR-100 rather than as a final optimized training recipe.

### Start from a pretrained ResNet checkpoint

The optimization starts from a pretrained ResNet checkpoint so that the experiment begins from a strong solution rather than relearning the task from scratch. After that initialization, multiple heads are attached to multi-scale backbone features and the summed prediction of all heads is optimized. The motivation is simple: we want to quickly test how much class-discriminative signal is already present in shallow and intermediate features on CIFAR-100 once a strong deep representation is available.

### Experimental setup

Two 200-epoch experiments are used here and are referred to as Run 1 and Run 2. Both runs use the same configuration: CIFAR-100, the originalresnet-based multi-scale pipeline, HBN heads, initialization from a pretrained ResNet checkpoint, batch size 128, and learning rate \(10^{-4}\). The prediction is formed by summing the logits of all five heads. The only difference between Run 1 and Run 2 is random variation across runs.

| Experiment | Initialization | Training length | Batch size | Learning rate | Best ensemble acc | Final ensemble acc |
| --- | --- | --- | --- | --- | --- | --- |
| Run 1 | pretrained ResNet checkpoint | 200 epochs | 128 | 1e-4 | 73.05 | 70.91 |
| Run 2 | pretrained ResNet checkpoint | 200 epochs | 128 | 1e-4 | 72.92 | 71.75 |

### Multi-scale feature sizes

The five heads receive features from progressively deeper stages of the backbone, with the following spatial scales:

| Head | Feature shape |
| --- | --- |
| Head 1 | 64 × 32 × 32 |
| Head 2 | 64 × 32 × 32 |
| Head 3 | 128 × 16 × 16 |
| Head 4 | 256 × 8 × 8 |
| Head 5 | 512 × 4 × 4 |

This hierarchy means that the first two heads operate on high-resolution but relatively shallow representations, while the last two heads operate on lower-resolution but much more semantic representations. The experiment therefore provides a direct way to compare how much classification power is available at different depths of the same backbone.

### Overall result

Across the completed 200-epoch runs, the same qualitative pattern appears consistently. Starting from a strong pretrained solution gives a reliable high-accuracy initialization. After moving to the multi-head multi-scale objective, the best ensemble accuracy appears relatively early, typically around the mid-30s to low-50s epochs, and stays in the low-73% range. By the end of 200 epochs, the final ensemble accuracy drops further to roughly 71–72%. This means that the current multi-scale training strategy does not improve over the strong pretrained baseline, even though it does produce a reasonably strong ensemble.

### Stage-by-stage behavior of the heads

The logs show a very consistent depth-wise pattern, so it is useful to expand the behavior explicitly for both runs.

#### Run 1

| Phase-2 epoch | Ensemble acc | Head 1 | Head 2 | Head 3 | Head 4 | Head 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 70.80 | 4.19 | 5.45 | 8.90 | 18.19 | 71.03 |
| 9 | 70.63 | 7.91 | 18.54 | 34.25 | 62.17 | 71.63 |
| 19 | 72.21 | 8.53 | 24.63 | 44.03 | 67.51 | 72.24 |
| 49 | 72.56 | 10.04 | 33.25 | 54.81 | 68.34 | 72.07 |
| 99 | 72.29 | 14.15 | 39.95 | 59.00 | 69.29 | 71.97 |
| 149 | 71.24 | 16.28 | 42.58 | 59.67 | 68.65 | 71.42 |
| 199 | 70.91 | 17.09 | 45.51 | 60.21 | 68.57 | 70.32 |

#### Run 2

| Phase-2 epoch | Ensemble acc | Head 1 | Head 2 | Head 3 | Head 4 | Head 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 72.13 | 4.60 | 6.14 | 10.49 | 20.90 | 72.28 |
| 9 | 71.19 | 7.77 | 18.13 | 35.11 | 62.82 | 72.02 |
| 19 | 72.39 | 8.43 | 24.28 | 44.43 | 67.58 | 72.43 |
| 49 | 72.16 | 10.02 | 33.45 | 54.67 | 68.16 | 71.92 |
| 99 | 71.84 | 14.72 | 39.83 | 59.62 | 68.19 | 71.63 |
| 149 | 70.89 | 16.01 | 43.14 | 60.35 | 68.19 | 70.68 |
| 199 | 71.75 | 16.66 | 44.75 | 61.09 | 69.95 | 71.54 |

#### Final head-wise summary

| Head | Feature shape | Run 1 final acc | Run 2 final acc | Interpretation |
| --- | --- | --- | --- | --- |
| Head 1 | 64 × 32 × 32 | 17.09 | 16.66 | Very shallow features; weak standalone classifier |
| Head 2 | 64 × 32 × 32 | 45.51 | 44.75 | Early features contain some category signal but remain limited |
| Head 3 | 128 × 16 × 16 | 60.21 | 61.09 | Middle-depth features become clearly discriminative |
| Head 4 | 256 × 8 × 8 | 68.57 | 69.95 | Deep intermediate features are already strong classifiers |
| Head 5 | 512 × 4 × 4 | 70.32 | 71.54 | Deepest feature remains the dominant predictor |

At the beginning of training, the ensemble is already dominated by the deepest head, while the shallower heads contribute very little. As optimization proceeds, the intermediate heads learn nontrivial decision boundaries, especially the third and fourth heads, and this demonstrates that multi-scale supervision does reveal useful class information in earlier layers. However, the shallowest head never becomes competitive with the deepest one, and the final ensemble remains anchored by the deepest representation.

### Interpretation

As the experimental result for the second SMHL-inspired direction, the conclusion is mixed but clear. On the positive side, attaching heads to multi-scale features verifies that shallow and intermediate backbone features contain usable classification information on CIFAR-100, and that information becomes increasingly strong from middle depth onward. On the negative side, the current multi-scale optimization does not convert that supervision into a better final classifier than the strong pretrained baseline. In other words, the experiment is successful as a diagnostic study of feature quality across depth, but it is not yet successful as a final accuracy-improving training strategy.

### Start from scratch

As a direct supplement to the pretrained-start setting above, we also examine multi-scale training from scratch on the ResNet-18 backbone. Here the model is trained without a pretrained checkpoint, while still following the stage-wise multi-head setup. This comparison is useful because it shows whether the multi-scale design alone is sufficient to yield good performance, or whether starting from a strong pretrained solution is essential.

Two runs are used for this comparison. Both use CIFAR-100, batch size 64, baseline heads, and stage-wise fulltrain optimization. The only difference is the boosting-stage schedule.

| Experiment | Initialization | Stage schedule | Batch size | Best final-stage candidate acc | Final acc |
| --- | --- | --- | --- | --- | --- |
| Run 1 | from scratch | 10,10,10,10,100 | 64 | 62.01 | 61.35 |
| Run 2 | from scratch | 10,10,30,30,50 | 64 | 61.63 | 60.53 |

#### Run 1

| Boosting stage | Epochs | Final stage acc | Final stage loss | Best candidate acc within stage | Final prev-logit acc |
| --- | --- | --- | --- | --- | --- |
| Stage 1 | 10 | 6.87 | 4.1543 | 6.89 | 6.87 |
| Stage 2 | 10 | 21.74 | 3.2598 | 21.74 | 7.32 |
| Stage 3 | 10 | 39.40 | 2.3158 | 39.40 | 23.98 |
| Stage 4 | 10 | 52.49 | 1.7818 | 52.54 | 44.01 |
| Stage 5 | 100 | 61.35 | 2.2559 | 62.01 | 54.94 |

#### Run 2

| Boosting stage | Epochs | Final stage acc | Final stage loss | Best candidate acc within stage | Final prev-logit acc |
| --- | --- | --- | --- | --- | --- |
| Stage 1 | 10 | 6.81 | 4.1590 | 6.82 | 6.81 |
| Stage 2 | 10 | 21.63 | 3.2618 | 21.63 | 7.17 |
| Stage 3 | 30 | 51.96 | 1.7907 | 52.04 | 24.41 |
| Stage 4 | 30 | 60.84 | 1.5828 | 60.84 | 54.90 |
| Stage 5 | 50 | 60.53 | 2.2763 | 61.63 | 61.81 |

These start-from-scratch runs show a different behavior from the pretrained-start experiments. Instead of beginning from a strong solution and probing how much shallow and intermediate features can contribute, the optimization must first build the representation itself. As a result, the stage-wise accuracy rises gradually from very low values in the early stages and only reaches about 60–61% at the end. This is much lower than the pretrained-start multi-scale experiments, whose best ensemble accuracy stays in the low-73% range. Therefore, the evidence suggests that for this multi-scale setup, a strong pretrained starting point is important if the goal is to use multiple heads as a probe of shallow and intermediate feature quality rather than to relearn the whole problem from scratch.
