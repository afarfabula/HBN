# Top-k Sample Weight Ratio Comparison

## Commands

| Setting | Command | Run |
| --- | --- | --- |
| binary baseline | `CUDA_VISIBLE_DEVICES=4 python3 -u -m HBN.boost_train --dataset cifar100 --basemodel originalresnet --empty-stage-num 12 --stage-epochs 100,100,100,100,100,100,100,100,100,100,100,100,100 --head-mode hbn --stage0load --loss-mode stage` | `20260403_015824_cifar100_HBNBoost_originalresnet` |
| topk 0.05 | `CUDA_VISIBLE_DEVICES=4 python3 -u -m HBN.boost_train --dataset cifar100 --basemodel originalresnet --empty-stage-num 12 --stage-epochs 100,100,100,100,100,100,100,100,100,100,100,100,100 --head-mode hbn --stage0load --loss-mode stage --sample-weight-mode topk --sample-weight-topk-ratio 0.05` | `20260403_011635_cifar100_HBNBoost_originalresnet` |
| topk 0.1 | `CUDA_VISIBLE_DEVICES=4 python3 -u -m HBN.boost_train --dataset cifar100 --basemodel originalresnet --empty-stage-num 12 --stage-epochs 100,100,100,100,100,100,100,100,100,100,100,100,100 --head-mode hbn --stage0load --loss-mode stage --sample-weight-mode topk --sample-weight-topk-ratio 0.1` | `20260403_021241_cifar100_HBNBoost_originalresnet` |
| topk 0.2 | `CUDA_VISIBLE_DEVICES=4 python3 -u -m HBN.boost_train --dataset cifar100 --basemodel originalresnet --empty-stage-num 12 --stage-epochs 100,100,100,100,100,100,100,100,100,100,100,100,100 --head-mode hbn --stage0load --loss-mode stage --sample-weight-mode topk --sample-weight-topk-ratio 0.2` | `20260403_021020_cifar100_HBNBoost_originalresnet` |
| topk 0.5 | `CUDA_VISIBLE_DEVICES=4 python3 -u -m HBN.boost_train --dataset cifar100 --basemodel originalresnet --empty-stage-num 12 --stage-epochs 100,100,100,100,100,100,100,100,100,100,100,100,100 --head-mode hbn --stage0load --loss-mode stage --sample-weight-mode topk --sample-weight-topk-ratio 0.5` | `20260403_023506_cifar100_HBNBoost_originalresnet` |

## Summary

| Setting | Stage1 pretrained acc | Best stage | Best acc | Final stage | Final acc | Final loss | Final epsilon | Final alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | 76.17 | 3 | 76.20 | 13 | 75.89 | 1.0970 | 0.768921 | 0.100000 |
| topk-0.05 | 76.17 | 2 | 76.26 | 13 | 76.21 | 1.0625 | 0.498113 | 0.003773 |
| topk-0.1 | 76.17 | 2 | 76.25 | 13 | 76.18 | 1.0609 | 0.525989 | 0.100000 |
| topk-0.2 | 76.17 | 13 | 76.22 | 13 | 76.22 | 1.0580 | 0.499710 | 0.000580 |
| topk-0.5 | 76.17 | 8 | 76.27 | 13 | 76.24 | 1.0709 | 0.737328 | 0.100000 |

## Stage-wise Accuracy (Stages 2-13)

Here, the binary baseline refers to the original sample-weight update policy that only uses correctness: misclassified samples are upweighted and correctly classified samples are left unchanged. The stage-wise accuracy trend shows the clearest difference between this binary baseline and the top-k variants. Under the binary run, accuracy stays around 76.0 but does not improve over stages, while the loss keeps increasing, indicating that the training process becomes increasingly concentrated on a small set of hard samples without translating that concentration into higher stage-wise accuracy. In contrast, all top-k settings maintain slightly higher accuracy throughout the later stages, which suggests that expanding the reweighted set beyond strictly misclassified samples helps prevent the stage-wise process from becoming stagnant. Among the top-k settings, 0.05 and 0.5 reach the strongest late-stage accuracy, while 0.2 is the most stable around 76.2 across the later stages.

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 76.18 | 76.26 | 76.25 | 76.17 | 76.18 |
| 3 | 76.20 | 76.15 | 76.19 | 76.15 | 76.18 |
| 4 | 76.09 | 76.20 | 76.20 | 76.18 | 76.21 |
| 5 | 75.96 | 76.20 | 76.18 | 76.16 | 76.24 |
| 6 | 75.97 | 76.24 | 76.19 | 76.18 | 76.23 |
| 7 | 75.98 | 76.19 | 76.19 | 76.18 | 76.25 |
| 8 | 75.87 | 76.19 | 76.21 | 76.20 | 76.27 |
| 9 | 75.87 | 76.19 | 76.22 | 76.21 | 76.27 |
| 10 | 75.89 | 76.20 | 76.22 | 76.21 | 76.26 |
| 11 | 75.89 | 76.19 | 76.22 | 76.21 | 76.25 |
| 12 | 75.90 | 76.21 | 76.16 | 76.21 | 76.25 |
| 13 | 75.89 | 76.21 | 76.18 | 76.22 | 76.24 |

## Stage-wise Epsilon (Stages 2-13)

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.001813 | 0.061063 | 0.108046 | 0.213794 | 0.503269 |
| 3 | 0.010408 | 0.166179 | 0.232232 | 0.389760 | 0.523843 |
| 4 | 0.026574 | 0.302957 | 0.345156 | 0.465095 | 0.546735 |
| 5 | 0.061608 | 0.384864 | 0.419178 | 0.479539 | 0.568382 |
| 6 | 0.063855 | 0.451853 | 0.454852 | 0.489219 | 0.590819 |
| 7 | 0.157271 | 0.460860 | 0.475012 | 0.492546 | 0.613105 |
| 8 | 0.207004 | 0.496405 | 0.488045 | 0.497370 | 0.634175 |
| 9 | 0.633992 | 0.497331 | 0.493508 | 0.495766 | 0.657091 |
| 10 | 0.262875 | 0.489836 | 0.498926 | 0.499373 | 0.678218 |
| 11 | 0.464918 | 0.497706 | 0.493879 | 0.496075 | 0.698840 |
| 12 | 0.333855 | 0.490069 | 0.503631 | 0.500863 | 0.718159 |
| 13 | 0.768921 | 0.498113 | 0.525989 | 0.499710 | 0.737328 |

## Stage-wise Alpha (Stages 2-13)

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 3.000000 | 1.366422 | 1.055429 | 0.650833 | 0.100000 |
| 3 | 2.277352 | 0.806476 | 0.597876 | 0.223949 | 0.100000 |
| 4 | 1.800453 | 0.416629 | 0.320200 | 0.069870 | 0.100000 |
| 5 | 1.361692 | 0.234477 | 0.163074 | 0.040943 | 0.100000 |
| 6 | 1.342578 | 0.096594 | 0.090542 | 0.021566 | 0.100000 |
| 7 | 0.839339 | 0.078439 | 0.050017 | 0.014916 | 0.100000 |
| 8 | 0.671541 | 0.007190 | 0.023915 | 0.005261 | 0.100000 |
| 9 | 0.100000 | 0.005338 | 0.012984 | 0.008469 | 0.100000 |
| 10 | 0.515539 | 0.020331 | 0.002148 | 0.001255 | 0.100000 |
| 11 | 0.070280 | 0.004587 | 0.012243 | 0.007849 | 0.100000 |
| 12 | 0.345401 | 0.019864 | 0.100000 | 0.100000 | 0.100000 |
| 13 | 0.100000 | 0.003773 | 0.100000 | 0.000580 | 0.100000 |

## Stage-wise Loss (Stages 2-13)

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.0509 | 1.0502 | 1.0504 | 1.0506 | 1.0558 |
| 3 | 1.0589 | 1.0545 | 1.0527 | 1.0522 | 1.0558 |
| 4 | 1.0726 | 1.0589 | 1.0550 | 1.0538 | 1.0563 |
| 5 | 1.0818 | 1.0607 | 1.0564 | 1.0546 | 1.0573 |
| 6 | 1.0874 | 1.0615 | 1.0572 | 1.0553 | 1.0586 |
| 7 | 1.0901 | 1.0620 | 1.0578 | 1.0558 | 1.0601 |
| 8 | 1.0931 | 1.0621 | 1.0581 | 1.0560 | 1.0618 |
| 9 | 1.0934 | 1.0621 | 1.0582 | 1.0563 | 1.0635 |
| 10 | 1.0952 | 1.0623 | 1.0582 | 1.0568 | 1.0654 |
| 11 | 1.0955 | 1.0623 | 1.0583 | 1.0570 | 1.0673 |
| 12 | 1.0963 | 1.0625 | 1.0595 | 1.0577 | 1.0691 |
| 13 | 1.0970 | 1.0625 | 1.0609 | 1.0580 | 1.0709 |

## Stage-wise Weight Distribution (Stages 2-13)

The weight-distribution statistics make the difference in reweighting behavior more explicit. Under the original binary baseline, the distribution rapidly becomes extremely sharp: even the p99 weight remains equal to the minimum weight throughout all later stages, while the maximum weight keeps growing from `0.0078` to `0.4170`. This means that only a tiny set of samples is repeatedly amplified. In contrast, the top-k variants spread the extra weight over a broader set of examples. Their p90 and p99 values separate from the minimum much earlier, and the maximum weight stays much smaller than in the binary run, which is consistent with the goal of expanding the reweighted sample set beyond strictly misclassified cases.

### Weight Min

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.94566e-05 | 1.68447e-05 | 1.65177e-05 | 1.65705e-05 | 1.88499e-05 |
| 3 | 1.76143e-05 | 1.38891e-05 | 1.38810e-05 | 1.44618e-05 | 1.78636e-05 |
| 4 | 1.55821e-05 | 1.20059e-05 | 1.22812e-05 | 1.32707e-05 | 1.68917e-05 |
| 5 | 1.14613e-05 | 1.08874e-05 | 1.14341e-05 | 1.26261e-05 | 1.59376e-05 |
| 6 | 7.67282e-06 | 1.04175e-05 | 1.09607e-05 | 1.23125e-05 | 1.50048e-05 |
| 7 | 5.63182e-06 | 1.00314e-05 | 1.06999e-05 | 1.21509e-05 | 1.40955e-05 |
| 8 | 3.72597e-06 | 9.99605e-06 | 1.05739e-05 | 1.20544e-05 | 1.32127e-05 |
| 9 | 3.50966e-06 | 9.97013e-06 | 1.05063e-05 | 1.20043e-05 | 1.23582e-05 |
| 10 | 2.42831e-06 | 9.86964e-06 | 1.04950e-05 | 1.19948e-05 | 1.15357e-05 |
| 11 | 2.29717e-06 | 9.84695e-06 | 1.04310e-05 | 1.19647e-05 | 1.07456e-05 |
| 12 | 1.79674e-06 | 9.74935e-06 | 9.90675e-06 | 1.19579e-05 | 9.99046e-06 |
| 13 | 1.67632e-06 | 9.73080e-06 | 9.38634e-06 | 1.19545e-05 | 9.27098e-06 |

### Weight P50

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.94566e-05 | 1.68447e-05 | 1.65177e-05 | 1.65705e-05 | 2.08324e-05 |
| 3 | 1.76143e-05 | 1.38891e-05 | 1.38810e-05 | 1.44618e-05 | 1.97424e-05 |
| 4 | 1.55821e-05 | 1.20059e-05 | 1.22812e-05 | 1.32707e-05 | 1.86682e-05 |
| 5 | 1.14613e-05 | 1.08874e-05 | 1.14341e-05 | 1.26261e-05 | 1.94662e-05 |
| 6 | 7.67282e-06 | 1.04175e-05 | 1.09607e-05 | 1.23125e-05 | 1.83270e-05 |
| 7 | 5.63182e-06 | 1.00314e-05 | 1.06999e-05 | 1.21509e-05 | 1.72163e-05 |
| 8 | 3.72597e-06 | 9.99605e-06 | 1.05739e-05 | 1.20544e-05 | 1.78352e-05 |
| 9 | 3.50966e-06 | 9.97013e-06 | 1.05063e-05 | 1.20043e-05 | 1.66819e-05 |
| 10 | 2.42831e-06 | 9.86964e-06 | 1.04950e-05 | 1.19948e-05 | 1.55715e-05 |
| 11 | 2.29717e-06 | 9.84695e-06 | 1.04310e-05 | 1.19647e-05 | 1.60306e-05 |
| 12 | 1.79674e-06 | 9.74935e-06 | 9.90675e-06 | 1.19579e-05 | 1.49040e-05 |
| 13 | 1.67632e-06 | 9.73080e-06 | 9.38634e-06 | 1.19545e-05 | 1.38307e-05 |

### Weight P90

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.94566e-05 | 1.68447e-05 | 4.74589e-05 | 3.22955e-05 | 2.08324e-05 |
| 3 | 1.76143e-05 | 1.38891e-05 | 3.98831e-05 | 4.10670e-05 | 2.18187e-05 |
| 4 | 1.55821e-05 | 1.20059e-05 | 3.52864e-05 | 4.61314e-05 | 2.28014e-05 |
| 5 | 1.14613e-05 | 1.08874e-05 | 3.28526e-05 | 4.89085e-05 | 2.37760e-05 |
| 6 | 7.67282e-06 | 1.04175e-05 | 3.14924e-05 | 5.02617e-05 | 2.47388e-05 |
| 7 | 5.63182e-06 | 1.00314e-05 | 3.07431e-05 | 5.09604e-05 | 2.56836e-05 |
| 8 | 3.72597e-06 | 9.99605e-06 | 3.03809e-05 | 5.13777e-05 | 2.66070e-05 |
| 9 | 3.50966e-06 | 9.97013e-06 | 3.01866e-05 | 5.15945e-05 | 2.75038e-05 |
| 10 | 2.42831e-06 | 9.86964e-06 | 3.01544e-05 | 5.16354e-05 | 2.83732e-05 |
| 11 | 2.29717e-06 | 9.84695e-06 | 2.99704e-05 | 5.17658e-05 | 2.92097e-05 |
| 12 | 1.79674e-06 | 9.74935e-06 | 2.84642e-05 | 5.17951e-05 | 3.00130e-05 |
| 13 | 1.67632e-06 | 9.73080e-06 | 2.69689e-05 | 5.18101e-05 | 3.07807e-05 |

### Weight P99

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.94566e-05 | 6.60528e-05 | 4.74589e-05 | 3.22955e-05 | 2.08324e-05 |
| 3 | 1.76143e-05 | 1.21997e-04 | 7.25175e-05 | 4.10670e-05 | 2.18187e-05 |
| 4 | 1.55821e-05 | 1.59960e-04 | 8.83736e-05 | 4.61314e-05 | 2.28014e-05 |
| 5 | 1.14613e-05 | 1.83390e-04 | 9.68517e-05 | 4.89085e-05 | 2.37760e-05 |
| 6 | 7.67282e-06 | 1.93270e-04 | 1.01640e-04 | 5.02617e-05 | 2.47388e-05 |
| 7 | 5.63182e-06 | 2.01291e-04 | 1.04311e-04 | 5.09604e-05 | 2.56836e-05 |
| 8 | 3.72597e-06 | 2.02030e-04 | 1.05577e-04 | 5.13777e-05 | 2.66070e-05 |
| 9 | 3.50966e-06 | 2.02585e-04 | 1.06273e-04 | 5.15945e-05 | 2.75038e-05 |
| 10 | 2.42831e-06 | 2.04662e-04 | 1.06387e-04 | 5.16354e-05 | 2.83732e-05 |
| 11 | 2.29717e-06 | 2.05130e-04 | 1.07041e-04 | 5.17658e-05 | 2.92097e-05 |
| 12 | 1.79674e-06 | 2.07171e-04 | 1.12353e-04 | 5.17951e-05 | 3.00130e-05 |
| 13 | 1.67632e-06 | 2.07559e-04 | 1.17646e-04 | 5.18101e-05 | 3.07807e-05 |

### Weight Max

| Stage | binary | topk-0.05 | topk-0.1 | topk-0.2 | topk-0.5 |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.00784934 | 0.00132671 | 0.000953237 | 0.000648672 | 0.00041843 |
| 3 | 0.0692906 | 0.00245038 | 0.00145655 | 0.000824853 | 0.00043824 |
| 4 | 0.0612964 | 0.00321288 | 0.00177503 | 0.000926575 | 0.000457979 |
| 5 | 0.175962 | 0.00368348 | 0.00194532 | 0.000982353 | 0.000477555 |
| 6 | 0.135911 | 0.00388193 | 0.00204149 | 0.00100953 | 0.000496892 |
| 7 | 0.230925 | 0.00404305 | 0.00209513 | 0.00102357 | 0.000515869 |
| 8 | 0.299025 | 0.00405788 | 0.00212056 | 0.00103195 | 0.000534417 |
| 9 | 0.311288 | 0.00406902 | 0.00213454 | 0.00103630 | 0.000552428 |
| 10 | 0.360659 | 0.00411074 | 0.00213685 | 0.00103713 | 0.000569891 |
| 11 | 0.366024 | 0.00412015 | 0.00214997 | 0.00103974 | 0.000586692 |
| 12 | 0.404395 | 0.00416115 | 0.00225667 | 0.00104033 | 0.000602827 |
| 13 | 0.416974 | 0.00416893 | 0.00236299 | 0.00104063 | 0.000618247 |

## Interpretation

These comparisons suggest that the sample-weight distribution itself is a critical factor in later-stage learning. When the weight distribution becomes too extreme, optimization is dominated by a very small subset of samples, and this concentration does not translate into better stage-wise accuracy. The binary baseline illustrates this clearly: the maximum weight keeps increasing sharply while most of the distribution remains collapsed near the minimum. In contrast, the top-k variants spread the reweighting effect over a broader subset of samples and maintain a smoother distribution, which is more compatible with stable later-stage learning. At the same time, the results also indicate that obtaining a smooth weight distribution is not a solved problem yet. Different top-k ratios lead to different trade-offs between stability and final accuracy, so the key open question is not simply whether to smooth the weights, but how to design a sample-reweighting strategy that produces a sufficiently smooth distribution without weakening the focus on genuinely hard examples.
