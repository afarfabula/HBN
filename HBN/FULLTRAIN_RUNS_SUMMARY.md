# Fulltrain Experiment Summary

## Overview

This document summarizes completed fulltrain runs found under `runs/`. It separates plain fulltrain from `--backwordstage2` runs and reports the final metric together with the best observed evaluation accuracy in each run.

## All Completed Fulltrain Runs

| Family | Run | Schedule | Final split | Final acc | Final loss | Best eval split | Best eval acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 / fulltrain | `20260402_200914_cifar100_HBNBoost_resnet18` | `stage_epochs=10` | stage_done@stage5 epoch 10 | 58.69 | 1.6122 | test_cand@stage5 epoch 9 | 58.69 |
| resnet18 / fulltrain | `20260402_202216_cifar100_HBNBoost_resnet18` | `stage_epochs=10` | stage_done@stage5 epoch 10 | 56.92 | 1.7245 | test_cand@stage5 epoch 8 | 57.96 |
| resnet18 / fulltrain | `20260402_203650_cifar100_HBNBoost_resnet18` | `stage_epochs=10,10,10,10,100` | stage_done@stage5 epoch 100 | 61.35 | 2.2559 | test_cand@stage5 epoch 85 | 62.01 |
| resnet18 / fulltrain | `20260402_205508_cifar100_HBNBoost_resnet18` | `stage_epochs=10,10,30,30,50` | stage_done@stage5 epoch 50 | 60.53 | 2.2763 | test_prev@stage5 epoch 15 | 62.57 |
| resnet18 / backwordstage2 | `20260402_211846_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=None, phase2=None` | phase2_done@stage2 epoch 1 | 0.00 | 4.6637 | phase1_test@stage1 epoch 0 | 0.00 |
| resnet18 / backwordstage2 | `20260402_223608_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=None, phase2=None` | phase2_done@stage2 epoch 5 | 59.51 | 1.4638 | phase2_test@stage2 epoch 4 | 59.51 |
| resnet18 / backwordstage2 | `20260402_224111_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=None, phase2=None` | phase2_done@stage2 epoch 50 | 69.44 | 1.6339 | phase2_test@stage2 epoch 48 | 70.25 |
| resnet18 / backwordstage2 | `20260402_225828_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=1, phase2=1` | phase2_done@stage2 epoch 1 | 3.00 | 4.6136 | phase1_test@stage1 epoch 0 | 4.00 |
| resnet18 / backwordstage2 | `20260402_230331_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=100, phase2=20` | phase2_done@stage2 epoch 20 | 69.88 | 1.6261 | phase2_test@stage2 epoch 8 | 70.54 |
| originalresnet / backwordstage2 | `20260402_234825_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=1, phase2=1` | phase2_done@stage2 epoch 1 | 62.00 | 1.5160 | phase1_load@stage1 epoch 0 | 74.00 |
| resnet18 / fulltrain | `20260402_235121_cifar100_HBNBoost_resnet18` | `stage_epochs=1` | stage_done@stage5 epoch 1 | 1.00 | 4.6347 | test_cand@stage2 epoch 0 | 1.00 |
| originalresnet / backwordstage2 | `20260402_235544_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=1, phase2=1` | phase2_done@stage2 epoch 1 | 76.00 | 1.0721 | phase1_load@stage1 epoch 0 | 76.00 |
| originalresnet / backwordstage2 | `20260403_000310_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=8, phase2=20` | phase2_done@stage2 epoch 20 | 72.66 | 1.5082 | phase1_load@stage1 epoch 0 | 76.17 |
| originalresnet / backwordstage2 | `20260403_001622_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=1, phase2=1` | phase2_done@stage2 epoch 1 | 75.00 | 1.1189 | phase1_load@stage1 epoch 0 | 76.00 |
| originalresnet / backwordstage2 | `20260403_001728_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=8, phase2=20` | phase2_done@stage2 epoch 20 | 72.08 | 2.5026 | phase1_load@stage1 epoch 0 | 76.17 |
| originalresnet / backwordstage2 | `20260403_002541_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=8, phase2=200` | phase2_done@stage2 epoch 200 | 70.91 | 4.5306 | phase1_load@stage1 epoch 0 | 76.17 |
| originalresnet / backwordstage2 | `20260403_003202_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=8, phase2=200` | phase2_done@stage2 epoch 200 | 71.75 | 4.4255 | phase1_load@stage1 epoch 0 | 76.17 |
| resnet18 / fulltrain | `20260403_015319_cifar100_HBNBoost_resnet18` | `stage_epochs=10,10,10,10` | stage_done@stage4 epoch 10 | 51.89 | 1.8062 | test_prev@stage5 epoch 0 | 54.57 |
| originalresnet / fulltrain | `20260403_015945_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50` | stage_done@stage4 epoch 50 | 65.92 | 3.0967 | test_prev@stage1 epoch 47 | 75.80 |
| originalresnet / fulltrain | `20260403_025617_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,100,100,100` | stage_done@stage4 epoch 100 | 65.20 | 2.3595 | test_prev@stage1 epoch 47 | 75.11 |
| originalresnet / fulltrain | `fulltrain_freeze_smoke` | `stage_epochs=1,1,1,1` | stage_done@stage4 epoch 1 | 0.00 | 4.6238 | test_prev@stage2 epoch 0 | 5.00 |

## resnet18 / fulltrain

| Run | Key schedule | Final acc | Final loss | Best eval acc | Notes |
| --- | --- | --- | --- | --- | --- |
| `20260402_200914_cifar100_HBNBoost_resnet18` | `stage_epochs=10` | 58.69 | 1.6122 | 58.69 | final = best |
| `20260402_202216_cifar100_HBNBoost_resnet18` | `stage_epochs=10` | 56.92 | 1.7245 | 57.96 |  |
| `20260402_203650_cifar100_HBNBoost_resnet18` | `stage_epochs=10,10,10,10,100` | 61.35 | 2.2559 | 62.01 |  |
| `20260402_205508_cifar100_HBNBoost_resnet18` | `stage_epochs=10,10,30,30,50` | 60.53 | 2.2763 | 62.57 |  |
| `20260402_235121_cifar100_HBNBoost_resnet18` | `stage_epochs=1` | 1.00 | 4.6347 | 1.00 | final = best; short/smoke-like |
| `20260403_015319_cifar100_HBNBoost_resnet18` | `stage_epochs=10,10,10,10` | 51.89 | 1.8062 | 54.57 |  |

## resnet18 / backwordstage2

| Run | Key schedule | Final acc | Final loss | Best eval acc | Notes |
| --- | --- | --- | --- | --- | --- |
| `20260402_211846_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=None, phase2=None` | 0.00 | 4.6637 | 0.00 | final = best; short/smoke-like |
| `20260402_223608_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=None, phase2=None` | 59.51 | 1.4638 | 59.51 | final = best; short/smoke-like |
| `20260402_224111_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=None, phase2=None` | 69.44 | 1.6339 | 70.25 |  |
| `20260402_225828_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=1, phase2=1` | 3.00 | 4.6136 | 4.00 | short/smoke-like |
| `20260402_230331_cifar100_HBNBoost_resnet18` | `stage_epochs=50,50,50,50, phase1=100, phase2=20` | 69.88 | 1.6261 | 70.54 |  |

## originalresnet / fulltrain

| Run | Key schedule | Final acc | Final loss | Best eval acc | Notes |
| --- | --- | --- | --- | --- | --- |
| `20260403_015945_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50` | 65.92 | 3.0967 | 75.80 |  |
| `20260403_025617_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,100,100,100` | 65.20 | 2.3595 | 75.11 |  |
| `fulltrain_freeze_smoke` | `stage_epochs=1,1,1,1` | 0.00 | 4.6238 | 5.00 | short/smoke-like |

## originalresnet / backwordstage2

| Run | Key schedule | Final acc | Final loss | Best eval acc | Notes |
| --- | --- | --- | --- | --- | --- |
| `20260402_234825_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=1, phase2=1` | 62.00 | 1.5160 | 74.00 | short/smoke-like |
| `20260402_235544_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=1, phase2=1` | 76.00 | 1.0721 | 76.00 | final = best; short/smoke-like |
| `20260403_000310_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=8, phase2=20` | 72.66 | 1.5082 | 76.17 |  |
| `20260403_001622_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=1, phase2=1` | 75.00 | 1.1189 | 76.00 | short/smoke-like |
| `20260403_001728_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=8, phase2=20` | 72.08 | 2.5026 | 76.17 |  |
| `20260403_002541_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=8, phase2=200` | 70.91 | 4.5306 | 76.17 |  |
| `20260403_003202_cifar100_HBNBoost_originalresnet` | `stage_epochs=50,50,50,50, phase1=8, phase2=200` | 71.75 | 4.4255 | 76.17 |  |
