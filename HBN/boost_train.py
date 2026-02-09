import argparse
import csv
import datetime
import os
import time

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from utils import progress_bar

from .merger import HBNMerger
from .toolkit import SplitModule
from .boosting_core import SampleWeights, compute_alpha_paper, set_trainable_for_stage, weighted_cross_entropy
from .boosting_eval import eval_candidate_sum, eval_ensemble
from .metrics_viz import auto_plot_run_dir


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx

def main():
    parser = argparse.ArgumentParser(description='HBN stage-wise boosting trainer')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--basemodel', default='resnet18', type=str)
    parser.add_argument('--empty-stage-num', default=None, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--stage-lrs', default='', type=str)
    parser.add_argument('--stage-epochs', default='50,50,50,50', type=str)
    parser.add_argument('--max-train-batches', default=0, type=int)
    parser.add_argument('--max-eval-batches', default=0, type=int)
    parser.add_argument('--eps-clip', default=1e-12, type=float)
    parser.add_argument('--alpha-clip', default=8.0, type=float)
    parser.add_argument('--alpha-mode', default='paper', choices=['paper'], type=str)
    parser.add_argument('--log-dir', default='./runs', type=str)
    parser.add_argument('--run-name', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(int(args.seed))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_name = args.dataset.lower()
    if dataset_name == 'cifar100':
        normalize_mean = (0.5071, 0.4867, 0.4408)
        normalize_std = (0.2675, 0.2565, 0.2761)
        dataset_cls = torchvision.datasets.CIFAR100
    else:
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
        dataset_cls = torchvision.datasets.CIFAR10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    trainset = dataset_cls(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = dataset_cls(root=args.data_dir, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        IndexedDataset(trainset), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = torch.utils.data.DataLoader(
        IndexedDataset(trainset), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    dataset_num_classes = 100 if dataset_name == 'cifar100' else 10
    cfg = SplitModule(args.basemodel, num_classes=dataset_num_classes, empty_stage_num=args.empty_stage_num).get_HBN_model_Config()
    model = HBNMerger(**cfg).to(device)
    num_stages = len(cfg['modules'])
    num_classes = int(cfg['num_classes'])

    try:
        stage_epochs = [int(x.strip()) for x in args.stage_epochs.split(',') if x.strip()]
    except Exception:
        raise ValueError('Invalid --stage-epochs: {}'.format(args.stage_epochs))
    if len(stage_epochs) != num_stages:
        raise ValueError('Expected {} values for --stage-epochs, got {}'.format(num_stages, len(stage_epochs)))

    stage_lrs = None
    if args.stage_lrs and args.stage_lrs.strip():
        try:
            stage_lrs = [float(x.strip()) for x in args.stage_lrs.split(',') if x.strip()]
        except Exception:
            raise ValueError('Invalid --stage-lrs: {}'.format(args.stage_lrs))
        if len(stage_lrs) == 1 and num_stages > 1:
            stage_lrs = stage_lrs * num_stages
        if len(stage_lrs) != num_stages:
            raise ValueError('Expected {} values for --stage-lrs, got {}'.format(num_stages, len(stage_lrs)))
    else:
        stage_lrs = [float(args.lr)] * num_stages

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or '{}_{}_HBNBoost_{}'.format(ts, dataset_name, args.basemodel)
    run_dir = os.path.join(args.log_dir, run_name) if args.log_dir else None
    if run_dir is not None and not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    metrics_writer = None
    metrics_fp = None
    if run_dir is not None:
        metrics_fp = open(os.path.join(run_dir, 'metrics.csv'), 'w', newline='')
        metrics_writer = csv.writer(metrics_fp)
        metrics_writer.writerow(['stage', 'epoch', 'split', 'loss', 'acc', 'lr', 'epsilon', 'alpha'])

    def log_row(stage, epoch, split, loss, acc, lr, epsilon=None, alpha=None):
        if metrics_writer is None:
            return
        metrics_writer.writerow([stage, epoch, split, loss, acc, lr, epsilon, alpha])

    sample_w = SampleWeights(len(trainset))
    alphas = []
    if args.max_eval_batches:
        print('warning: max-eval-batches is set; alpha/epsilon will be approximate')

    print('HBN config | stages: {} | stage_epochs: {} | stage_lrs: {}'.format(num_stages, stage_epochs, stage_lrs))

    for stage_idx in range(1, num_stages + 1):
        set_trainable_for_stage(model, stage_idx)
        stage_params = [p for p in model.parameters() if p.requires_grad]
        stage_lr = float(stage_lrs[stage_idx - 1])
        stage_opt = optim.SGD(stage_params, lr=stage_lr, momentum=0.9, weight_decay=5e-4)
        stage_sched = torch.optim.lr_scheduler.CosineAnnealingLR(stage_opt, T_max=stage_epochs[stage_idx - 1])
        print('Stage {} | lr {:.6g} | epochs {}'.format(stage_idx, stage_lr, stage_epochs[stage_idx - 1]))

        if hasattr(model.head_list[stage_idx - 1], 'classifyheadweight'):
            with torch.no_grad():
                model.head_list[stage_idx - 1].classifyheadweight.fill_(0.0)

        for ep in range(stage_epochs[stage_idx - 1]):
            model.eval()
            stage_i = int(stage_idx) - 1
            model.modules_list[stage_i].train()
            if stage_i < len(model.adapter_list):
                model.adapter_list[stage_i].train()
            model.head_list[stage_i].train()
            t0 = time.time()
            loss_sum = 0.0
            correct = 0
            total = 0
            train_total_batches = len(trainloader)
            if args.max_train_batches and args.max_train_batches > 0:
                train_total_batches = min(train_total_batches, int(args.max_train_batches))
            for batch_idx, (inputs, targets, indices) in enumerate(trainloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                w_batch = sample_w.batch_weights(indices, device=device, dtype=torch.float32)

                stage_opt.zero_grad()

                if stage_idx > 1:
                    with torch.no_grad():
                        logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
                else:
                    logits_prev = torch.zeros((inputs.size(0), num_classes), device=device, dtype=torch.float32)

                logits_t = model.forward_stage_logits(inputs, stage_idx=stage_idx)
                loss = weighted_cross_entropy(logits_prev.detach() + logits_t, targets, w_batch) + weighted_cross_entropy(logits_t, targets, w_batch)
                #loss = weighted_cross_entropy(logits_t, targets, w_batch)
               
                loss.backward()
                stage_opt.step()

                logits_sum = logits_prev.detach() + logits_t.detach()
                preds = torch.argmax(logits_sum, dim=1)
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
                loss_sum += float(loss.item())

                progress_bar(batch_idx, train_total_batches, 'Stage {} | Loss {:.3f} | Acc {:.2f}%'.format(stage_idx, loss_sum / (batch_idx + 1), 100.0 * correct / total))
                if (batch_idx + 1) >= train_total_batches:
                    break

            stage_sched.step()
            lr = stage_opt.param_groups[0]['lr']
            train_loss = loss_sum / (batch_idx + 1)
            train_acc = 100.0 * correct / total if total else 0.0
            log_row(stage_idx, ep, 'train', train_loss, train_acc, lr)

            test_cand_loss, test_cand_acc = eval_candidate_sum(model, testloader, stage_idx=stage_idx, alpha=1.0, num_classes=num_classes, max_batches=args.max_eval_batches, device=device)
            if stage_idx > 1:
                test_prev_loss, test_prev_acc = eval_ensemble(model, testloader, upto_stage=stage_idx - 1, max_batches=args.max_eval_batches, device=device)
            else:
                test_prev_loss, test_prev_acc = test_cand_loss, test_cand_acc
            log_row(stage_idx, ep, 'test_prev', test_prev_loss, test_prev_acc, lr)
            log_row(stage_idx, ep, 'test_cand', test_cand_loss, test_cand_acc, lr)
            print('Stage {} | Epoch {}/{} | train acc {:.2f}% | test prev {:.2f}% | test cand {:.2f}%'.format(stage_idx, ep + 1, stage_epochs[stage_idx - 1], train_acc, test_prev_acc, test_cand_acc))
            _ = time.time() - t0

        model.eval()
        errors01_unweighted = torch.zeros(len(trainset), dtype=torch.float32)
        with torch.no_grad():
            for batch_idx, (inputs, targets, indices) in enumerate(train_eval_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                if stage_idx > 1:
                    logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
                else:
                    logits_prev = torch.zeros((inputs.size(0), num_classes), device=device, dtype=torch.float32)
                logits_t = model.forward_stage_logits(inputs, stage_idx=stage_idx)
                logits_sum_unweighted = logits_prev + logits_t
                preds = torch.argmax(logits_sum_unweighted, dim=1)
                wrong = preds.ne(targets).to(torch.float32).cpu()
                errors01_unweighted[indices] = wrong
                if args.max_eval_batches and (batch_idx + 1) >= args.max_eval_batches:
                    break

        epsilon = sample_w.weighted_error(errors01_unweighted)
        alpha, eps_used = compute_alpha_paper(epsilon, eps_clip=args.eps_clip, alpha_clip=args.alpha_clip)
        alpha_used = float(alpha)
        if alpha_used < 0:
            alpha_used = 0.1
        alphas.append(alpha_used)
        with torch.no_grad():
            model.head_list[stage_idx - 1].classifyheadweight.fill_(alpha_used)
        errors01_weighted = torch.zeros(len(trainset), dtype=torch.float32)
        with torch.no_grad():
            for batch_idx, (inputs, targets, indices) in enumerate(train_eval_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                if stage_idx > 1:
                    logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
                else:
                    logits_prev = torch.zeros((inputs.size(0), num_classes), device=device, dtype=torch.float32)
                logits_t = model.forward_stage_logits(inputs, stage_idx=stage_idx)
                logits_sum_weighted = logits_prev + logits_t * alpha_used
                preds = torch.argmax(logits_sum_weighted, dim=1)
                wrong = preds.ne(targets).to(torch.float32).cpu()
                errors01_weighted[indices] = wrong
                if args.max_eval_batches and (batch_idx + 1) >= args.max_eval_batches:
                    break
        sample_w.update(errors01_weighted, alpha_used, alpha_clip=args.alpha_clip)

        test_loss, test_acc = eval_ensemble(model, testloader, upto_stage=stage_idx, max_batches=args.max_eval_batches, device=device)
        log_row(stage_idx, stage_epochs[stage_idx - 1], 'stage_done', test_loss, test_acc, 0.0, eps_used, alpha_used)
        print('Stage {} done | epsilon {:.6f} | alpha {:.6f} | test acc {:.2f}%'.format(stage_idx, eps_used, alpha_used, test_acc))

        if run_dir is not None:
            torch.save(
                {
                    'net': model.state_dict(),
                    'stage': stage_idx,
                    'alphas': alphas,
                    'epsilon': eps_used,
                    'sample_log_w': sample_w.log_w.cpu(),
                },
                os.path.join(run_dir, 'hbn_stage{}_ckpt.pth'.format(stage_idx)),
            )

    if metrics_fp is not None:
        metrics_fp.close()
    if run_dir is not None and args.plot:
        auto_plot_run_dir(run_dir)


if __name__ == '__main__':
    main()
