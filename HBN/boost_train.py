import argparse
import csv
import datetime
import json
import os
import shlex
import sys
import time

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from utils import progress_bar

from .merger import HBNMerger
from .toolkit import SplitModule
from .boosting_core import SampleWeights, compute_alpha_paper, set_trainable_for_backwordstage2_phase, set_trainable_for_fulltrain_stage, set_trainable_for_smhl_stage, set_trainable_for_stage, weighted_cross_entropy
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


class _TeeStream(object):
    def __init__(self, original_stream, file_stream):
        self.original_stream = original_stream
        self.file_stream = file_stream

    def write(self, data):
        self.original_stream.write(data)
        self.file_stream.write(data)
        self.file_stream.flush()
        return len(data)

    def flush(self):
        self.original_stream.flush()
        self.file_stream.flush()

    def isatty(self):
        return bool(getattr(self.original_stream, 'isatty', lambda: False)())

    @property
    def encoding(self):
        return getattr(self.original_stream, 'encoding', 'utf-8')

def main():
    parser = argparse.ArgumentParser(description='HBN stage-wise boosting trainer')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--basemodel', default='resnet18', type=str)
    parser.add_argument('--empty-stage-num', default=None, type=int)
    parser.add_argument('--head-mode', default='auto', choices=['auto', 'baseline', 'hbn'], type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--stage-lrs', default='', type=str)
    parser.add_argument('--stage-epochs', default='50,50,50,50', type=str)
    parser.add_argument('--max-train-batches', default=0, type=int)
    parser.add_argument('--max-eval-batches', default=0, type=int)
    parser.add_argument('--eps-clip', default=1e-12, type=float)
    parser.add_argument('--alpha-clip', default=3.0, type=float)
    parser.add_argument('--alpha-mode', default='paper', choices=['paper'], type=str)
    parser.add_argument('--alpha-error-mode', default='stage', choices=['stage', 'sum'], type=str)
    parser.add_argument('--loss-mode', default='sum', choices=['stage', 'sum'], type=str)
    parser.add_argument('--normalize-head-weight', dest='normalize_head_weight', action='store_true')
    parser.add_argument('--no-normalize-head-weight', dest='normalize_head_weight', action='store_false')
    parser.add_argument('--backbone-update', dest='backbone_update', action='store_true')
    parser.add_argument('--backbone_update', dest='backbone_update', action='store_true')
    parser.add_argument('--double-adapter', dest='double_adapter', action='store_true')
    parser.add_argument('--no-double-adapter', dest='double_adapter', action='store_false')
    parser.add_argument('--unit-head-weight', dest='unit_head_weight', action='store_true')
    parser.add_argument('--unit_head_weight', dest='unit_head_weight', action='store_true')
    parser.add_argument('--prev-logit-ce', dest='prev_logit_ce', action='store_true')
    parser.add_argument('--prev_logit_ce', dest='prev_logit_ce', action='store_true')
    parser.add_argument('--prev-logit-ce-weight', default=0.0, type=float)
    parser.add_argument('--prev_logit_ce_weight', default=0.0, type=float)
    parser.add_argument('--use-pretrain', dest='use_pretrain', action='store_true')
    parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true')
    parser.add_argument('--pretrain-stage1-path', default='', type=str)
    parser.add_argument('--pretrain_stage1_path', default='', type=str)
    parser.add_argument('--stage0load', dest='stage0load', action='store_true')
    parser.add_argument('--stage0load-path', default='', type=str)
    parser.add_argument('--fulltrain', dest='fulltrain', action='store_true')
    parser.add_argument('--full_train', dest='fulltrain', action='store_true')
    parser.add_argument('--backwordstage2', dest='backwordstage2', action='store_true')
    parser.add_argument('--backwordstage2-epochs', default=10, type=int)
    parser.add_argument('--backwordstage2-phase1-epochs', default=0, type=int)
    parser.add_argument('--backwordstage2-phase2-epochs', default=0, type=int)
    parser.add_argument('--backwordstage2-lr', default=1e-4, type=float)
    parser.add_argument('--backwordstage2-per-head-loss', dest='backwordstage2_per_head_loss', action='store_true')
    parser.add_argument('--no-backwordstage2-per-head-loss', dest='backwordstage2_per_head_loss', action='store_false')
    parser.add_argument('--smhl-disable-sample-weight', dest='smhl_sample_weight', action='store_false')
    parser.add_argument('--smhl-enable-sample-weight', dest='smhl_sample_weight', action='store_true')
    parser.add_argument('--smhl-disable-head-weight', dest='smhl_head_weight', action='store_false')
    parser.add_argument('--smhl-enable-head-weight', dest='smhl_head_weight', action='store_true')
    parser.add_argument('--sample-weight-mode', default='binary', choices=['binary', 'topk', 'confall'], type=str)
    parser.add_argument('--sample-weight-topk-ratio', default=0.05, type=float)
    parser.add_argument('--sample-weight-gamma', default=2.0, type=float)
    parser.add_argument('--log-dir', default='./runs', type=str)
    parser.add_argument('--run-name', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--progress-bar', dest='progress_bar', action='store_true')
    parser.add_argument('--no-progress-bar', dest='progress_bar', action='store_false')
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True, normalize_head_weight=True, progress_bar=False, smhl_sample_weight=False, smhl_head_weight=False, backwordstage2_per_head_loss=False, double_adapter=False)
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(int(args.seed))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.backwordstage2 and (not args.fulltrain):
        raise ValueError('--backwordstage2 requires --fulltrain')
    if args.backwordstage2 and str(args.basemodel).lower() not in ("originalresnet", "original_resnet", "resnet18original", "resnet18_original", "resnet18-original"):
        raise ValueError('--backwordstage2 currently only supports --basemodel originalresnet')
    if args.sample_weight_topk_ratio < 0.0 or args.sample_weight_topk_ratio > 1.0:
        raise ValueError('--sample-weight-topk-ratio must be in [0, 1]')
    if args.sample_weight_gamma <= 0.0:
        raise ValueError('--sample-weight-gamma must be positive')

    if args.stage0load:
        args.use_pretrain = True
        if args.stage0load_path and args.stage0load_path.strip():
            args.pretrain_stage1_path = args.stage0load_path.strip()
        else:
            bn = str(args.basemodel).lower()
            if bn in ("originalresnet", "original_resnet", "resnet18original", "resnet18_original", "resnet18-original"):
                args.pretrain_stage1_path = "/mlx_devbox/users/quyanyi/playground/pytorch-cifar/runs/20260402_155143_cifar100_HBNBoost_originalresnet/hbn_stage1_ckpt.pth"

    use_backwordstage2 = bool(args.backwordstage2)
    use_smhl = bool(args.fulltrain and str(args.basemodel).lower() == 'resnet18' and (not use_backwordstage2))
    if use_smhl:
        if float(args.lr) == 0.1 and (not (args.stage_lrs and str(args.stage_lrs).strip())):
            args.lr = 1e-4
        if int(args.batch_size) == 128:
            args.batch_size = 64
        if str(args.stage_epochs).strip() == '50,50,50,50':
            args.stage_epochs = '10'

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

    if str(args.head_mode).lower() == 'hbn' and str(args.alpha_error_mode).lower() == 'stage':
        args.alpha_error_mode = 'sum'

    dataset_num_classes = 100 if dataset_name == 'cifar100' else 10
    basemodel_name = args.basemodel
    adapter_multiplier = 2.0 if (args.double_adapter and (not args.fulltrain)) else 1.0
    if use_smhl:
        basemodel_name = 'resnet18_smhl'
        args.head_mode = 'baseline'
        args.normalize_head_weight = False
    elif use_backwordstage2:
        basemodel_name = 'originalresnet_split5'
        args.head_mode = 'hbn'
        args.normalize_head_weight = False
    cfg = SplitModule(
        basemodel_name,
        num_classes=dataset_num_classes,
        empty_stage_num=args.empty_stage_num,
        head_mode=args.head_mode,
        adapter_multiplier=adapter_multiplier,
    ).get_HBN_model_Config()
    model = HBNMerger(**cfg).to(device)
    model.force_unit_head_weight = bool(args.unit_head_weight) or use_backwordstage2 or (use_smhl and (not args.smhl_head_weight))
    num_stages = len(cfg['modules'])
    num_classes = int(cfg['num_classes'])

    try:
        stage_epochs = [int(x.strip()) for x in args.stage_epochs.split(',') if x.strip()]
    except Exception:
        raise ValueError('Invalid --stage-epochs: {}'.format(args.stage_epochs))
    if use_smhl:
        if len(stage_epochs) == 4 and num_stages == 5:
            stage_epochs = [stage_epochs[0]] + stage_epochs
        elif len(stage_epochs) == 1 and num_stages > 1:
            stage_epochs = stage_epochs * num_stages
    if (not use_backwordstage2) and len(stage_epochs) != num_stages:
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
    stdout_fp = None
    stderr_fp = None
    if run_dir is not None:
        with open(os.path.join(run_dir, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(vars(args)), f, ensure_ascii=False, indent=2, sort_keys=True)
        exe = sys.executable or 'python3'
        cmd = [exe] + sys.argv
        with open(os.path.join(run_dir, 'command.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(shlex.quote(x) for x in cmd) + '\n')
        stdout_fp = open(os.path.join(run_dir, 'stdout.log'), 'a', encoding='utf-8', buffering=1)
        stderr_fp = open(os.path.join(run_dir, 'stderr.log'), 'a', encoding='utf-8', buffering=1)
        sys.stdout = _TeeStream(sys.stdout, stdout_fp)
        sys.stderr = _TeeStream(sys.stderr, stderr_fp)

    metrics_writer = None
    metrics_fp = None
    if run_dir is not None:
        metrics_fp = open(os.path.join(run_dir, 'metrics.csv'), 'w', newline='')
        metrics_writer = csv.writer(metrics_fp)
        metrics_writer.writerow(['stage', 'epoch', 'split', 'loss', 'acc', 'lr', 'epsilon', 'alpha'])
        metrics_fp.flush()
        os.fsync(metrics_fp.fileno())

    def log_row(stage, epoch, split, loss, acc, lr, epsilon=None, alpha=None):
        if metrics_writer is None:
            return
        metrics_writer.writerow([stage, epoch, split, loss, acc, lr, epsilon, alpha])
        if metrics_fp is not None:
            metrics_fp.flush()
            os.fsync(metrics_fp.fileno())

    def _format_duration(seconds):
        s = max(0, int(round(float(seconds))))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h > 0:
            return '{:02d}:{:02d}:{:02d}'.format(h, m, sec)
        return '{:02d}:{:02d}'.format(m, sec)

    def _compute_difficulty_scores_from_logits(logits, targets):
        mode = str(args.sample_weight_mode).lower()
        preds = torch.argmax(logits, dim=1)
        errors01 = preds.ne(targets).to(torch.float32)
        if mode == 'binary':
            return errors01
        probs = torch.softmax(logits, dim=1)
        true_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        if mode == 'confall':
            gamma = float(args.sample_weight_gamma)
            return torch.pow(torch.clamp(1.0 - true_probs, min=0.0, max=1.0), gamma).to(torch.float32)
        if mode == 'topk':
            difficulty = errors01.clone()
            correct_mask = preds.eq(targets)
            num_correct = int(correct_mask.sum().item())
            ratio = float(args.sample_weight_topk_ratio)
            if num_correct > 0 and ratio > 0.0:
                num_pick = max(1, int(num_correct * ratio + 0.999999))
                num_pick = min(num_pick, num_correct)
                correct_indices = torch.nonzero(correct_mask, as_tuple=False).squeeze(1)
                correct_true_probs = true_probs[correct_indices]
                _, order = torch.topk(correct_true_probs, k=num_pick, largest=False)
                chosen = correct_indices[order]
                difficulty[chosen] = 1.0
            return difficulty.to(torch.float32)
        raise ValueError('Unsupported sample_weight_mode: {}'.format(args.sample_weight_mode))

    def _weight_summary_string():
        w = torch.exp(sample_w.log_w).detach().cpu().to(torch.float32)
        if w.numel() == 0:
            return 'weight q|min 0.000000 | p50 0.000000 | p90 0.000000 | p99 0.000000 | max 0.000000'
        qs = torch.quantile(w, torch.tensor([0.5, 0.9, 0.99], dtype=w.dtype))
        return 'weight q|min {:.6g} | p50 {:.6g} | p90 {:.6g} | p99 {:.6g} | max {:.6g}'.format(
            float(torch.min(w).item()),
            float(qs[0].item()),
            float(qs[1].item()),
            float(qs[2].item()),
            float(torch.max(w).item()),
        )

    sample_w = SampleWeights(len(trainset))
    alphas = []
    if args.max_eval_batches:
        print('warning: max-eval-batches is set; alpha/epsilon will be approximate')

    print(
        'HBN config | stages: {} | stage_epochs: {} | stage_lrs: {} | head_mode: {} | alpha_error_mode: {} | loss_mode: {} | normalize_head_weight: {} | alpha_clip: {} | adapter_multiplier: {}'.format(
            num_stages,
            stage_epochs,
            stage_lrs,
            args.head_mode,
            args.alpha_error_mode,
            args.loss_mode,
            args.normalize_head_weight,
            args.alpha_clip,
            adapter_multiplier,
        )
    )
    if str(args.sample_weight_mode).lower() != 'binary':
        print(
            'Sample weight config | mode: {} | topk_ratio: {} | gamma: {}'.format(
                args.sample_weight_mode,
                args.sample_weight_topk_ratio,
                args.sample_weight_gamma,
            )
        )

    def _resolve_stage1_pretrain_path():
        if args.stage0load and args.stage0load_path and args.stage0load_path.strip():
            return args.stage0load_path.strip()
        if args.pretrain_stage1_path and args.pretrain_stage1_path.strip():
            return args.pretrain_stage1_path.strip()
        bn = str(args.basemodel).lower()
        pretrain_root = os.path.join(os.path.dirname(__file__), "pretrain", bn)
        candidates = [
            os.path.join(pretrain_root, "hbn_stage1_ckpt.pth"),
        ]
        if bn in ("originalresnet", "original_resnet", "resnet18original", "resnet18_original", "resnet18-original"):
            candidates.append(
                "/mlx_devbox/users/quyanyi/playground/pytorch-cifar/runs/20260402_155143_cifar100_HBNBoost_originalresnet/hbn_stage1_ckpt.pth"
            )
            candidates.append(
                "/mlx_devbox/users/quyanyi/playground/pytorch-cifar/runs/20260213_104246_cifar100_HBNBoost_originalresnet/hbn_stage1_ckpt.pth"
            )
        for p in candidates:
            if p and os.path.isfile(p):
                return p
        return candidates[0]

    if use_backwordstage2:
        phase1_epochs = int(args.backwordstage2_phase1_epochs)
        phase2_epochs = int(args.backwordstage2_phase2_epochs)
        if phase1_epochs > 0 or phase2_epochs > 0:
            if phase1_epochs <= 0 or phase2_epochs <= 0:
                raise ValueError('--backwordstage2-phase1-epochs and --backwordstage2-phase2-epochs must both be positive when either is set')
        else:
            total_phase_epochs = int(args.backwordstage2_epochs)
            if total_phase_epochs < 2:
                raise ValueError('--backwordstage2-epochs must be at least 2')
            phase1_epochs = total_phase_epochs // 2
            phase2_epochs = total_phase_epochs - phase1_epochs
        phase_lr = float(args.backwordstage2_lr)

        def _eval_last_head():
            model.eval()
            correct = 0
            total = 0
            loss_sum = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(testloader):
                    if len(batch) == 3:
                        inputs, targets, _ = batch
                    else:
                        inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    logits = model.forward_stage_logits(inputs, stage_idx=num_stages)
                    loss = F.cross_entropy(logits, targets)
                    loss_sum += float(loss.item())
                    preds = torch.argmax(logits, dim=1)
                    total += targets.size(0)
                    correct += preds.eq(targets).sum().item()
                    if args.max_eval_batches and (batch_idx + 1) >= args.max_eval_batches:
                        break
            avg_loss = loss_sum / (batch_idx + 1)
            acc = 100.0 * correct / total if total else 0.0
            return avg_loss, acc

        def _eval_all_heads():
            model.eval()
            head_correct = [0 for _ in range(num_stages)]
            total = 0
            head_loss_sum = [0.0 for _ in range(num_stages)]
            merged_correct = 0
            merged_loss_sum = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(testloader):
                    if len(batch) == 3:
                        inputs, targets, _ = batch
                    else:
                        inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    stage_logits = [model.forward_stage_logits(inputs, stage_idx=i + 1) for i in range(num_stages)]
                    merged_logits = None
                    for i, logits in enumerate(stage_logits):
                        head_loss_sum[i] += float(F.cross_entropy(logits, targets).item())
                        preds = torch.argmax(logits, dim=1)
                        head_correct[i] += preds.eq(targets).sum().item()
                        merged_logits = logits if merged_logits is None else merged_logits + logits
                    merged_loss_sum += float(F.cross_entropy(merged_logits, targets).item())
                    merged_preds = torch.argmax(merged_logits, dim=1)
                    merged_correct += merged_preds.eq(targets).sum().item()
                    total += targets.size(0)
                    if args.max_eval_batches and (batch_idx + 1) >= args.max_eval_batches:
                        break
            batches = batch_idx + 1
            head_accs = [100.0 * x / total if total else 0.0 for x in head_correct]
            head_losses = [x / batches for x in head_loss_sum]
            merged_acc = 100.0 * merged_correct / total if total else 0.0
            merged_loss = merged_loss_sum / batches
            return head_losses, head_accs, merged_loss, merged_acc

        def _format_head_accs(head_accs):
            return ' | '.join('h{} {:.2f}%'.format(i + 1, acc) for i, acc in enumerate(head_accs))

        def _load_original_stage1_ckpt_into_split_model(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
            mapped = {}
            for k, v in state.items():
                if k.startswith('modules_list.0.conv1.') or k.startswith('modules_list.0.bn1.'):
                    mapped[k] = v
                elif k.startswith('modules_list.0.layer1.'):
                    mapped['modules_list.1.' + k[len('modules_list.0.layer1.'):]] = v
                elif k.startswith('modules_list.0.layer2.'):
                    mapped['modules_list.2.' + k[len('modules_list.0.layer2.'):]] = v
                elif k.startswith('modules_list.0.layer3.'):
                    mapped['modules_list.3.' + k[len('modules_list.0.layer3.'):]] = v
                elif k.startswith('modules_list.0.layer4.'):
                    mapped['modules_list.4.0.' + k[len('modules_list.0.layer4.'):]] = v
                elif k.startswith('adapter_list.0.'):
                    mapped['modules_list.4.1.' + k[len('adapter_list.0.'):]] = v
                elif k.startswith('head_list.0.'):
                    mapped['head_list.4.' + k[len('head_list.0.'):]] = v
            missing, unexpected = model.load_state_dict(mapped, strict=False)
            if missing or unexpected:
                print('warning: backwordstage2 phase1 load missing_keys={} unexpected_keys={}'.format(len(missing), len(unexpected)))
            return ckpt

        print(
            'Backwordstage2 config | phase1_epochs: {} | phase2_epochs: {} | lr: {} | batch_size: {} | per_head_loss: {}'.format(
                phase1_epochs,
                phase2_epochs,
                phase_lr,
                args.batch_size,
                args.backwordstage2_per_head_loss,
            )
        )

        phase1_loaded = False
        phase1_ckpt_path = _resolve_stage1_pretrain_path()
        if phase1_ckpt_path and os.path.isfile(phase1_ckpt_path):
            _load_original_stage1_ckpt_into_split_model(phase1_ckpt_path)
            phase1_loaded = True
            test_loss, test_acc = _eval_last_head()
            log_row(1, 0, 'phase1_load', test_loss, test_acc, 0.0)
            log_row(1, 0, 'phase1_done', test_loss, test_acc, 0.0)
            print('Backwordstage2 Phase 1 loaded | path {} | test acc {:.2f}%'.format(phase1_ckpt_path, test_acc))
        else:
            set_trainable_for_backwordstage2_phase(model, 1)
            phase1_params = [p for p in model.parameters() if p.requires_grad]
            phase1_opt = optim.Adam(phase1_params, lr=phase_lr, weight_decay=1e-4)
            print('Backwordstage2 Phase 1 | lr {:.6g} | epochs {}'.format(phase_lr, phase1_epochs))
            for ep in range(phase1_epochs):
                set_trainable_for_backwordstage2_phase(model, 1)
                loss_sum = 0.0
                correct = 0
                total = 0
                train_total_batches = len(trainloader)
                if args.max_train_batches and args.max_train_batches > 0:
                    train_total_batches = min(train_total_batches, int(args.max_train_batches))
                for batch_idx, (inputs, targets, indices) in enumerate(trainloader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    phase1_opt.zero_grad()
                    logits = model.forward_stage_logits(inputs, stage_idx=num_stages)
                    loss = F.cross_entropy(logits, targets)
                    loss.backward()
                    phase1_opt.step()
                    preds = torch.argmax(logits.detach(), dim=1)
                    total += targets.size(0)
                    correct += preds.eq(targets).sum().item()
                    loss_sum += float(loss.item())
                    if (batch_idx + 1) >= train_total_batches:
                        break
                train_loss = loss_sum / (batch_idx + 1)
                train_acc = 100.0 * correct / total if total else 0.0
                log_row(1, ep, 'phase1_train', train_loss, train_acc, phase1_opt.param_groups[0]['lr'])
                test_loss, test_acc = _eval_last_head()
                log_row(1, ep, 'phase1_test', test_loss, test_acc, phase1_opt.param_groups[0]['lr'])
                print('Backwordstage2 Phase 1 | Epoch {}/{} | train acc {:.2f}% | test acc {:.2f}%'.format(ep + 1, phase1_epochs, train_acc, test_acc))

            test_loss, test_acc = _eval_last_head()
            log_row(1, phase1_epochs, 'phase1_done', test_loss, test_acc, phase1_opt.param_groups[0]['lr'])
            print('Backwordstage2 Phase 1 done | test acc {:.2f}%'.format(test_acc))

        set_trainable_for_backwordstage2_phase(model, 2)
        phase2_params = [p for p in model.parameters() if p.requires_grad]
        phase2_opt = optim.Adam(phase2_params, lr=phase_lr, weight_decay=1e-4)
        print('Backwordstage2 Phase 2 | lr {:.6g} | epochs {}'.format(phase_lr, phase2_epochs))
        for ep in range(phase2_epochs):
            set_trainable_for_backwordstage2_phase(model, 2)
            loss_sum = 0.0
            correct = 0
            total = 0
            head_correct = [0 for _ in range(num_stages)]
            train_total_batches = len(trainloader)
            if args.max_train_batches and args.max_train_batches > 0:
                train_total_batches = min(train_total_batches, int(args.max_train_batches))
            for batch_idx, (inputs, targets, indices) in enumerate(trainloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                phase2_opt.zero_grad()
                stage_logits = [model.forward_stage_logits(inputs, stage_idx=i + 1) for i in range(num_stages)]
                logits = None
                loss = 0.0
                for head_idx, head_logits in enumerate(stage_logits):
                    logits = head_logits if logits is None else logits + head_logits
                    if args.backwordstage2_per_head_loss:
                        loss = loss + F.cross_entropy(head_logits, targets)
                    preds_i = torch.argmax(head_logits.detach(), dim=1)
                    head_correct[head_idx] += preds_i.eq(targets).sum().item()
                if not args.backwordstage2_per_head_loss:
                    loss = F.cross_entropy(logits, targets)
                loss.backward()
                phase2_opt.step()
                preds = torch.argmax(logits.detach(), dim=1)
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
                loss_sum += float(loss.item())
                if (batch_idx + 1) >= train_total_batches:
                    break
            train_loss = loss_sum / (batch_idx + 1)
            train_acc = 100.0 * correct / total if total else 0.0
            train_head_accs = [100.0 * x / total if total else 0.0 for x in head_correct]
            log_row(2, ep, 'phase2_train', train_loss, train_acc, phase2_opt.param_groups[0]['lr'])
            test_head_losses, test_head_accs, test_loss, test_acc = _eval_all_heads()
            log_row(2, ep, 'phase2_test', test_loss, test_acc, phase2_opt.param_groups[0]['lr'])
            for head_idx, (head_loss, head_acc) in enumerate(zip(test_head_losses, test_head_accs), start=1):
                log_row(2, ep, 'phase2_test_head{}'.format(head_idx), head_loss, head_acc, phase2_opt.param_groups[0]['lr'])
            print(
                'Backwordstage2 Phase 2 | Epoch {}/{} | train acc {:.2f}% | test acc {:.2f}% | train heads {} | test heads {}'.format(
                    ep + 1,
                    phase2_epochs,
                    train_acc,
                    test_acc,
                    _format_head_accs(train_head_accs),
                    _format_head_accs(test_head_accs),
                )
            )

        test_head_losses, test_head_accs, test_loss, test_acc = _eval_all_heads()
        log_row(2, phase2_epochs, 'phase2_done', test_loss, test_acc, phase2_opt.param_groups[0]['lr'])
        for head_idx, (head_loss, head_acc) in enumerate(zip(test_head_losses, test_head_accs), start=1):
            log_row(2, phase2_epochs, 'phase2_done_head{}'.format(head_idx), head_loss, head_acc, phase2_opt.param_groups[0]['lr'])
        print('Backwordstage2 Phase 2 done | test acc {:.2f}% | test heads {}'.format(test_acc, _format_head_accs(test_head_accs)))

        if run_dir is not None:
            torch.save(
                {
                    'net': model.state_dict(),
                    'phase': 2,
                    'phase1_loaded': phase1_loaded,
                    'phase1_ckpt_path': phase1_ckpt_path if phase1_loaded else '',
                },
                os.path.join(run_dir, 'hbn_backwordstage2_final_ckpt.pth'),
            )
        if metrics_fp is not None:
            metrics_fp.close()
        return

    if (args.use_pretrain or args.stage0load) and (not use_backwordstage2):
        ckpt_path = _resolve_stage1_pretrain_path()
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError("Stage1 pretrain checkpoint not found: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("warning: load_state_dict missing_keys={} unexpected_keys={}".format(len(missing), len(unexpected)))
        if isinstance(ckpt, dict) and "sample_log_w" in ckpt:
            sample_w.log_w = ckpt["sample_log_w"].to(device=sample_w.log_w.device, dtype=sample_w.log_w.dtype)
        if isinstance(ckpt, dict) and "alphas" in ckpt:
            alphas = list(ckpt["alphas"])
        if args.unit_head_weight:
            model.force_unit_head_weight = True

    prev_ce_weight = float(getattr(args, "prev_logit_ce_weight", 0.0))
    use_prev_ce = bool(args.prev_logit_ce) or (prev_ce_weight > 0)
    if args.fulltrain:
        prev_ce_weight = 0.0
        use_prev_ce = False

    if use_smhl and (not args.smhl_sample_weight):
        sample_w.log_w.zero_()

    skip_stage1_train = bool((args.use_pretrain or args.stage0load) and (not use_backwordstage2))
    planned_epoch_total = int(sum(stage_epochs[1:] if skip_stage1_train else stage_epochs))
    completed_epoch_count = 0
    cumulative_epoch_time = 0.0

    for stage_idx in range(1, num_stages + 1):
        if (args.use_pretrain or args.stage0load) and stage_idx == 1:
            test_loss, test_acc = eval_ensemble(
                model,
                testloader,
                upto_stage=1,
                max_batches=args.max_eval_batches,
                device=device,
                normalize_head_weights=args.normalize_head_weight,
            )
            log_row(1, 0, 'pretrained_stage1', test_loss, test_acc, 0.0, None, alphas[0] if alphas else None)
            print('Stage 1 pretrained | test acc {:.2f}%'.format(test_acc))
            continue
        if use_smhl:
            set_trainable_for_smhl_stage(model, stage_idx)
        elif args.fulltrain:
            set_trainable_for_fulltrain_stage(model, stage_idx)
        else:
            set_trainable_for_stage(model, stage_idx, backbone_update=args.backbone_update)
        stage_params = [p for p in model.parameters() if p.requires_grad]
        stage_lr = float(stage_lrs[stage_idx - 1])
        if use_smhl:
            stage_opt = optim.Adam(stage_params, lr=stage_lr, weight_decay=1e-4)
            stage_sched = None
        else:
            stage_opt = optim.SGD(stage_params, lr=stage_lr, momentum=0.9, weight_decay=5e-4)
            stage_sched = torch.optim.lr_scheduler.CosineAnnealingLR(stage_opt, T_max=stage_epochs[stage_idx - 1])
        print('Stage {} | lr {:.6g} | epochs {}'.format(stage_idx, stage_lr, stage_epochs[stage_idx - 1]))

        if hasattr(model.head_list[stage_idx - 1], 'classifyheadweight') and (not args.unit_head_weight):
            with torch.no_grad():
                model.head_list[stage_idx - 1].classifyheadweight.fill_(0.0)

        for ep in range(stage_epochs[stage_idx - 1]):
            if use_smhl:
                model.train()
            elif args.fulltrain:
                model.eval()
            else:
                model.eval()
            stage_i = int(stage_idx) - 1
            if use_smhl:
                model.eval()
                for m in model.modules_list:
                    m.train()
                for a in model.adapter_list:
                    a.train()
                for i, h in enumerate(model.head_list):
                    if i == stage_i:
                        h.train()
                    else:
                        h.eval()
            elif args.fulltrain:
                for m in model.modules_list:
                    m.train()
                for a in model.adapter_list:
                    a.train()
                for i, h in enumerate(model.head_list):
                    if i == stage_i:
                        h.train()
                    else:
                        h.eval()
            elif not args.fulltrain:
                model.modules_list[stage_i].train()
                if args.backbone_update and stage_i > 0:
                    model.modules_list[0].train()
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
                    if use_smhl or args.fulltrain:
                        logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
                    elif use_prev_ce and prev_ce_weight > 0:
                        logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
                    else:
                        with torch.no_grad():
                            logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
                else:
                    logits_prev = torch.zeros((inputs.size(0), num_classes), device=device, dtype=torch.float32)

                logits_t = model.forward_stage_logits(inputs, stage_idx=stage_idx)
                if use_smhl:
                    loss = weighted_cross_entropy(logits_prev + logits_t, targets, w_batch)
                elif args.fulltrain:
                    loss = weighted_cross_entropy(logits_prev.detach() + logits_t, targets, w_batch)
                else:
                    if args.loss_mode == 'stage':
                        loss = weighted_cross_entropy(logits_t, targets, w_batch)
                    else:
                        loss = weighted_cross_entropy(logits_prev.detach() + logits_t, targets, w_batch)
                    if stage_idx > 1 and use_prev_ce and prev_ce_weight > 0:
                        loss = loss + F.cross_entropy(logits_prev, targets) * prev_ce_weight
               
                loss.backward()
                stage_opt.step()

                logits_sum = logits_prev.detach() + logits_t.detach()
                preds = torch.argmax(logits_sum, dim=1)
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
                loss_sum += float(loss.item())

                if args.progress_bar:
                    progress_bar(batch_idx, train_total_batches, 'Stage {} | Loss {:.3f} | Acc {:.2f}%'.format(stage_idx, loss_sum / (batch_idx + 1), 100.0 * correct / total))
                if (batch_idx + 1) >= train_total_batches:
                    break

            if stage_sched is not None:
                stage_sched.step()
            lr = stage_opt.param_groups[0]['lr']
            train_loss = loss_sum / (batch_idx + 1)
            train_acc = 100.0 * correct / total if total else 0.0
            log_row(stage_idx, ep, 'train', train_loss, train_acc, lr)

            test_cand_loss, test_cand_acc = eval_candidate_sum(
                model,
                testloader,
                stage_idx=stage_idx,
                alpha=1.0,
                num_classes=num_classes,
                max_batches=args.max_eval_batches,
                device=device,
                normalize_head_weights=args.normalize_head_weight,
            )
            if stage_idx > 1:
                test_prev_loss, test_prev_acc = eval_ensemble(
                    model,
                    testloader,
                    upto_stage=stage_idx - 1,
                    max_batches=args.max_eval_batches,
                    device=device,
                    normalize_head_weights=args.normalize_head_weight,
                )
            else:
                test_prev_loss, test_prev_acc = test_cand_loss, test_cand_acc
            log_row(stage_idx, ep, 'test_prev', test_prev_loss, test_prev_acc, lr)
            log_row(stage_idx, ep, 'test_cand', test_cand_loss, test_cand_acc, lr)
            epoch_time = time.time() - t0
            completed_epoch_count += 1
            cumulative_epoch_time += epoch_time
            avg_epoch_time = cumulative_epoch_time / completed_epoch_count if completed_epoch_count > 0 else epoch_time
            remaining_epoch_count = max(0, planned_epoch_total - completed_epoch_count)
            eta_seconds = avg_epoch_time * remaining_epoch_count
            eta_finish = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
            print(
                'Stage {} | Epoch {}/{} | train acc {:.2f}% | test prev {:.2f}% | test cand {:.2f}% | epoch {} | eta {} | finish {}'.format(
                    stage_idx,
                    ep + 1,
                    stage_epochs[stage_idx - 1],
                    train_acc,
                    test_prev_acc,
                    test_cand_acc,
                    _format_duration(epoch_time),
                    _format_duration(eta_seconds),
                    eta_finish.strftime('%Y-%m-%d %H:%M:%S'),
                )
            )

        eps_used = None
        alpha_used = None
        if (not use_smhl) or args.smhl_sample_weight or args.smhl_head_weight:
            mode = str(args.sample_weight_mode).lower()
            model.eval()
            errors01_unweighted = torch.zeros(len(trainset), dtype=torch.float32)
            difficulty_unweighted = torch.zeros(len(trainset), dtype=torch.float32)
            with torch.no_grad():
                for batch_idx, (inputs, targets, indices) in enumerate(train_eval_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    logits_t = model.forward_stage_logits(inputs, stage_idx=stage_idx)
                    if args.alpha_error_mode == 'sum':
                        if stage_idx > 1:
                            logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
                        else:
                            logits_prev = torch.zeros((inputs.size(0), num_classes), device=device, dtype=torch.float32)
                        logits_for_err = logits_prev + logits_t
                    else:
                        logits_for_err = logits_t
                    preds = torch.argmax(logits_for_err, dim=1)
                    errors01 = preds.ne(targets).to(torch.float32)
                    errors01_unweighted[indices] = errors01.detach().cpu()
                    if mode == 'binary':
                        scores = errors01
                    else:
                        scores = _compute_difficulty_scores_from_logits(logits_for_err, targets)
                    difficulty_unweighted[indices] = scores.detach().cpu()
                    if args.max_eval_batches and (batch_idx + 1) >= args.max_eval_batches:
                        break

            epsilon = sample_w.weighted_error(errors01_unweighted)
            alpha, eps_used = compute_alpha_paper(epsilon, eps_clip=args.eps_clip, alpha_clip=args.alpha_clip)
            alpha_used = float(alpha)
            if alpha_used < 0:
                alpha_used = 0.1
            alphas.append(alpha_used)
            with torch.no_grad():
                if args.unit_head_weight:
                    model.head_list[stage_idx - 1].classifyheadweight.fill_(1.0)
                else:
                    model.head_list[stage_idx - 1].classifyheadweight.fill_(alpha_used)
            difficulty_weighted = torch.zeros(len(trainset), dtype=torch.float32)
            with torch.no_grad():
                for batch_idx, (inputs, targets, indices) in enumerate(train_eval_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    logits_t = model.forward_stage_logits(inputs, stage_idx=stage_idx)
                    if args.alpha_error_mode == 'sum':
                        if stage_idx > 1:
                            logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
                        else:
                            logits_prev = torch.zeros((inputs.size(0), num_classes), device=device, dtype=torch.float32)
                        logits_for_err = logits_prev + logits_t * alpha_used
                    else:
                        logits_for_err = logits_t
                    if mode == 'binary':
                        preds = torch.argmax(logits_for_err, dim=1)
                        scores = preds.ne(targets).to(torch.float32)
                    else:
                        scores = _compute_difficulty_scores_from_logits(logits_for_err, targets)
                    difficulty_weighted[indices] = scores.detach().cpu()
                    if args.max_eval_batches and (batch_idx + 1) >= args.max_eval_batches:
                        break
            sample_w.update(difficulty_weighted, alpha_used, alpha_clip=args.alpha_clip)

        test_loss, test_acc = eval_ensemble(
            model,
            testloader,
            upto_stage=stage_idx,
            max_batches=args.max_eval_batches,
            device=device,
            normalize_head_weights=args.normalize_head_weight,
        )
        log_row(stage_idx, stage_epochs[stage_idx - 1], 'stage_done', test_loss, test_acc, 0.0, eps_used, alpha_used)
        if eps_used is None or alpha_used is None:
            print('Stage {} done | test acc {:.2f}%'.format(stage_idx, test_acc))
        else:
            print('Stage {} done | epsilon {:.6f} | alpha {:.6f} | test acc {:.2f}%'.format(stage_idx, eps_used, alpha_used, test_acc))
        print('Stage {} | {}'.format(stage_idx, _weight_summary_string()))

        if run_dir is not None and stage_idx == num_stages:
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
