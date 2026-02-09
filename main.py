'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import csv
import time
import datetime
import math
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'],
                    help='dataset')
parser.add_argument('--model', default='ResNet18', type=str,
                    help='model name (e.g., ResNet18)')
parser.add_argument('--epochs', default=200, type=int, help='total epochs')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--num-workers', default=2, type=int, help='dataloader workers')
parser.add_argument('--data-dir', default='./data', type=str, help='dataset root dir')
parser.add_argument('--max-train-batches', default=0, type=int, help='limit train batches per epoch (0=all)')
parser.add_argument('--max-test-batches', default=0, type=int, help='limit test batches per epoch (0=all)')
parser.add_argument('--eval-only', action='store_true', help='only run evaluation on test set')
parser.add_argument('--checkpoint-path', default='./checkpoint/ckpt.pth', type=str, help='checkpoint path')
parser.add_argument('--log-dir', default='./runs', type=str, help='log directory (empty to disable)')
parser.add_argument('--run-name', default='', type=str, help='run name (default: timestamped)')
parser.add_argument('--hbn', action='store_true', help='stage-wise boosting with frozen previous heads')
parser.add_argument('--stage-epochs', default='50,50,50,50', type=str, help='comma-separated epochs per stage')
parser.add_argument('--hbn-eps-clip', default=1e-12, type=float, help='epsilon clip for alpha stability')
parser.add_argument('--hbn-alpha-clip', default=8.0, type=float, help='alpha clip for weight update stability')
parser.add_argument('--hbn-eval-stage', default=4, type=int, help='stage index for eval-only (1-4)')
parser.add_argument('--hbn-eval-mode', default='ensemble', choices=['head', 'ensemble'], help='eval-only mode')
parser.add_argument('--run-tests', action='store_true', help='run minimal self-tests and exit')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

def load_checkpoint_into_model(model, checkpoint_path, map_location):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint['net']
    model_has_module = any(k.startswith('module.') for k in model.state_dict().keys())
    ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())

    if ckpt_has_module and not model_has_module:
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    elif (not ckpt_has_module) and model_has_module:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return checkpoint

run_dir = None
metrics_writer = None
metrics_fp = None
log_fp = None
orig_stdout = sys.stdout
orig_stderr = sys.stderr

if args.log_dir:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or '{}_{}_{}'.format(ts, args.dataset, args.model)
    run_dir = os.path.join(args.log_dir, run_name)
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    log_fp = open(os.path.join(run_dir, 'train.log'), 'w', buffering=1)
    sys.stdout = Tee(orig_stdout, log_fp)
    sys.stderr = Tee(orig_stderr, log_fp)

    metrics_fp = open(os.path.join(run_dir, 'metrics.csv'), 'w', newline='')
    metrics_writer = csv.writer(metrics_fp)
    metrics_writer.writerow(['epoch', 'split', 'loss', 'acc', 'lr', 'time_sec', 'stage', 'alpha', 'epsilon'])

def _log_row(epoch, split, loss, acc, lr, time_sec, stage=None, alpha=None, epsilon=None):
    if metrics_writer is None:
        return
    metrics_writer.writerow([epoch, split, loss, acc, lr, time_sec, stage, alpha, epsilon])

def _log_close():
    if metrics_fp is not None:
        metrics_fp.close()
    if log_fp is not None:
        log_fp.close()
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx

class SampleWeights(object):
    def __init__(self, num_samples, device='cpu', dtype=torch.float64):
        self.log_w = torch.zeros(num_samples, dtype=dtype, device=device)
        self._normalize_inplace()

    def _normalize_inplace(self):
        m = torch.max(self.log_w)
        self.log_w = self.log_w - (m + torch.log(torch.sum(torch.exp(self.log_w - m))))

    def weights(self):
        return torch.exp(self.log_w)

    def batch_weights(self, indices, device=None, dtype=torch.float32):
        w = torch.exp(self.log_w[indices])
        if device is not None:
            w = w.to(device)
        return w.to(dtype)

    def weighted_error(self, errors01):
        w = torch.exp(self.log_w)
        return torch.sum(w * errors01.to(w.dtype)).item()

    def update(self, errors01, alpha, alpha_clip=8.0):
        # w_i^(t+1) = w_i^(t) * exp(alpha_t * I[wrong])
        # We keep log-weights to avoid overflow: log(w) += alpha_t * I[wrong], then normalize.
        if alpha > alpha_clip:
            alpha = alpha_clip
        if alpha < -alpha_clip:
            alpha = -alpha_clip
        self.log_w = self.log_w + float(alpha) * errors01.to(self.log_w.dtype)
        self._normalize_inplace()

def weighted_cross_entropy(logits, targets, sample_w):
    losses = F.cross_entropy(logits, targets, reduction='none')
    w = sample_w.to(losses.dtype)
    denom = torch.sum(w)
    if denom.item() == 0:
        return torch.mean(losses)
    return torch.sum(losses * w) / denom

def compute_alpha(epsilon, eps_clip=1e-12, alpha_clip=8.0):
    # alpha_t = 0.5 * ln((1 - eps_t) / eps_t)
    # eps_t is the weighted misclassification rate; we clip it for numerical stability.
    eps = float(epsilon)
    if eps < eps_clip:
        eps = eps_clip
    if eps > 1.0 - eps_clip:
        eps = 1.0 - eps_clip
    alpha = 0.5 * math.log((1.0 - eps) / eps)
    if alpha > alpha_clip:
        alpha = alpha_clip
    if alpha < -alpha_clip:
        alpha = -alpha_clip
    return alpha, eps

def ensemble_logits_from_list(logits_list, alphas):
    out = None
    for i, logits in enumerate(logits_list):
        a = float(alphas[i]) if i < len(alphas) else 0.0
        if out is None:
            out = logits * a
        else:
            out = out + logits * a
    return out

def _run_self_tests():
    w = SampleWeights(4)
    s = torch.sum(w.weights()).item()
    assert abs(s - 1.0) < 1e-9

    errors = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    eps = w.weighted_error(errors)
    alpha, eps_used = compute_alpha(eps, eps_clip=1e-12, alpha_clip=8.0)
    assert eps_used > 0.0 and eps_used < 1.0
    w.update(errors, alpha, alpha_clip=8.0)
    s2 = torch.sum(w.weights()).item()
    assert abs(s2 - 1.0) < 1e-9

    l1 = torch.tensor([[1.0, 2.0]])
    l2 = torch.tensor([[3.0, 4.0]])
    ens = ensemble_logits_from_list([l1, l2], [0.5, 2.0])
    assert torch.allclose(ens, l1 * 0.5 + l2 * 2.0)

if args.run_tests:
    _run_self_tests()
    print('self-tests: ok')
    _log_close()
    raise SystemExit(0)

# Data
print('==> Preparing data..')
dataset_name = args.dataset.lower()
if dataset_name == 'cifar100':
    normalize_mean = (0.5071, 0.4867, 0.4408)
    normalize_std = (0.2675, 0.2565, 0.2761)
    num_classes = 100
else:
    normalize_mean = (0.4914, 0.4822, 0.4465)
    normalize_std = (0.2023, 0.1994, 0.2010)
    num_classes = 10

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

dataset_cls = torchvision.datasets.CIFAR100 if dataset_name == 'cifar100' else torchvision.datasets.CIFAR10
trainset = dataset_cls(
    root=args.data_dir, train=True, download=True, transform=transform_train)
trainset_for_loader = IndexedDataset(trainset) if args.hbn else trainset
trainloader = torch.utils.data.DataLoader(
    trainset_for_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

testset = dataset_cls(
    root=args.data_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

# Model
print('==> Building model..')
model_fn = globals().get(args.model)
if model_fn is None:
    raise ValueError('Unknown model: {}'.format(args.model))

try:
    net = model_fn(num_classes=num_classes)
except TypeError:
    net = model_fn()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = load_checkpoint_into_model(net, args.checkpoint_path, device)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def _unwrap_model(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m

def hbn_predict_logits(m, inputs, upto_stage, alphas, return_all=False):
    base = _unwrap_model(m)
    logits_list = base(inputs, upto_stage=upto_stage, return_all=True)
    ens = ensemble_logits_from_list(logits_list, alphas)
    if return_all:
        return ens, logits_list
    return ens

def hbn_eval_head(m, loader, stage_idx, max_batches=0):
    m.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            logits = _unwrap_model(m)(inputs, upto_stage=stage_idx, return_all=False)
            loss = F.cross_entropy(logits, targets)
            loss_sum += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if max_batches and (batch_idx + 1) >= max_batches:
                break
    avg_loss = loss_sum / (batch_idx + 1)
    acc = 100.0 * correct / total if total else 0.0
    return avg_loss, acc

def hbn_eval_ensemble(m, loader, upto_stage, alphas, max_batches=0):
    m.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            logits = hbn_predict_logits(m, inputs, upto_stage=upto_stage, alphas=alphas)
            loss = F.cross_entropy(logits, targets)
            loss_sum += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if max_batches and (batch_idx + 1) >= max_batches:
                break
    avg_loss = loss_sum / (batch_idx + 1)
    acc = 100.0 * correct / total if total else 0.0
    return avg_loss, acc


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    t0 = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
            break
    avg_loss = train_loss / (batch_idx + 1)
    acc = 100. * correct / total if total else 0.0
    return {'loss': avg_loss, 'acc': acc, 'time_sec': time.time() - t0}


def test(epoch):
    global best_acc
    t0 = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if args.max_test_batches and (batch_idx + 1) >= args.max_test_batches:
                break

    # Save checkpoint.
    avg_loss = test_loss / (batch_idx + 1)
    acc = 100.*correct/total if total else 0.0
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return {'loss': avg_loss, 'acc': acc, 'time_sec': time.time() - t0}


if args.eval_only:
    if not args.resume:
        checkpoint = load_checkpoint_into_model(net, args.checkpoint_path, device)
        best_acc = checkpoint.get('acc', 0)
        start_epoch = checkpoint.get('epoch', 0)
    if args.hbn:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        alphas = ckpt.get('alphas', [])
        stage_idx = int(args.hbn_eval_stage)
        if stage_idx < 1 or stage_idx > 4:
            raise ValueError('--hbn-eval-stage must be in [1,4]')
        if args.hbn_eval_mode == 'head':
            loss, acc = hbn_eval_head(net, testloader, stage_idx, max_batches=args.max_test_batches)
            print('HBN eval head{} | loss {:.4f} | acc {:.2f}%'.format(stage_idx, loss, acc))
            _log_row(start_epoch, 'hbn_eval_head{}'.format(stage_idx), loss, acc, 0.0, 0.0, stage=stage_idx)
        else:
            loss, acc = hbn_eval_ensemble(net, testloader, stage_idx, alphas[:stage_idx], max_batches=args.max_test_batches)
            print('HBN eval ens{} | loss {:.4f} | acc {:.2f}%'.format(stage_idx, loss, acc))
            _log_row(start_epoch, 'hbn_eval_ens{}'.format(stage_idx), loss, acc, 0.0, 0.0, stage=stage_idx)
    else:
        test_metrics = test(start_epoch)
        lr = optimizer.param_groups[0]['lr']
        _log_row(start_epoch, 'test', test_metrics['loss'], test_metrics['acc'], lr, test_metrics['time_sec'])
elif args.hbn:
    if args.model != 'HBNResNet18':
        raise ValueError('When --hbn is set, use --model HBNResNet18')

    try:
        stage_epochs = [int(x.strip()) for x in args.stage_epochs.split(',') if x.strip()]
    except Exception:
        raise ValueError('Invalid --stage-epochs: {}'.format(args.stage_epochs))
    if len(stage_epochs) != 4:
        raise ValueError('Expected 4 comma-separated values for --stage-epochs, got {}'.format(args.stage_epochs))

    base = _unwrap_model(net)
    if not hasattr(base, 'freeze_before_stage'):
        raise ValueError('Model does not support stage freezing: {}'.format(args.model))

    train_eval_loader = torch.utils.data.DataLoader(
        IndexedDataset(trainset), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample_w = SampleWeights(len(trainset))
    alphas = []

    for stage_idx in [1, 2, 3, 4]:
        base.freeze_before_stage(stage_idx)

        stage_params = [p for p in net.parameters() if p.requires_grad]
        stage_opt = optim.SGD(stage_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        stage_sched = torch.optim.lr_scheduler.CosineAnnealingLR(stage_opt, T_max=stage_epochs[stage_idx - 1])

        for local_epoch in range(stage_epochs[stage_idx - 1]):
            epoch_tag = (stage_idx - 1) * 10000 + local_epoch
            print('\nStage: %d/4 | Epoch: %d/%d' % (stage_idx, local_epoch, stage_epochs[stage_idx - 1] - 1))
            t0 = time.time()
            net.train()
            loss_sum = 0.0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets, indices) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                indices = indices.to('cpu')
                w_batch = sample_w.batch_weights(indices, device=device, dtype=torch.float32)

                stage_opt.zero_grad()
                logits = base(inputs, upto_stage=stage_idx, return_all=False)
                loss = weighted_cross_entropy(logits, targets, w_batch)
                loss.backward()
                stage_opt.step()

                loss_sum += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (loss_sum/(batch_idx+1), 100.*correct/total, correct, total))
                if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
                    break

            stage_sched.step()
            lr = stage_opt.param_groups[0]['lr']
            _log_row(epoch_tag, 'train_head{}'.format(stage_idx), loss_sum/(batch_idx+1), 100.*correct/total, lr, time.time() - t0, stage=stage_idx)

            head_loss, head_acc = hbn_eval_head(net, testloader, stage_idx, max_batches=args.max_test_batches)
            ens_loss, ens_acc = hbn_eval_ensemble(net, testloader, stage_idx, alphas + [0.0], max_batches=args.max_test_batches) if stage_idx > 1 else (head_loss, head_acc)
            _log_row(epoch_tag, 'test_head{}'.format(stage_idx), head_loss, head_acc, lr, 0.0, stage=stage_idx)
            _log_row(epoch_tag, 'test_ens{}'.format(stage_idx), ens_loss, ens_acc, lr, 0.0, stage=stage_idx)

        net.eval()
        errors01 = torch.zeros(len(trainset), dtype=torch.float32)
        with torch.no_grad():
            for batch_idx, (inputs, targets, indices) in enumerate(train_eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = base(inputs, upto_stage=stage_idx, return_all=False)
                preds = torch.argmax(logits, dim=1)
                wrong = preds.ne(targets).to(torch.float32).cpu()
                errors01[indices] = wrong
                if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
                    break

        epsilon = sample_w.weighted_error(errors01)
        alpha, eps_used = compute_alpha(epsilon, eps_clip=args.hbn_eps_clip, alpha_clip=args.hbn_alpha_clip)
        alphas.append(alpha)
        sample_w.update(errors01, alpha, alpha_clip=args.hbn_alpha_clip)

        print('Stage %d done | epsilon: %.6f | alpha: %.6f' % (stage_idx, eps_used, alpha))

        head_loss, head_acc = hbn_eval_head(net, testloader, stage_idx, max_batches=args.max_test_batches)
        ens_loss, ens_acc = hbn_eval_ensemble(net, testloader, stage_idx, alphas, max_batches=args.max_test_batches)
        _log_row((stage_idx - 1) * 10000 + stage_epochs[stage_idx - 1], 'stage_done_head{}'.format(stage_idx), head_loss, head_acc, 0.0, 0.0, stage=stage_idx, alpha=alpha, epsilon=eps_used)
        _log_row((stage_idx - 1) * 10000 + stage_epochs[stage_idx - 1], 'stage_done_ens{}'.format(stage_idx), ens_loss, ens_acc, 0.0, 0.0, stage=stage_idx, alpha=alpha, epsilon=eps_used)

        if run_dir is not None:
            torch.save(
                {
                    'net': net.state_dict(),
                    'stage': stage_idx,
                    'alphas': alphas,
                    'epsilon': eps_used,
                },
                os.path.join(run_dir, 'hbn_stage{}_ckpt.pth'.format(stage_idx))
            )
else:
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_metrics = train(epoch)
        test_metrics = test(epoch)
        lr = optimizer.param_groups[0]['lr']
        _log_row(epoch, 'train', train_metrics['loss'], train_metrics['acc'], lr, train_metrics['time_sec'])
        _log_row(epoch, 'test', test_metrics['loss'], test_metrics['acc'], lr, test_metrics['time_sec'])
        scheduler.step()

_log_close()
