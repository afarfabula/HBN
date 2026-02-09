import argparse
import os
import re

import torch
import torchvision
import torchvision.transforms as transforms

from .merger import HBNMerger
from .toolkit import SplitModule


def _resolve_ckpt_path(path_or_run_dir: str) -> str:
    p = os.path.abspath(os.path.expanduser(path_or_run_dir))
    if os.path.isfile(p):
        return p
    if not os.path.isdir(p):
        raise FileNotFoundError(p)

    best = None
    best_stage = None
    for name in os.listdir(p):
        m = re.match(r"^hbn_stage(\d+)_ckpt\.pth$", name)
        if not m:
            continue
        s = int(m.group(1))
        if best_stage is None or s > best_stage:
            best_stage = s
            best = os.path.join(p, name)
    if best is None:
        raise FileNotFoundError("No hbn_stage*_ckpt.pth found under: {}".format(p))
    return best


def _infer_num_stages_from_state_dict(state_dict) -> int:
    max_head = -1
    for k in state_dict.keys():
        m = re.match(r"(?:module\.)?head_list\.(\d+)\.", k)
        if m:
            max_head = max(max_head, int(m.group(1)))
    if max_head >= 0:
        return max_head + 1
    raise ValueError("Unable to infer num_stages from checkpoint state_dict")


def _load_state_dict_compat(model: torch.nn.Module, state_dict) -> None:
    model_has_module = any(k.startswith("module.") for k in model.state_dict().keys())
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())

    if ckpt_has_module and not model_has_module:
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    elif (not ckpt_has_module) and model_has_module:
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def _build_model_for_ckpt(basemodel: str, num_classes: int, ckpt_state_dict, empty_stage_num=None) -> HBNMerger:
    num_stages = _infer_num_stages_from_state_dict(ckpt_state_dict)
    if empty_stage_num is not None:
        cfg = SplitModule(basemodel, num_classes=num_classes, empty_stage_num=empty_stage_num).get_HBN_model_Config()
        if len(cfg["modules"]) != num_stages:
            raise ValueError(
                "empty_stage_num {} gives {} stages, but checkpoint has {} stages".format(
                    empty_stage_num, len(cfg["modules"]), num_stages
                )
            )
        return HBNMerger(**cfg)

    for es in [max(0, num_stages - 1)] + list(range(0, max(0, num_stages * 2 + 1))):
        try:
            cfg = SplitModule(basemodel, num_classes=num_classes, empty_stage_num=es).get_HBN_model_Config()
        except Exception:
            continue
        if len(cfg["modules"]) == num_stages:
            return HBNMerger(**cfg)

    raise ValueError(
        "Failed to construct model for basemodel={} with num_stages={}. "
        "Try passing --empty-stage-num explicitly.".format(basemodel, num_stages)
    )


@torch.no_grad()
def _eval_stage_acc(model: HBNMerger, loader, num_stages: int, device: str, max_batches: int = 0):
    model.eval()
    correct = [0 for _ in range(num_stages)]
    total = 0
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            inputs, targets, _ = batch
        else:
            inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        _, logits_list = model(inputs)
        if len(logits_list) != num_stages:
            raise ValueError("Model returned {} stage logits, expected {}".format(len(logits_list), num_stages))

        for i in range(num_stages):
            preds = torch.argmax(logits_list[i], dim=1)
            correct[i] += int(preds.eq(targets).sum().item())
        total += int(targets.size(0))

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    return [100.0 * c / total if total else 0.0 for c in correct]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--basemodel", default="originalresnet", type=str)
    parser.add_argument("--empty-stage-num", default=None, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--max-batches", default=0, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--train-aug", action="store_true")
    args = parser.parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = args.dataset.lower()
    if dataset_name == "cifar100":
        normalize_mean = (0.5071, 0.4867, 0.4408)
        normalize_std = (0.2675, 0.2565, 0.2761)
        dataset_cls = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
        dataset_cls = torchvision.datasets.CIFAR10
        num_classes = 10

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    train_transform = transform_train if args.train_aug else transform_test
    trainset = dataset_cls(root=args.data_dir, train=True, download=True, transform=train_transform)
    testset = dataset_cls(root=args.data_dir, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    ckpt_path = _resolve_ckpt_path(args.run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "net" not in ckpt:
        raise ValueError("Unexpected checkpoint format: {}".format(type(ckpt)))
    state_dict = ckpt["net"]

    model = _build_model_for_ckpt(
        basemodel=args.basemodel,
        num_classes=num_classes,
        ckpt_state_dict=state_dict,
        empty_stage_num=args.empty_stage_num,
    ).to(device)
    _load_state_dict_compat(model, state_dict)

    num_stages = len(model.head_list)
    train_accs = _eval_stage_acc(model, trainloader, num_stages=num_stages, device=device, max_batches=args.max_batches)
    test_accs = _eval_stage_acc(model, testloader, num_stages=num_stages, device=device, max_batches=args.max_batches)

    print("ckpt:", ckpt_path)
    print("device:", device)
    print("dataset:", dataset_name)
    print("basemodel:", args.basemodel)
    print("stages:", num_stages)
    print("")
    print("{:>5s}  {:>10s}  {:>10s}".format("stage", "train_acc", "test_acc"))
    for s in range(1, num_stages + 1):
        print("{:>5d}  {:>10.3f}  {:>10.3f}".format(s, float(train_accs[s - 1]), float(test_accs[s - 1])))


if __name__ == "__main__":
    main()
