import torch

from .boosting_core import weighted_cross_entropy
from .merger import HBNMerger
from .toolkit import SplitModule


def main():
    cfg = SplitModule("resnet18").get_HBN_model_Config()
    model = HBNMerger(**cfg)
    for p in model.parameters():
        p.requires_grad = True

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 100, (4,))
    w = torch.ones((4,), dtype=torch.float32)

    logits_prev = model.forward_merged_logits(x, upto_stage=1)
    logits_t = model.forward_stage_logits(x, stage_idx=2)
    loss = weighted_cross_entropy(logits_prev + logits_t, y, w)
    loss.backward()

    assert any(p.grad is not None for p in model.modules_list[0].parameters())
    assert any(p.grad is not None for p in model.modules_list[1].parameters())


if __name__ == "__main__":
    main()

