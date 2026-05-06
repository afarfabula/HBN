import torch

from .merger import HBNMerger
from .boosting_core import set_trainable_for_stage
from .toolkit import SplitModule


def main():
    cfg = SplitModule('resnet18').get_HBN_model_Config()
    merger = HBNMerger(**cfg)
    x = torch.randn(4, 3, 32, 32)
    merged, logits_list = merger(x)
    assert tuple(merged.shape) == (4, 100)
    assert isinstance(logits_list, list)
    assert len(logits_list) == 4
    assert tuple(logits_list[0].shape) == (4, 100)

    set_trainable_for_stage(merger, 2, backbone_update=True)
    assert any(p.requires_grad for p in merger.modules_list[0].parameters())

    for name in ("convnext_tiny_cifar", "vit_tiny_cifar"):
        cfg2 = SplitModule(name).get_HBN_model_Config()
        merger2 = HBNMerger(**cfg2)
        merged2, logits_list2 = merger2(x)
        assert tuple(merged2.shape) == (4, 100)
        assert isinstance(logits_list2, list)
        assert len(logits_list2) == 4

    cfg3 = SplitModule('resnet18').get_HBN_model_Config()
    merger3 = HBNMerger(**cfg3)
    set_trainable_for_stage(merger3, 2, backbone_update=True)
    x2 = torch.randn(2, 3, 32, 32)
    y2 = torch.randint(0, 100, (2,))
    logits_prev = merger3.forward_merged_logits(x2, upto_stage=1)
    _ = torch.nn.functional.cross_entropy(logits_prev, y2)


if __name__ == '__main__':
    main()
