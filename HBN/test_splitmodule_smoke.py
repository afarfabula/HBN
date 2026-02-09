import torch

from .merger import HBNMerger
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


if __name__ == '__main__':
    main()

