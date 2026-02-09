import torch
import torch.nn as nn

from .merger import HBNMerger
from .toolkit import Adapter, GAPHead


def main():
    num_classes = 100
    modules = [
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
    ]
    adapters = [Adapter(16), Adapter(16)]
    heads = [GAPHead(num_classes), GAPHead(num_classes), GAPHead(num_classes)]
    merger = HBNMerger(
        modules=modules,
        adapter_list=adapters,
        head_list=heads,
        num_classes=num_classes,
        intermediate_feature_shape_list=[(3, 32, 32), (16, 32, 32), (16, 32, 32), (32, 32, 32)],
    )

    x = torch.randn(4, 3, 32, 32)
    merged, logits_list = merger(x)
    assert tuple(merged.shape) == (4, num_classes)
    assert isinstance(logits_list, list)
    assert len(logits_list) == 3
    assert tuple(logits_list[0].shape) == (4, num_classes)
    assert tuple(logits_list[1].shape) == (4, num_classes)
    assert tuple(logits_list[2].shape) == (4, num_classes)


if __name__ == '__main__':
    main()
