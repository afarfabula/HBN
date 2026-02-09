import torch
import torch.nn as nn

from .merger import HBNMerger
from .toolkit import Adapter, GAPHead


def main():
    num_classes = 7
    modules = [
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
    ]
    adapter_list = [Adapter(16), Adapter(32), Adapter(64)]
    head_list = [GAPHead(num_classes), GAPHead(num_classes), GAPHead(num_classes), GAPHead(num_classes)]
    shapes = [(3, 32, 32), (16, 32, 32), (32, 32, 32), (64, 32, 32), (64, 32, 32)]

    merger = HBNMerger(modules, adapter_list, head_list, num_classes, shapes)
    x = torch.randn(2, 3, 32, 32)
    merged, _ = merger(x)
    assert tuple(merged.shape) == (2, num_classes)

    subs = merger.get_SubMergedModu()
    assert len(subs) == 3
    for i, sm in enumerate(subs, start=1):
        merged_i, _ = sm(x)
        assert tuple(merged_i.shape) == (2, num_classes)


if __name__ == '__main__':
    main()

