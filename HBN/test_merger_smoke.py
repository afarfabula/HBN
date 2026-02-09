import torch
import torch.nn as nn

from .merger import HBNMerger
from .toolkit import GAPHead


class IdentityAdapter(nn.Module):
    def forward(self, x):
        return x


def main():
    num_classes = 7
    modules = [
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
    ]
    adapters = [IdentityAdapter(), IdentityAdapter()]
    heads = [GAPHead(num_classes) for _ in range(len(modules))]
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
    assert isinstance(logits_list, list) and len(logits_list) == len(modules)
    assert tuple(logits_list[0].shape) == (4, num_classes)

    class BadAdapter(nn.Module):
        def forward(self, x):
            return x[:, :, :16, :16]

    try:
        HBNMerger(
            modules=modules,
            adapter_list=[BadAdapter(), IdentityAdapter()],
            head_list=heads,
            num_classes=num_classes,
            intermediate_feature_shape_list=[(3, 32, 32), (16, 32, 32), (16, 32, 32), (32, 32, 32)],
        )
    except Exception:
        pass
    else:
        raise AssertionError('expected bad adapter to fail')


if __name__ == '__main__':
    main()
