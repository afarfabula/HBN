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

    merger.force_unit_head_weight = True
    for i, h in enumerate(merger.head_list):
        with torch.no_grad():
            h.classifyheadweight.fill_(0.0 if i == 0 else 2.0)
    merged2, _ = merger(x)
    assert torch.allclose(merged2, logits_list[0] + logits_list[1] + logits_list[2], atol=1e-5)

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
