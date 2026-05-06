import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import models  # noqa: F401
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class _ConvBnRelu(nn.Module):
    def __init__(self, conv: nn.Module, bn: nn.Module):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _EfficientNetStem(nn.Module):
    def __init__(self, conv: nn.Module, bn: nn.Module, swish_fn):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.swish = swish_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(self.bn(self.conv(x)))


class _EfficientNetBackbone(nn.Module):
    def __init__(self, stem: nn.Module, layers: nn.Module):
        super().__init__()
        self.stem = stem
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        return x


class _DenseNetStage1(nn.Module):
    def __init__(self, conv1: nn.Module, dense1: nn.Module, trans1: nn.Module):
        super().__init__()
        self.conv1 = conv1
        self.dense1 = dense1
        self.trans1 = trans1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        return x


class _DenseNetStage(nn.Module):
    def __init__(self, dense: nn.Module, trans: nn.Module):
        super().__init__()
        self.dense = dense
        self.trans = trans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.trans(x)
        return x


class _DenseNetStage4(nn.Module):
    def __init__(self, dense4: nn.Module):
        super().__init__()
        self.dense4 = dense4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense4(x)


class PreActResNet18Split(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from models.preact_resnet import PreActResNet18

        base = PreActResNet18()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)

        self.stage1 = nn.Sequential(base.conv1, base.layer1)
        self.stage2 = base.layer2
        self.stage3 = base.layer3
        self.stage4 = base.layer4

    def get_submodules(self):
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalPreActResNet18(nn.Module):
    def __init__(self, num_classes: int = 100, empty_stage_num: int = 3):
        super().__init__()
        from models.preact_resnet import PreActResNet18

        base = PreActResNet18()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError("empty_stage_num must be non-negative")

        backbone = nn.Sequential(base.conv1, base.layer1, base.layer2, base.layer3, base.layer4)
        stages = [backbone]
        for _ in range(self.empty_stage_num):
            stages.append(nn.Identity())
        self.stages = nn.ModuleList(stages)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class ResNeXt29_32x4dSplit(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from models.resnext import ResNeXt29_32x4d

        base = ResNeXt29_32x4d()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)

        self.stage1 = _ConvBnRelu(base.conv1, base.bn1)
        self.stage2 = base.layer1
        self.stage3 = base.layer2
        self.stage4 = base.layer3

    def get_submodules(self):
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalResNeXt29_32x4d(nn.Module):
    def __init__(self, num_classes: int = 100, empty_stage_num: int = 3):
        super().__init__()
        from models.resnext import ResNeXt29_32x4d

        base = ResNeXt29_32x4d()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError("empty_stage_num must be non-negative")

        backbone = nn.Sequential(_ConvBnRelu(base.conv1, base.bn1), base.layer1, base.layer2, base.layer3)
        stages = [backbone]
        for _ in range(self.empty_stage_num):
            stages.append(nn.Identity())
        self.stages = nn.ModuleList(stages)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class RegNetX_200MFSplit(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from models.regnet import RegNet

        cfg = {
            "depths": [1, 1, 4, 7],
            "widths": [24, 56, 152, 368],
            "strides": [1, 1, 2, 2],
            "group_width": 8,
            "bottleneck_ratio": 1,
            "se_ratio": 0,
        }
        base = RegNet(cfg, num_classes=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)

        self.stage1 = nn.Sequential(_ConvBnRelu(base.conv1, base.bn1), base.layer1)
        self.stage2 = base.layer2
        self.stage3 = base.layer3
        self.stage4 = base.layer4

    def get_submodules(self):
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalRegNetX_200MF(nn.Module):
    def __init__(self, num_classes: int = 100, empty_stage_num: int = 3):
        super().__init__()
        from models.regnet import RegNet

        cfg = {
            "depths": [1, 1, 4, 7],
            "widths": [24, 56, 152, 368],
            "strides": [1, 1, 2, 2],
            "group_width": 8,
            "bottleneck_ratio": 1,
            "se_ratio": 0,
        }
        base = RegNet(cfg, num_classes=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError("empty_stage_num must be non-negative")

        backbone = nn.Sequential(
            _ConvBnRelu(base.conv1, base.bn1),
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        stages = [backbone]
        for _ in range(self.empty_stage_num):
            stages.append(nn.Identity())
        self.stages = nn.ModuleList(stages)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class RegNetY_400MFSplit(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from models.regnet import RegNet

        cfg = {
            "depths": [1, 2, 7, 12],
            "widths": [32, 64, 160, 384],
            "strides": [1, 1, 2, 2],
            "group_width": 16,
            "bottleneck_ratio": 1,
            "se_ratio": 0.25,
        }
        base = RegNet(cfg, num_classes=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)

        self.stage1 = nn.Sequential(_ConvBnRelu(base.conv1, base.bn1), base.layer1)
        self.stage2 = base.layer2
        self.stage3 = base.layer3
        self.stage4 = base.layer4

    def get_submodules(self):
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalRegNetY_400MF(nn.Module):
    def __init__(self, num_classes: int = 100, empty_stage_num: int = 3):
        super().__init__()
        from models.regnet import RegNet

        cfg = {
            "depths": [1, 2, 7, 12],
            "widths": [32, 64, 160, 384],
            "strides": [1, 1, 2, 2],
            "group_width": 16,
            "bottleneck_ratio": 1,
            "se_ratio": 0.25,
        }
        base = RegNet(cfg, num_classes=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError("empty_stage_num must be non-negative")

        backbone = nn.Sequential(
            _ConvBnRelu(base.conv1, base.bn1),
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        stages = [backbone]
        for _ in range(self.empty_stage_num):
            stages.append(nn.Identity())
        self.stages = nn.ModuleList(stages)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class EfficientNetB0Split(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from models.efficientnet import EfficientNet, swish

        cfg = {
            "num_blocks": [1, 2, 2, 3, 3, 4, 1],
            "expansion": [1, 6, 6, 6, 6, 6, 6],
            "out_channels": [16, 24, 40, 80, 112, 192, 320],
            "kernel_size": [3, 3, 5, 3, 5, 5, 3],
            "stride": [1, 2, 2, 2, 1, 2, 1],
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
        }
        base = EfficientNet(cfg, num_classes=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)

        stem = _EfficientNetStem(base.conv1, base.bn1, swish)
        blocks = list(base.layers.children())
        stage1 = nn.Sequential(*blocks[0:1])
        stage2 = nn.Sequential(*blocks[1:3])
        stage3 = nn.Sequential(*blocks[3:5])
        stage4 = nn.Sequential(*blocks[5:])

        self.stage1 = nn.Sequential(stem, stage1)
        self.stage2 = stage2
        self.stage3 = stage3
        self.stage4 = stage4

    def get_submodules(self):
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalEfficientNetB0(nn.Module):
    def __init__(self, num_classes: int = 100, empty_stage_num: int = 3):
        super().__init__()
        from models.efficientnet import EfficientNet, swish

        cfg = {
            "num_blocks": [1, 2, 2, 3, 3, 4, 1],
            "expansion": [1, 6, 6, 6, 6, 6, 6],
            "out_channels": [16, 24, 40, 80, 112, 192, 320],
            "kernel_size": [3, 3, 5, 3, 5, 5, 3],
            "stride": [1, 2, 2, 2, 1, 2, 1],
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
        }
        base = EfficientNet(cfg, num_classes=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError("empty_stage_num must be non-negative")

        stem = _EfficientNetStem(base.conv1, base.bn1, swish)
        backbone = _EfficientNetBackbone(stem, base.layers)
        stages = [backbone]
        for _ in range(self.empty_stage_num):
            stages.append(nn.Identity())
        self.stages = nn.ModuleList(stages)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class DenseNet121Split(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        from models.densenet import Bottleneck, DenseNet

        base = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)

        self.stage1 = _DenseNetStage1(base.conv1, base.dense1, base.trans1)
        self.stage2 = _DenseNetStage(base.dense2, base.trans2)
        self.stage3 = _DenseNetStage(base.dense3, base.trans3)
        self.stage4 = _DenseNetStage4(base.dense4)

    def get_submodules(self):
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalDenseNet121(nn.Module):
    def __init__(self, num_classes: int = 100, empty_stage_num: int = 3):
        super().__init__()
        from models.densenet import Bottleneck, DenseNet

        base = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError("empty_stage_num must be non-negative")

        backbone = nn.Sequential(
            _DenseNetStage1(base.conv1, base.dense1, base.trans1),
            _DenseNetStage(base.dense2, base.trans2),
            _DenseNetStage(base.dense3, base.trans3),
            _DenseNetStage4(base.dense4),
        )
        stages = [backbone]
        for _ in range(self.empty_stage_num):
            stages.append(nn.Identity())
        self.stages = nn.ModuleList(stages)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels), eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        d = int(dim)
        self.dwconv = nn.Conv2d(d, d, kernel_size=7, padding=3, groups=d)
        self.norm = _LayerNorm2d(d)
        self.pwconv1 = nn.Conv2d(d, int(mlp_ratio) * d, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(int(mlp_ratio) * d, d, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual


class _ConvNeXtStage(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        blocks = []
        for _ in range(int(depth)):
            blocks.append(_ConvNeXtBlock(dim=int(dim)))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _ConvNeXtDownsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = _LayerNorm2d(int(in_dim))
        self.conv = nn.Conv2d(int(in_dim), int(out_dim), kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.conv(x)


class ConvNeXtTinyCifarSplit(nn.Module):
    def __init__(self, num_classes: int = 100, dims=(96, 192, 384, 768), depths=(3, 3, 9, 3)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)

        d0, d1, d2, d3 = (int(x) for x in dims)
        n0, n1, n2, n3 = (int(x) for x in depths)

        self.stem = nn.Sequential(
            nn.Conv2d(3, d0, kernel_size=3, stride=1, padding=1),
            _LayerNorm2d(d0),
        )

        self.stage1 = _ConvNeXtStage(d0, n0)
        self.down12 = _ConvNeXtDownsample(d0, d1)
        self.stage2 = _ConvNeXtStage(d1, n1)
        self.down23 = _ConvNeXtDownsample(d1, d2)
        self.stage3 = _ConvNeXtStage(d2, n2)
        self.down34 = _ConvNeXtDownsample(d2, d3)
        self.stage4 = _ConvNeXtStage(d3, n3)

        self.mod1 = nn.Sequential(self.stem, self.stage1)
        self.mod2 = nn.Sequential(self.down12, self.stage2)
        self.mod3 = nn.Sequential(self.down23, self.stage3)
        self.mod4 = nn.Sequential(self.down34, self.stage4)

    def get_submodules(self):
        return [self.mod1, self.mod2, self.mod3, self.mod4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalConvNeXtTinyCifar(nn.Module):
    def __init__(self, num_classes: int = 100, empty_stage_num: int = 3, dims=(96, 192, 384, 768), depths=(3, 3, 9, 3)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError("empty_stage_num must be non-negative")

        backbone = ConvNeXtTinyCifarSplit(num_classes=num_classes, dims=dims, depths=depths)
        full = nn.Sequential(*backbone.get_submodules())
        stages = [full]
        for _ in range(self.empty_stage_num):
            stages.append(nn.Identity())
        self.stages = nn.ModuleList(stages)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class _ViTPatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(int(in_chans), int(embed_dim), kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class _ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class _ViTBlocks(nn.Module):
    def __init__(self, depth: int, dim: int, num_heads: int):
        super().__init__()
        blocks = []
        for _ in range(int(depth)):
            blocks.append(_ViTBlock(dim=int(dim), num_heads=int(num_heads)))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _ViTTokensToMap(nn.Module):
    def __init__(self, grid_size: int, embed_dim: int):
        super().__init__()
        self.grid_size = int(grid_size)
        self.embed_dim = int(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        g = self.grid_size
        if n != g * g:
            raise ValueError("token count mismatch")
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, g, g)


class ViTTinyCifarSplit(nn.Module):
    def __init__(self, num_classes: int = 100, patch_size: int = 4, embed_dim: int = 384, depth: int = 8, num_heads: int = 6):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)

        self.patch = _ViTPatchEmbed(img_size=32, patch_size=int(patch_size), in_chans=3, embed_dim=int(embed_dim))
        self.grid_size = self.patch.grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch.num_patches, int(embed_dim)))
        self.drop = nn.Dropout(0.0)

        d = int(depth)
        splits = [d // 4, d // 4, d // 4, d - 3 * (d // 4)]
        self.blocks1 = _ViTBlocks(splits[0], embed_dim, num_heads)
        self.blocks2 = _ViTBlocks(splits[1], embed_dim, num_heads)
        self.blocks3 = _ViTBlocks(splits[2], embed_dim, num_heads)
        self.blocks4 = _ViTBlocks(splits[3], embed_dim, num_heads)
        self.to_map = _ViTTokensToMap(self.grid_size, embed_dim)

        self.stage1 = nn.Sequential(self.patch, _AddPos(self.pos_embed), self.drop, self.blocks1, self.to_map)
        self.stage2 = nn.Sequential(_MapToTokens(), self.blocks2, self.to_map)
        self.stage3 = nn.Sequential(_MapToTokens(), self.blocks3, self.to_map)
        self.stage4 = nn.Sequential(_MapToTokens(), self.blocks4, self.to_map)

    def get_submodules(self):
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class _AddPos(nn.Module):
    def __init__(self, pos_embed: nn.Parameter):
        super().__init__()
        self.pos_embed = pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


class _MapToTokens(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("expected NCHW feature map")
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class OriginalViTTinyCifar(nn.Module):
    def __init__(self, num_classes: int = 100, empty_stage_num: int = 3, patch_size: int = 4, embed_dim: int = 384, depth: int = 8, num_heads: int = 6):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError("empty_stage_num must be non-negative")

        backbone = ViTTinyCifarSplit(num_classes=num_classes, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        full = nn.Sequential(*backbone.get_submodules())
        stages = [full]
        for _ in range(self.empty_stage_num):
            stages.append(nn.Identity())
        self.stages = nn.ModuleList(stages)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out
