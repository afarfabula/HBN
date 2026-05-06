import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.resnet import BasicBlock
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from models.resnet import BasicBlock


class _StemLayer1(nn.Module):
    def __init__(self, conv1, bn1, layer1):
        super(_StemLayer1, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.layer1 = layer1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out


class _StemOnly(nn.Module):
    def __init__(self, conv1, bn1):
        super(_StemOnly, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))


class _FeatureAdapter(nn.Module):
    def __init__(self, channels):
        super(_FeatureAdapter, self).__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(x + out, inplace=True)


class _StemLayer12(nn.Module):
    def __init__(self, conv1, bn1, layer1, layer2):
        super(_StemLayer12, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class _StemLayer123(nn.Module):
    def __init__(self, conv1, bn1, layer1, layer2, layer3):
        super(_StemLayer123, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class _StemLayer1234(nn.Module):
    def __init__(self, conv1, bn1, layer1, layer2, layer3, layer4):
        super(_StemLayer1234, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class ResNet18Split(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18Split, self).__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.in_planes = 64

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)

        self.stage1 = _StemLayer1(conv1, bn1, layer1)
        self.stage2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.stage3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.stage4 = self._make_layer(BasicBlock, 512, 2, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_submodules(self):
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x):
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class ResNet18SMHLSplit(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18SMHLSplit, self).__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.in_planes = 64

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        self.stage0 = _StemOnly(conv1, bn1)
        self.stage1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.stage2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.stage3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.stage4 = self._make_layer(BasicBlock, 512, 2, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_submodules(self):
        return [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x):
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalResnet5Split(nn.Module):
    def __init__(self, num_classes=100):
        super(OriginalResnet5Split, self).__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.in_planes = 64

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        self.stage0 = _StemOnly(conv1, bn1)
        self.stage1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.stage2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.stage3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.stage4 = nn.Sequential(
            self._make_layer(BasicBlock, 512, 2, stride=2),
            _FeatureAdapter(512),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_submodules(self):
        return [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x):
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class _EmptyStage(nn.Module):
    def __init__(self):
        super(_EmptyStage, self).__init__()

    def forward(self, x):
        return x


class ResNetShared(nn.Module):
    def __init__(self, num_classes=100, empty_stage_num=3):
        super(ResNetShared, self).__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.in_planes = 64
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError('empty_stage_num must be non-negative')

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)

        stages = [_StemLayer1(conv1, bn1, layer1)]
        for _ in range(self.empty_stage_num):
            stages.append(_EmptyStage())
        self.stages = nn.ModuleList(stages)

        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x):
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class ResNetLargeShared(nn.Module):
    def __init__(self, num_classes=100, empty_stage_num=3):
        super(ResNetLargeShared, self).__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.in_planes = 64
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError('empty_stage_num must be non-negative')

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)

        stages = [_StemLayer12(conv1, bn1, layer1, layer2)]
        for _ in range(self.empty_stage_num):
            stages.append(_EmptyStage())
        self.stages = nn.ModuleList(stages)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x):
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class ResNetXLShared(nn.Module):
    def __init__(self, num_classes=100, empty_stage_num=3):
        super(ResNetXLShared, self).__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.in_planes = 64
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError('empty_stage_num must be non-negative')

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

        stages = [_StemLayer123(conv1, bn1, layer1, layer2, layer3)]
        for _ in range(self.empty_stage_num):
            stages.append(_EmptyStage())
        self.stages = nn.ModuleList(stages)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x):
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


class OriginalResnet(nn.Module):
    def __init__(self, num_classes=100, empty_stage_num=3):
        super(OriginalResnet, self).__init__()
        self.num_classes = int(num_classes)
        self.input_shape = (3, 32, 32)
        self.in_planes = 64
        self.empty_stage_num = int(empty_stage_num)
        if self.empty_stage_num < 0:
            raise ValueError('empty_stage_num must be non-negative')

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        stages = [_StemLayer1234(conv1, bn1, layer1, layer2, layer3, layer4)]
        for _ in range(self.empty_stage_num):
            stages.append(_EmptyStage())
        self.stages = nn.ModuleList(stages)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_submodules(self):
        return list(self.stages)

    def forward(self, x):
        out = x
        for m in self.get_submodules():
            out = m(out)
        return out


def _smoke():
    m = ResNet18Split()
    stages = m.get_submodules()
    assert isinstance(stages, list) and len(stages) == 4
    x = torch.randn(2, 3, 32, 32)
    out = x
    shapes = []
    for s in stages:
        out = s(out)
        shapes.append(tuple(out.shape))
    assert shapes == [(2, 64, 32, 32), (2, 128, 16, 16), (2, 256, 8, 8), (2, 512, 4, 4)]

    m2 = ResNetShared(empty_stage_num=3)
    stages2 = m2.get_submodules()
    assert isinstance(stages2, list) and len(stages2) == 4
    out2 = x
    shapes2 = []
    for s in stages2:
        out2 = s(out2)
        shapes2.append(tuple(out2.shape))
    assert shapes2 == [(2, 64, 32, 32), (2, 64, 32, 32), (2, 64, 32, 32), (2, 64, 32, 32)]

    m3 = ResNetShared(empty_stage_num=0)
    stages3 = m3.get_submodules()
    assert isinstance(stages3, list) and len(stages3) == 1
    out3 = x
    for s in stages3:
        out3 = s(out3)
    assert tuple(out3.shape) == (2, 64, 32, 32)

    m4 = ResNetShared(empty_stage_num=5)
    stages4 = m4.get_submodules()
    assert isinstance(stages4, list) and len(stages4) == 6
    out4 = x
    for s in stages4:
        out4 = s(out4)
    assert tuple(out4.shape) == (2, 64, 32, 32)

    m5 = ResNetLargeShared(empty_stage_num=3)
    stages5 = m5.get_submodules()
    assert isinstance(stages5, list) and len(stages5) == 4
    out5 = x
    shapes5 = []
    for s in stages5:
        out5 = s(out5)
        shapes5.append(tuple(out5.shape))
    assert shapes5 == [(2, 128, 16, 16), (2, 128, 16, 16), (2, 128, 16, 16), (2, 128, 16, 16)]

    m6 = ResNetLargeShared(empty_stage_num=0)
    stages6 = m6.get_submodules()
    assert isinstance(stages6, list) and len(stages6) == 1
    out6 = x
    for s in stages6:
        out6 = s(out6)
    assert tuple(out6.shape) == (2, 128, 16, 16)

    m7 = ResNetXLShared(empty_stage_num=3)
    stages7 = m7.get_submodules()
    assert isinstance(stages7, list) and len(stages7) == 4
    out7 = x
    shapes7 = []
    for s in stages7:
        out7 = s(out7)
        shapes7.append(tuple(out7.shape))
    assert shapes7 == [(2, 256, 8, 8), (2, 256, 8, 8), (2, 256, 8, 8), (2, 256, 8, 8)]

    m8 = ResNetXLShared(empty_stage_num=0)
    stages8 = m8.get_submodules()
    assert isinstance(stages8, list) and len(stages8) == 1
    out8 = x
    for s in stages8:
        out8 = s(out8)
    assert tuple(out8.shape) == (2, 256, 8, 8)

    m9 = OriginalResnet(empty_stage_num=3)
    stages9 = m9.get_submodules()
    assert isinstance(stages9, list) and len(stages9) == 4
    out9 = x
    shapes9 = []
    for s in stages9:
        out9 = s(out9)
        shapes9.append(tuple(out9.shape))
    assert shapes9 == [(2, 512, 4, 4), (2, 512, 4, 4), (2, 512, 4, 4), (2, 512, 4, 4)]

    m10 = OriginalResnet(empty_stage_num=0)
    stages10 = m10.get_submodules()
    assert isinstance(stages10, list) and len(stages10) == 1
    out10 = x
    for s in stages10:
        out10 = s(out10)
    assert tuple(out10.shape) == (2, 512, 4, 4)


if __name__ == '__main__':
    _smoke()
