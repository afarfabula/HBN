import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, channels):
        super(Adapter, self).__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError('channels must be positive')

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


class GAPHead(nn.Module):
    def __init__(self, num_classes, hidden_dim=512):
        super(GAPHead, self).__init__()
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError('num_classes must be positive')
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')

        self.classifyheadweight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.fc1 = nn.LazyLinear(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class LinearGAPHead(nn.Module):
    def __init__(self, num_classes):
        super(LinearGAPHead, self).__init__()
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError('num_classes must be positive')
        self.classifyheadweight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.fc = nn.LazyLinear(self.num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SplitModule(object):
    def __init__(self, basemodel_name, num_classes=None, empty_stage_num=None):
        self.basemodel_name = str(basemodel_name).lower()
        self.num_classes = None if num_classes is None else int(num_classes)
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError('num_classes must be positive')
        self.empty_stage_num = None if empty_stage_num is None else int(empty_stage_num)
        if self.empty_stage_num is not None and self.empty_stage_num < 0:
            raise ValueError('empty_stage_num must be non-negative')

    def _use_baseline_like_head(self):
        n = self.basemodel_name
        return n in (
            'originalresnet',
            'original_resnet',
            'resnet18original',
            'resnet18_original',
            'resnet18-original',
        )

    def _build_basemodel(self):
        n = self.basemodel_name
        if n == 'resnet18':
            from .basemodel.resnet18 import ResNet18Split
            return ResNet18Split(num_classes=self.num_classes or 100)
        if n in ('resnetshared', 'resnet_shared', 'resnet18shared', 'resnet18_shared', 'resnet18-shared'):
            from .basemodel.resnet18 import ResNetShared
            if self.empty_stage_num is None:
                return ResNetShared(num_classes=self.num_classes or 100)
            return ResNetShared(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'resnetlargeshared',
            'resnet_large_shared',
            'resnet18largeshared',
            'resnet18_large_shared',
            'resnet18-large-shared',
        ):
            from .basemodel.resnet18 import ResNetLargeShared
            if self.empty_stage_num is None:
                return ResNetLargeShared(num_classes=self.num_classes or 100)
            return ResNetLargeShared(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'resnetxlshared',
            'resnet_xl_shared',
            'resnet18xlshared',
            'resnet18_xl_shared',
            'resnet18-xl-shared',
        ):
            from .basemodel.resnet18 import ResNetXLShared
            if self.empty_stage_num is None:
                return ResNetXLShared(num_classes=self.num_classes or 100)
            return ResNetXLShared(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'originalresnet',
            'original_resnet',
            'resnet18original',
            'resnet18_original',
            'resnet18-original',
        ):
            from .basemodel.resnet18 import OriginalResnet
            if self.empty_stage_num is None:
                return OriginalResnet(num_classes=self.num_classes or 100)
            return OriginalResnet(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        raise ValueError('Unknown basemodel: {}'.format(self.basemodel_name))

    def get_HBN_model_Config(self):
        base = self._build_basemodel()
        if not hasattr(base, 'get_submodules'):
            raise ValueError('Basemodel {} missing get_submodules()'.format(self.basemodel_name))
        modules = list(base.get_submodules())
        if len(modules) == 0:
            raise ValueError('Basemodel {} returned empty submodules'.format(self.basemodel_name))

        if not hasattr(base, 'num_classes'):
            raise ValueError('Basemodel {} missing num_classes'.format(self.basemodel_name))
        if not hasattr(base, 'input_shape'):
            raise ValueError('Basemodel {} missing input_shape'.format(self.basemodel_name))
        num_classes = int(base.num_classes)
        input_shape = tuple(int(x) for x in base.input_shape)
        if self._use_baseline_like_head():
            heads = [LinearGAPHead(num_classes) for _ in range(len(modules))]
        else:
            heads = [GAPHead(num_classes) for _ in range(len(modules))]

        x = torch.zeros((2,) + input_shape, dtype=torch.float32)
        out = x
        shapes = [tuple(out.shape[1:])]
        adapters = []
        with torch.no_grad():
            for i, m in enumerate(modules):
                out = m(out)
                if not torch.is_tensor(out):
                    raise TypeError('submodule[{}] returned {}; expected Tensor'.format(i, type(out)))
                shapes.append(tuple(out.shape[1:]))
                if i < len(modules) - 1:
                    c = int(out.shape[1])
                    if self._use_baseline_like_head():
                        adapters.append(nn.Identity())
                    else:
                        adapters.append(Adapter(c))

        return {
            'modules': modules,
            'adapter_list': adapters,
            'head_list': heads,
            'num_classes': num_classes,
            'intermediate_feature_shape_list': shapes,
        }

    def get_HBN_model_Condig(self):
        return self.get_HBN_model_Config()
