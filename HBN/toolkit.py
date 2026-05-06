import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, channels, hidden_multiplier=1.0):
        super(Adapter, self).__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError('channels must be positive')
        m = float(hidden_multiplier)
        if m <= 0:
            raise ValueError('hidden_multiplier must be positive')
        hidden_c = max(1, int(round(c * m)))

        self.conv1 = nn.Conv2d(c, hidden_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_c)
        self.conv2 = nn.Conv2d(hidden_c, c, kernel_size=3, stride=1, padding=1, bias=False)
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
    def __init__(self, basemodel_name, num_classes=None, empty_stage_num=None, head_mode='auto', adapter_multiplier=1.0):
        self.basemodel_name = str(basemodel_name).lower()
        self.num_classes = None if num_classes is None else int(num_classes)
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError('num_classes must be positive')
        self.empty_stage_num = None if empty_stage_num is None else int(empty_stage_num)
        if self.empty_stage_num is not None and self.empty_stage_num < 0:
            raise ValueError('empty_stage_num must be non-negative')
        self.head_mode = str(head_mode or 'auto').lower()
        if self.head_mode not in ('auto', 'baseline', 'hbn'):
            raise ValueError("head_mode must be one of: auto, baseline, hbn")
        self.adapter_multiplier = float(adapter_multiplier)
        if self.adapter_multiplier <= 0:
            raise ValueError('adapter_multiplier must be positive')

    def _use_baseline_like_head(self):
        if self.head_mode == 'baseline':
            return True
        if self.head_mode == 'hbn':
            return False
        n = self.basemodel_name
        return n in (
            'originalresnet',
            'original_resnet',
            'resnet18original',
            'resnet18_original',
            'resnet18-original',
            'originalpreactresnet18',
            'original_preactresnet18',
            'preactresnet18original',
            'preactresnet18_original',
            'preactresnet18-original',
            'originalresnext29_32x4d',
            'original_resnext29_32x4d',
            'resnext29_32x4doriginal',
            'resnext29_32x4d_original',
            'resnext29_32x4d-original',
            'originalregnetx_200mf',
            'original_regnetx_200mf',
            'regnetx_200mforiginal',
            'regnetx_200mf_original',
            'regnetx_200mf-original',
            'originalregnety_400mf',
            'original_regnety_400mf',
            'regnety_400mforiginal',
            'regnety_400mf_original',
            'regnety_400mf-original',
            'originalefficientnetb0',
            'original_efficientnetb0',
            'efficientnetb0original',
            'efficientnetb0_original',
            'efficientnetb0-original',
            'originaldensenet121',
            'original_densenet121',
            'densenet121original',
            'densenet121_original',
            'densenet121-original',
            'originalconvnexttinycifar',
            'original_convnext_tiny_cifar',
            'convnext_tiny_cifaroriginal',
            'convnext_tiny_cifar_original',
            'convnext_tiny_cifar-original',
            'originalvittinycifar',
            'original_vit_tiny_cifar',
            'vit_tiny_cifaroriginal',
            'vit_tiny_cifar_original',
            'vit_tiny_cifar-original',
        )

    def _build_basemodel(self):
        n = self.basemodel_name
        if n == 'resnet18':
            from .basemodel.resnet18 import ResNet18Split
            return ResNet18Split(num_classes=self.num_classes or 100)
        if n in ('resnet18_smhl', 'resnet18-smhl', 'smhlresnet18', 'smhl_resnet18'):
            from .basemodel.resnet18 import ResNet18SMHLSplit
            return ResNet18SMHLSplit(num_classes=self.num_classes or 100)
        if n in ('originalresnet_split5', 'originalresnet-split5', 'original_resnet_split5', 'originalresnet5split'):
            from .basemodel.resnet18 import OriginalResnet5Split
            return OriginalResnet5Split(num_classes=self.num_classes or 100)
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
        if n in (
            'preactresnet18',
            'preact_resnet18',
            'preact-resnet18',
        ):
            from .basemodel.cifar_backbones import PreActResNet18Split
            return PreActResNet18Split(num_classes=self.num_classes or 100)
        if n in (
            'originalpreactresnet18',
            'original_preactresnet18',
            'preactresnet18original',
            'preactresnet18_original',
            'preactresnet18-original',
        ):
            from .basemodel.cifar_backbones import OriginalPreActResNet18
            if self.empty_stage_num is None:
                return OriginalPreActResNet18(num_classes=self.num_classes or 100)
            return OriginalPreActResNet18(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'resnext29_32x4d',
            'resnext29(32x4d)',
            'resnext29-32x4d',
            'resnext29_32x4d_split',
            'resnext29_32x4d-split',
        ):
            from .basemodel.cifar_backbones import ResNeXt29_32x4dSplit
            return ResNeXt29_32x4dSplit(num_classes=self.num_classes or 100)
        if n in (
            'originalresnext29_32x4d',
            'original_resnext29_32x4d',
            'resnext29_32x4doriginal',
            'resnext29_32x4d_original',
            'resnext29_32x4d-original',
        ):
            from .basemodel.cifar_backbones import OriginalResNeXt29_32x4d
            if self.empty_stage_num is None:
                return OriginalResNeXt29_32x4d(num_classes=self.num_classes or 100)
            return OriginalResNeXt29_32x4d(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'regnetx_200mf',
            'regnetx200mf',
            'regnetx_200mf_split',
            'regnetx_200mf-split',
        ):
            from .basemodel.cifar_backbones import RegNetX_200MFSplit
            return RegNetX_200MFSplit(num_classes=self.num_classes or 100)
        if n in (
            'originalregnetx_200mf',
            'original_regnetx_200mf',
            'regnetx_200mforiginal',
            'regnetx_200mf_original',
            'regnetx_200mf-original',
        ):
            from .basemodel.cifar_backbones import OriginalRegNetX_200MF
            if self.empty_stage_num is None:
                return OriginalRegNetX_200MF(num_classes=self.num_classes or 100)
            return OriginalRegNetX_200MF(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'regnety_400mf',
            'regnety400mf',
            'regnety_400mf_split',
            'regnety_400mf-split',
        ):
            from .basemodel.cifar_backbones import RegNetY_400MFSplit
            return RegNetY_400MFSplit(num_classes=self.num_classes or 100)
        if n in (
            'originalregnety_400mf',
            'original_regnety_400mf',
            'regnety_400mforiginal',
            'regnety_400mf_original',
            'regnety_400mf-original',
        ):
            from .basemodel.cifar_backbones import OriginalRegNetY_400MF
            if self.empty_stage_num is None:
                return OriginalRegNetY_400MF(num_classes=self.num_classes or 100)
            return OriginalRegNetY_400MF(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'efficientnetb0',
            'efficientnet_b0',
            'efficientnet-b0',
        ):
            from .basemodel.cifar_backbones import EfficientNetB0Split
            return EfficientNetB0Split(num_classes=self.num_classes or 100)
        if n in (
            'originalefficientnetb0',
            'original_efficientnetb0',
            'efficientnetb0original',
            'efficientnetb0_original',
            'efficientnetb0-original',
        ):
            from .basemodel.cifar_backbones import OriginalEfficientNetB0
            if self.empty_stage_num is None:
                return OriginalEfficientNetB0(num_classes=self.num_classes or 100)
            return OriginalEfficientNetB0(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'densenet121',
            'densenet_121',
            'densenet-121',
        ):
            from .basemodel.cifar_backbones import DenseNet121Split
            return DenseNet121Split(num_classes=self.num_classes or 100)
        if n in (
            'originaldensenet121',
            'original_densenet121',
            'densenet121original',
            'densenet121_original',
            'densenet121-original',
        ):
            from .basemodel.cifar_backbones import OriginalDenseNet121
            if self.empty_stage_num is None:
                return OriginalDenseNet121(num_classes=self.num_classes or 100)
            return OriginalDenseNet121(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'convnext_tiny_cifar',
            'convnexttinycifar',
            'convnext_tiny',
            'convnexttiny',
        ):
            from .basemodel.cifar_backbones import ConvNeXtTinyCifarSplit
            return ConvNeXtTinyCifarSplit(num_classes=self.num_classes or 100)
        if n in (
            'originalconvnexttinycifar',
            'original_convnext_tiny_cifar',
            'convnext_tiny_cifaroriginal',
            'convnext_tiny_cifar_original',
            'convnext_tiny_cifar-original',
        ):
            from .basemodel.cifar_backbones import OriginalConvNeXtTinyCifar
            if self.empty_stage_num is None:
                return OriginalConvNeXtTinyCifar(num_classes=self.num_classes or 100)
            return OriginalConvNeXtTinyCifar(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
        if n in (
            'vit_tiny_cifar',
            'vittinycifar',
            'vit_tiny',
            'vittiny',
        ):
            from .basemodel.cifar_backbones import ViTTinyCifarSplit
            return ViTTinyCifarSplit(num_classes=self.num_classes or 100)
        if n in (
            'originalvittinycifar',
            'original_vit_tiny_cifar',
            'vit_tiny_cifaroriginal',
            'vit_tiny_cifar_original',
            'vit_tiny_cifar-original',
        ):
            from .basemodel.cifar_backbones import OriginalViTTinyCifar
            if self.empty_stage_num is None:
                return OriginalViTTinyCifar(num_classes=self.num_classes or 100)
            return OriginalViTTinyCifar(num_classes=self.num_classes or 100, empty_stage_num=self.empty_stage_num)
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
        use_identity_side_adapters = self.basemodel_name in (
            'originalresnet_split5',
            'originalresnet-split5',
            'original_resnet_split5',
            'originalresnet5split',
        )
        with torch.no_grad():
            for i, m in enumerate(modules):
                out = m(out)
                if not torch.is_tensor(out):
                    raise TypeError('submodule[{}] returned {}; expected Tensor'.format(i, type(out)))
                shapes.append(tuple(out.shape[1:]))
                if i < len(modules) - 1:
                    c = int(out.shape[1])
                    if self._use_baseline_like_head() or use_identity_side_adapters:
                        adapters.append(nn.Identity())
                    else:
                        adapters.append(Adapter(c, hidden_multiplier=self.adapter_multiplier))

        return {
            'modules': modules,
            'adapter_list': adapters,
            'head_list': heads,
            'num_classes': num_classes,
            'intermediate_feature_shape_list': shapes,
        }

    def get_HBN_model_Condig(self):
        return self.get_HBN_model_Config()
