# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class HBNMerger(nn.Module):
    def __init__(self, modules, adapter_list, head_list, num_classes, intermediate_feature_shape_list):
        super(HBNMerger, self).__init__()
        if modules is None or len(modules) == 0:
            raise ValueError('modules must be a non-empty list of nn.Module')
        if adapter_list is None:
            adapter_list = []
        if head_list is None:
            head_list = []
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError('num_classes must be positive')
        if len(adapter_list) != len(modules) - 1:
            raise ValueError('adapter_list length must be len(modules)-1')
        if len(head_list) != len(modules):
            raise ValueError('head_list length must be len(modules)')
        if not isinstance(intermediate_feature_shape_list, (tuple, list)) or len(intermediate_feature_shape_list) == 0:
            raise ValueError('intermediate_feature_shape_list must be a non-empty list of shapes')

        for i, m in enumerate(modules):
            if not isinstance(m, nn.Module):
                raise TypeError('modules[{}] is not nn.Module: {}'.format(i, type(m)))
        for i, m in enumerate(adapter_list):
            if not isinstance(m, nn.Module):
                raise TypeError('adapter_list[{}] is not nn.Module: {}'.format(i, type(m)))
        for i, m in enumerate(head_list):
            if not isinstance(m, nn.Module):
                raise TypeError('head_list[{}] is not nn.Module: {}'.format(i, type(m)))

        self.modules_list = nn.ModuleList(list(modules))
        self.adapter_list = nn.ModuleList(list(adapter_list))
        self.head_list = nn.ModuleList(list(head_list))
        self.intermediate_feature_shape_list = [tuple(int(x) for x in s) for s in intermediate_feature_shape_list]
        self.force_unit_head_weight = False
        self.output_shape = self._infer_output_shapes_and_validate()

    def _head_weight(self, head, logits: torch.Tensor) -> torch.Tensor:
        if getattr(self, "force_unit_head_weight", False):
            return torch.ones((), dtype=logits.dtype, device=logits.device)
        w = head.classifyheadweight
        if not torch.is_tensor(w) or w.numel() != 1:
            raise ValueError('head_list classifyheadweight must be scalar Tensor/Parameter')
        return w.to(dtype=logits.dtype, device=logits.device)

    def _collect_head_weights(self, upto, dtype, device, normalize_head_weights: bool):
        ws = []
        for i in range(int(upto)):
            head = self.head_list[i]
            if getattr(self, "force_unit_head_weight", False):
                w = torch.ones((), dtype=dtype, device=device)
            else:
                w = head.classifyheadweight
                if not torch.is_tensor(w) or w.numel() != 1:
                    raise ValueError('head_list classifyheadweight must be scalar Tensor/Parameter')
                w = w.to(dtype=dtype, device=device)
            ws.append(w)
        if (not normalize_head_weights) or (not ws):
            return ws
        wv = torch.stack([w.reshape(()) for w in ws])
        wv = torch.clamp(wv, min=0)
        denom = torch.sum(wv)
        if float(denom.detach().cpu().item()) == 0.0:
            wv = torch.ones_like(wv) / float(int(wv.numel()))
        else:
            wv = wv / denom
        return [wv[i] for i in range(int(wv.numel()))]
    #检查submodule是否首尾衔接起来
    def _infer_output_shapes_and_validate(self):
        x0_shape = self.intermediate_feature_shape_list[0]
        x = torch.zeros((2,) + x0_shape, dtype=torch.float32)
        with torch.no_grad():
            out = x
            shapes = [tuple(out.shape[1:])]
            for i in range(len(self.modules_list)):
                out = self.modules_list[i](out)
                if isinstance(out, (tuple, list)):
                    raise TypeError('modules[{}] returned tuple/list; expected Tensor'.format(i))
                if not torch.is_tensor(out):
                    raise TypeError('modules[{}] returned {}; expected Tensor'.format(i, type(out)))
                shapes.append(tuple(out.shape[1:]))

                if i < len(self.adapter_list):
                    before = tuple(out.shape[1:])
                    out = self.adapter_list[i](out)
                    if isinstance(out, (tuple, list)):
                        raise TypeError('adapter_list[{}] returned tuple/list; expected Tensor'.format(i))
                    if not torch.is_tensor(out):
                        raise TypeError('adapter_list[{}] returned {}; expected Tensor'.format(i, type(out)))
                    after = tuple(out.shape[1:])
                    if after != before:
                        raise ValueError('adapter_list[{}] changed shape: {} -> {}'.format(i, before, after))

                logits = self.head_list[i](out)
                if not hasattr(self.head_list[i], 'classifyheadweight'):
                    raise AttributeError('head_list[{}] missing classifyheadweight'.format(i))
                w = getattr(self.head_list[i], 'classifyheadweight')
                if not torch.is_tensor(w):
                    raise TypeError('head_list[{}].classifyheadweight must be Tensor/Parameter'.format(i))
                if w.numel() != 1:
                    raise ValueError('head_list[{}].classifyheadweight must be scalar'.format(i))
                if isinstance(logits, (tuple, list)):
                    raise TypeError('head_list[{}] returned tuple/list; expected Tensor'.format(i))
                if not torch.is_tensor(logits):
                    raise TypeError('head_list[{}] returned {}; expected Tensor'.format(i, type(logits)))
                if logits.dim() != 2:
                    raise ValueError('head_list[{}] must output 2D logits, got shape {}'.format(i, tuple(logits.shape)))
                if int(logits.shape[1]) != self.num_classes:
                    raise ValueError('head_list[{}] logits dim mismatch: expected {}, got {}'.format(i, self.num_classes, int(logits.shape[1])))

            if len(self.intermediate_feature_shape_list) != len(shapes):
                raise ValueError('intermediate_feature_shape_list length mismatch: expected {}, got {}'.format(len(shapes), len(self.intermediate_feature_shape_list)))
            for i, (exp_s, got_s) in enumerate(zip(self.intermediate_feature_shape_list, shapes)):
                if tuple(exp_s) != tuple(got_s):
                    raise ValueError('shape mismatch at index {}: expected {}, got {}'.format(i, exp_s, got_s))

        return shapes[-1]

    def forward(self, x):
        out = x
        logits_list = []
        merged_logits = None
        for i in range(len(self.modules_list)):
            out = self.modules_list[i](out)
            if i < len(self.adapter_list):
                out = self.adapter_list[i](out)
            head = self.head_list[i]
            if not hasattr(head, 'classifyheadweight'):
                raise AttributeError('head_list[{}] missing classifyheadweight'.format(i))
            w = head.classifyheadweight
            if not torch.is_tensor(w) or w.numel() != 1:
                raise ValueError('head_list[{}].classifyheadweight must be scalar Tensor/Parameter'.format(i))
            logits = head(out)
            logits_list.append(logits)
            ww = self._head_weight(head, logits)
            merged_logits = logits * ww if merged_logits is None else merged_logits + logits * ww

        return merged_logits, logits_list
    #递归获得各阶段输出模块，训练时冻结
    def get_SubMergedModu(self):
        n_total = len(self.modules_list)
        sub_mergers = []
        for n in range(1, n_total):
            modules = [m for m in self.modules_list[:n]]
            adapters = [a for a in self.adapter_list[: max(0, n - 1)]]
            heads = [h for h in self.head_list[:n]]
            shapes = [s for s in self.intermediate_feature_shape_list[: n + 1]]
            sub_mergers.append(
                HBNMerger(
                    modules=modules,
                    adapter_list=adapters,
                    head_list=heads,
                    num_classes=self.num_classes,
                    intermediate_feature_shape_list=shapes,
                )
            )
            sub_mergers[-1].force_unit_head_weight = getattr(self, "force_unit_head_weight", False)
        return sub_mergers

    def forward_merged_logits(self, x, upto_stage, normalize_head_weights: bool = False):
        upto = int(upto_stage)
        if upto < 0 or upto > len(self.modules_list):
            raise ValueError('upto_stage out of range')
        if upto == 0:
            return torch.zeros((x.shape[0], self.num_classes), device=x.device, dtype=x.dtype)

        out = x
        merged_logits = None
        weights = self._collect_head_weights(upto, dtype=x.dtype, device=x.device, normalize_head_weights=bool(normalize_head_weights))
        for i in range(upto):
            out = self.modules_list[i](out)
            if i < len(self.adapter_list):
                out = self.adapter_list[i](out)
            head = self.head_list[i]
            logits = head(out)
            ww = weights[i]
            merged_logits = logits * ww if merged_logits is None else merged_logits + logits * ww
        return merged_logits

    def forward_stage_logits(self, x, stage_idx):
        s = int(stage_idx)
        if s < 1 or s > len(self.modules_list):
            raise ValueError('stage_idx out of range')
        out = x
        for i in range(s):
            out = self.modules_list[i](out)
            if i < len(self.adapter_list):
                out = self.adapter_list[i](out)
        return self.head_list[s - 1](out)
