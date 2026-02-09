import math

import torch
import torch.nn.functional as F


class SampleWeights(object):
    def __init__(self, num_samples, device='cpu', dtype=torch.float64):
        self.log_w = torch.zeros(num_samples, dtype=dtype, device=device)
        self._normalize_inplace()

    def _normalize_inplace(self):
        m = torch.max(self.log_w)
        self.log_w = self.log_w - (m + torch.log(torch.sum(torch.exp(self.log_w - m))))

    def batch_weights(self, indices, device=None, dtype=torch.float32):
        w = torch.exp(self.log_w[indices])
        if device is not None:
            w = w.to(device)
        return w.to(dtype)

    def weighted_error(self, errors01):
        w = torch.exp(self.log_w)
        return torch.sum(w * errors01.to(w.dtype)).item()

    def update(self, errors01, alpha, alpha_clip=8.0):
        a = float(alpha)
        if a > alpha_clip:
            a = alpha_clip
        if a < -alpha_clip:
            a = -alpha_clip
        self.log_w = self.log_w + a * errors01.to(self.log_w.dtype)
        self._normalize_inplace()


def weighted_cross_entropy(logits, targets, sample_w):
    losses = F.cross_entropy(logits, targets, reduction='none')
    w = sample_w.to(losses.dtype)
    denom = torch.sum(w)
    if denom.item() == 0:
        return torch.mean(losses)
    return torch.sum(losses * w) / denom


def compute_alpha_samme(epsilon, num_classes, eps_clip=1e-12, alpha_clip=8.0):
    eps = float(epsilon)
    if eps < eps_clip:
        eps = eps_clip
    if eps > 1.0 - eps_clip:
        eps = 1.0 - eps_clip
    alpha = math.log((1.0 - eps) / eps) + math.log(max(1, int(num_classes) - 1))
    if alpha > alpha_clip:
        alpha = alpha_clip
    if alpha < -alpha_clip:
        alpha = -alpha_clip
    return alpha, eps


def compute_alpha_paper(epsilon, eps_clip=1e-12, alpha_clip=8.0):
    eps = float(epsilon)
    if eps < eps_clip:
        eps = eps_clip
    if eps > 1.0 - eps_clip:
        eps = 1.0 - eps_clip
    alpha = 0.5 * math.log((1.0 - eps) / eps)
    if alpha > alpha_clip:
        alpha = alpha_clip
    if alpha < -alpha_clip:
        alpha = -alpha_clip
    return alpha, eps


def set_trainable_for_stage(model, stage_idx):
    for p in model.parameters():
        p.requires_grad = False

    stage_i = int(stage_idx) - 1
    for p in model.modules_list[stage_i].parameters():
        p.requires_grad = True
    if stage_i < len(model.adapter_list):
        for p in model.adapter_list[stage_i].parameters():
            p.requires_grad = True
    for p in model.head_list[stage_i].parameters():
        p.requires_grad = True
    if hasattr(model.head_list[stage_i], 'classifyheadweight'):
        model.head_list[stage_i].classifyheadweight.requires_grad = False

    for i in range(stage_i):
        model.modules_list[i].eval()
        if i < len(model.adapter_list):
            model.adapter_list[i].eval()
        model.head_list[i].eval()

    model.modules_list[stage_i].train()
    if stage_i < len(model.adapter_list):
        model.adapter_list[stage_i].train()
    model.head_list[stage_i].train()
