import torch
import torch.nn.functional as F


def eval_ensemble(model, loader, upto_stage, max_batches=0, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model.forward_merged_logits(inputs, upto_stage=upto_stage)
            loss = F.cross_entropy(logits, targets)
            loss_sum += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            if max_batches and (batch_idx + 1) >= max_batches:
                break
    avg_loss = loss_sum / (batch_idx + 1)
    acc = 100.0 * correct / total if total else 0.0
    return avg_loss, acc


def eval_candidate_sum(model, loader, stage_idx, alpha, num_classes, max_batches=0, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    a = float(alpha)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            if stage_idx > 1:
                logits_prev = model.forward_merged_logits(inputs, upto_stage=stage_idx - 1)
            else:
                logits_prev = torch.zeros((inputs.size(0), num_classes), device=device, dtype=torch.float32)
            logits_t = model.forward_stage_logits(inputs, stage_idx=stage_idx)
            logits = logits_prev + logits_t * a
            loss = F.cross_entropy(logits, targets)
            loss_sum += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            if max_batches and (batch_idx + 1) >= max_batches:
                break
    avg_loss = loss_sum / (batch_idx + 1)
    acc = 100.0 * correct / total if total else 0.0
    return avg_loss, acc

