
import csv
import json
import logging
import math
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_metrics(logits, labels, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    fn = ((1 - preds) * labels).sum(0)

    tp_s, fp_s, fn_s = tp.sum(), fp.sum(), fn.sum()
    micro_p = tp_s / (tp_s + fp_s).clamp(min=1)
    micro_r = tp_s / (tp_s + fn_s).clamp(min=1)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r).clamp(min=1e-8)

    per_p = tp / (tp + fp).clamp(min=1)
    per_r = tp / (tp + fn).clamp(min=1)
    per_f1 = 2 * per_p * per_r / (per_p + per_r).clamp(min=1e-8)
    has = (tp + fn) > 0
    macro_f1 = per_f1[has].mean() if has.any() else torch.tensor(0.0)
    macro_p = per_p[has].mean() if has.any() else torch.tensor(0.0)
    macro_r = per_r[has].mean() if has.any() else torch.tensor(0.0)

    return {
        'micro_f1': micro_f1.item(), 'micro_p': micro_p.item(),
        'micro_r': micro_r.item(), 'macro_f1': macro_f1.item(),
        'macro_p': macro_p.item(), 'macro_r': macro_r.item(),
        'n_classes_with_support': int(has.sum()),
    }


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps,
                                    min_lr_ratio=0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state, path):
    torch.save(state, path)
    logger.info(f"  checkpoint -> {path}")


def load_component(path, model, component=None):
    """Load a checkpoint into a model or a specific submodule.

    Parameters
    ----------
    path : str
        Path to checkpoint file.
    model : nn.Module
        The full model.
    component : str, optional
        If given, load state dict into ``getattr(model, component)``
        instead of the full model.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    target = getattr(model, component) if component else model
    target.load_state_dict(state)
    label = component or 'full model'
    logger.info(f"  Loaded {label} from {os.path.basename(path)}")
    return ckpt


def load_best_or_last(ckpt_dir, phase_name, model, component=None):
    """Load best (or last) checkpoint for a phase.

    When ``component`` is given, loads into that submodule only.
    Returns True if a checkpoint was found and loaded.
    """
    for prefix in ('best', 'last'):
        path = os.path.join(ckpt_dir, f'{prefix}_{phase_name}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu',
                              weights_only=False)
            state = ckpt['model_state_dict']
            target = getattr(model, component) if component else model
            target.load_state_dict(state)
            logger.info(f"  Loaded {os.path.basename(path)}"
                        f"{f' -> {component}' if component else ''}")
            return True
    return False# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row):
        self._writer.writerow({k: row.get(k, '') for k in self.fieldnames})
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------

def calibrate_temperature(model, loader, device, unpack_fn,
                          lr=0.01, max_iter=200):
    """Learn a single temperature scalar T that minimises BCE NLL on val set.

    Logits are divided by T before sigmoid:  p = sigmoid(logit / T).
    Returns (T, logits, labels) so callers can compute FDR thresholds.
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            lo, la = unpack_fn(model, batch, device)
            all_logits.append(lo.cpu())
            all_labels.append(la.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Optimise T on CPU
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()  # positive
        loss = F.binary_cross_entropy_with_logits(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = log_T.exp().item()
    final_loss = F.binary_cross_entropy_with_logits(
        logits / T, labels).item()
    uncal_loss = F.binary_cross_entropy_with_logits(logits, labels).item()

    logger.info(f"  Temperature calibration: T={T:.4f}")
    logger.info(f"    BCE before: {uncal_loss:.5f}  after: {final_loss:.5f}")
    return T, logits, labels


def find_fdr_thresholds(logits, labels, temperature,
                        target_fdrs=(0.10, 0.20), n_steps=500):
    """Find score thresholds that achieve target FDR levels.

    FDR = FP / (FP + TP) = 1 - micro-precision.
    Sweeps thresholds on calibrated probabilities from low to high,
    picks the lowest threshold where precision >= 1 - target_fdr.
    Returns dict {fdr_value: threshold}.
    """
    probs = torch.sigmoid(logits / temperature)
    thresholds = torch.linspace(0.01, 0.99, n_steps)

    # Precompute precision at each threshold
    precisions = []
    for t in thresholds:
        preds = (probs >= t.item()).float()
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        prec = (tp / (tp + fp)).item() if (tp + fp) > 0 else 1.0
        precisions.append(prec)

    results = {}
    for target_fdr in target_fdrs:
        target_precision = 1.0 - target_fdr
        best_t = None

        # Scan low→high: first threshold where precision >= target
        for t, prec in zip(thresholds, precisions):
            if prec >= target_precision:
                best_t = t.item()
                break

        if best_t is None:
            # Precision never reaches target — use highest threshold
            best_t = thresholds[-1].item()
            logger.warning(f"  FDR {target_fdr*100:.0f}%: could not reach "
                           f"target precision {target_precision:.2f}, "
                           f"using threshold={best_t:.4f}")

        # Report actual metrics at chosen threshold
        preds = (probs >= best_t).float()
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        actual_prec = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0
        actual_recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0
        actual_fdr = 1.0 - actual_prec

        logger.info(f"  FDR {target_fdr*100:.0f}%: threshold={best_t:.4f}  "
                     f"actual_FDR={actual_fdr:.4f}  "
                     f"precision={actual_prec:.4f}  recall={actual_recall:.4f}")
        results[target_fdr] = best_t

    return results


def find_blocked_classes(logits, labels, temperature,
                        min_precision=0.6, min_support=1):
    """Identify classes with poor validation precision to block at inference.

    A class is blocked if it has at least ``min_support`` positive samples
    AND its precision falls below ``min_precision``.  Classes with zero
    support (never seen in validation) are left unblocked.

    Returns sorted list of blocked class indices.
    """
    probs = torch.sigmoid(logits / temperature)
    # Use 0.5 threshold for precision calculation (standard decision boundary)
    preds = (probs >= 0.5).float()
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    fn = ((1 - preds) * labels).sum(0)
    support = tp + fn

    per_p = tp / (tp + fp).clamp(min=1)

    blocked = []
    for i in range(len(per_p)):
        if support[i] >= min_support and per_p[i] < min_precision:
            blocked.append(int(i))

    if blocked:
        logger.info(f"  Blocked {len(blocked)} classes with precision < "
                    f"{min_precision} (min_support={min_support})")
    else:
        logger.info(f"  No classes blocked (all meet precision >= "
                    f"{min_precision})")
    return sorted(blocked)


def save_calibration(path, temperature, hosts, model_config,
                     threshold=0.5, fdr_thresholds=None,
                     blocked_classes=None):
    """Save calibration.json alongside checkpoints."""
    data = {
        'temperature': temperature,
        'threshold': threshold,
        'hosts': hosts.tolist() if hasattr(hosts, 'tolist') else list(hosts),
        'model_config': model_config,
    }
    if fdr_thresholds:
        data['fdr_thresholds'] = {
            f"fdr_{int(fdr*100):02d}": round(t, 5)
            for fdr, t in fdr_thresholds.items()
        }
    if blocked_classes is not None:
        data['blocked_classes'] = blocked_classes
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"  calibration saved -> {path}")


def run_calibration(model, loader, device, unpack_fn, run_dir, hosts,
                    model_config, eval_threshold, min_val_precision,
                    min_val_support):
    """Run temperature calibration, FDR thresholds, and class blocking."""
    T, logits, labels = calibrate_temperature(
        model, loader, device, unpack_fn)
    fdr_thresholds = find_fdr_thresholds(logits, labels, T)
    blocked = find_blocked_classes(
        logits, labels, T,
        min_precision=min_val_precision,
        min_support=min_val_support,
    ) if min_val_precision > 0 else []
    save_calibration(
        os.path.join(run_dir, 'calibration.json'),
        temperature=T, hosts=hosts, model_config=model_config,
        threshold=eval_threshold, fdr_thresholds=fdr_thresholds,
        blocked_classes=blocked,
    )
