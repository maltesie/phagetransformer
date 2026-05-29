
import csv
import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Biologically-informed codon embedding initialization
# ---------------------------------------------------------------------------

_BASES = 'ACGT'
_CODONS = [a + b + c for a in _BASES for b in _BASES for c in _BASES]

_GENETIC_CODE = {
    'AAA': 'K', 'AAC': 'N', 'AAG': 'K', 'AAT': 'N',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AGA': 'R', 'AGC': 'S', 'AGG': 'R', 'AGT': 'S',
    'ATA': 'I', 'ATC': 'I', 'ATG': 'M', 'ATT': 'I',
    'CAA': 'Q', 'CAC': 'H', 'CAG': 'Q', 'CAT': 'H',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'GAA': 'E', 'GAC': 'D', 'GAG': 'E', 'GAT': 'D',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'TAA': '*', 'TAC': 'Y', 'TAG': '*', 'TAT': 'Y',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TGA': '*', 'TGC': 'C', 'TGG': 'W', 'TGT': 'C',
    'TTA': 'L', 'TTC': 'F', 'TTG': 'L', 'TTT': 'F',
}

CODON_TABLE = [_GENETIC_CODE[c] for c in _CODONS]

_AA_PROPERTIES = {
    #        hydro    MW     vol     pI   charge  arom
    'A': [ 1.8,   89.1,   88.6,  6.00,  0.0,  0.0],
    'R': [-4.5,  174.2,  173.4, 10.76,  1.0,  0.0],
    'N': [-3.5,  132.1,  114.1,  5.41,  0.0,  0.0],
    'D': [-3.5,  133.1,  111.1,  2.77, -1.0,  0.0],
    'C': [ 2.5,  121.2,  108.5,  5.07,  0.0,  0.0],
    'E': [-3.5,  147.1,  138.4,  3.22, -1.0,  0.0],
    'Q': [-3.5,  146.2,  143.8,  5.65,  0.0,  0.0],
    'G': [-0.4,   75.0,   60.1,  5.97,  0.0,  0.0],
    'H': [-3.2,  155.2,  153.2,  7.59,  0.1,  0.5],
    'I': [ 4.5,  131.2,  166.7,  6.02,  0.0,  0.0],
    'L': [ 3.8,  131.2,  166.7,  5.98,  0.0,  0.0],
    'K': [-3.9,  146.2,  168.6,  9.74,  1.0,  0.0],
    'M': [ 1.9,  149.2,  162.9,  5.74,  0.0,  0.0],
    'F': [ 2.8,  165.2,  189.9,  5.48,  0.0,  1.0],
    'P': [-1.6,  115.1,  112.7,  6.30,  0.0,  0.0],
    'S': [-0.8,  105.1,   89.0,  5.68,  0.0,  0.0],
    'T': [-0.7,  119.1,  116.1,  5.60,  0.0,  0.0],
    'W': [-0.9,  204.2,  227.8,  5.89,  0.0,  1.0],
    'Y': [-1.3,  181.2,  193.6,  5.66,  0.0,  1.0],
    'V': [ 4.2,  117.1,  140.0,  5.96,  0.0,  0.0],
    '*': [ 0.0,    0.0,    0.0,  0.00,  0.0,  0.0],
}


def build_codon_embeddings(vocab_size: int = 65, embed_dim: int = 64,
                           codon_offset: int = 0,
                           seed: int = 42) -> torch.Tensor:
    """Build initial codon embeddings where synonymous codons cluster.

    Each amino acid gets a base vector derived from biochemical properties
    (hydrophobicity, MW, volume, pI, charge, aromaticity) projected to
    ``embed_dim`` via a fixed random matrix. Synonymous codons share their
    amino acid's base vector plus small noise to break exact degeneracy.

    Args:
        vocab_size: total vocabulary size (codons + special tokens).
        embed_dim: embedding dimension.
        codon_offset: index of the first codon in the vocabulary.
            0 when codons occupy indices 0-63 (our model),
            2 when PAD=0, UNK=1, codons at 2-65 (other model).
        seed: random seed for reproducible projection.

    Returns a (vocab_size, embed_dim) tensor suitable for assigning to
    ``nn.Embedding.weight.data``.
    """
    rng = np.random.RandomState(seed)
    n_props = 6

    props = np.array([_AA_PROPERTIES[aa] for aa in sorted(_AA_PROPERTIES)])
    aa_list = sorted(_AA_PROPERTIES.keys())
    mean = props.mean(axis=0)
    std = props.std(axis=0).clip(min=1e-8)
    props = (props - mean) / std
    aa_to_vec = {aa: props[i] for i, aa in enumerate(aa_list)}

    projection = rng.randn(n_props, embed_dim).astype(np.float32)
    projection *= 0.02 / np.sqrt(n_props)

    embeddings = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    for codon_idx in range(64):
        aa = CODON_TABLE[codon_idx]
        base = aa_to_vec[aa] @ projection
        noise = rng.randn(embed_dim).astype(np.float32) * 0.002
        embeddings[codon_offset + codon_idx] = base + noise

    return torch.from_numpy(embeddings)


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


def _optimize_temperature(logits, labels, lr=0.01, max_iter=200):
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = F.binary_cross_entropy_with_logits(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_T.exp().item()


def calibrate_temperature_split(model, loader, device, unpack_fn,
                                n_extra_classes=1, lr=0.01, max_iter=200):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            lo, la = unpack_fn(model, batch, device)
            all_logits.append(lo.cpu())
            all_labels.append(la.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    n_host = logits.shape[1] - n_extra_classes
    phage_mask = labels[:, -1] == 0

    uncal_loss = F.binary_cross_entropy_with_logits(logits, labels).item()

    host_logits = logits[phage_mask, :n_host]
    host_labels = labels[phage_mask, :n_host]
    T_host = _optimize_temperature(host_logits, host_labels, lr, max_iter)

    bact_logits = logits[:, n_host:]
    bact_labels = labels[:, n_host:]
    T_bact = _optimize_temperature(bact_logits, bact_labels, lr, max_iter)

    T_vec = torch.full((logits.shape[1],), T_host)
    T_vec[n_host:] = T_bact
    cal_loss = F.binary_cross_entropy_with_logits(logits / T_vec, labels).item()

    n_phage = phage_mask.sum().item()
    n_bact = (~phage_mask).sum().item()
    logger.info(f"  Split temperature calibration:")
    logger.info(f"    T_host={T_host:.4f} (on {n_phage} phage samples, "
                f"{n_host} classes)")
    logger.info(f"    T_bact={T_bact:.4f} (on {n_phage + n_bact} samples, "
                f"{n_extra_classes} classes)")
    logger.info(f"    BCE before: {uncal_loss:.5f}  after: {cal_loss:.5f}")

    return T_host, T_bact, T_vec, logits, labels


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
                     blocked_classes=None, eval_stride=None,
                     temperature_host=None, temperature_bacterial=None):
    """Save calibration.json alongside checkpoints."""
    data = {
        'temperature': temperature,
        'threshold': threshold,
        'hosts': hosts.tolist() if hasattr(hosts, 'tolist') else list(hosts),
        'model_config': model_config,
    }
    if temperature_host is not None:
        data['temperature_host'] = temperature_host
    if temperature_bacterial is not None:
        data['temperature_bacterial'] = temperature_bacterial
    if eval_stride is not None:
        data['eval_stride'] = eval_stride
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
                    min_val_support, eval_stride=None, n_extra_classes=0):
    """Run temperature calibration, FDR thresholds, and class blocking."""
    T_host = None
    T_bact = None

    if n_extra_classes > 0:
        T_host, T_bact, T_vec, logits, labels = calibrate_temperature_split(
            model, loader, device, unpack_fn,
            n_extra_classes=n_extra_classes)
        T = T_vec
    else:
        T, logits, labels = calibrate_temperature(
            model, loader, device, unpack_fn)

    fdr_thresholds = find_fdr_thresholds(logits, labels, T)
    blocked = find_blocked_classes(
        logits, labels, T,
        min_precision=min_val_precision,
        min_support=min_val_support,
    ) if min_val_precision > 0 else []

    T_scalar = T if isinstance(T, float) else T_host
    save_calibration(
        os.path.join(run_dir, 'calibration.json'),
        temperature=T_scalar, hosts=hosts, model_config=model_config,
        threshold=eval_threshold, fdr_thresholds=fdr_thresholds,
        blocked_classes=blocked, eval_stride=eval_stride,
        temperature_host=T_host, temperature_bacterial=T_bact,
    )
