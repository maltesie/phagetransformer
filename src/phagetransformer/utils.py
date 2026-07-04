
import csv
import gc
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


def find_blocked_classes(logits, labels, temperature, block_threshold,
                        max_fdr=0.5, min_predictions=1):
    """Block classes that produce confident false positives.

    Predictions are taken at ``block_threshold`` (by default the 10% FDR
    threshold). A class is blocked if it makes at least ``min_predictions``
    predictions above that threshold AND its empirical FDR there
    (FP / (TP + FP)) exceeds ``max_fdr``.

    Classes that make no (or too few) predictions produce no confident false
    positives, so they are never blocked — a class is not penalised merely for
    low recall (which the old precision-at-0.5 rule did, since precision is 0
    when nothing is predicted).

    Returns sorted list of blocked class indices.
    """
    probs = torch.sigmoid(logits / temperature)
    preds = (probs >= block_threshold).float()
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    n_pred = tp + fp

    blocked = []
    for i in range(len(n_pred)):
        npi = n_pred[i].item()
        if npi >= min_predictions:
            fdr_i = (fp[i] / n_pred[i]).item()
            if fdr_i > max_fdr:
                blocked.append(int(i))

    if blocked:
        logger.info(f"  Blocked {len(blocked)} classes with FDR > {max_fdr} at "
                    f"threshold {block_threshold:.4f} "
                    f"(min_predictions={min_predictions})")
    else:
        logger.info(f"  No classes blocked (none exceed FDR {max_fdr} with "
                    f">= {min_predictions} predictions at "
                    f"threshold {block_threshold:.4f})")
    return sorted(blocked)


# ===========================================================================
# Out-of-distribution detection — class-conditional Mahalanobis on per-patch
# embeddings, fit on training phages, thresholded on validation.
#
# This replaces per-class blocking: rather than silencing low-precision host
# classes, we score how far a sequence's patch embeddings fall from the ID
# phage manifold and let a single rejection threshold (calibrated to a target
# ID rejection rate) handle both genuinely OOD inputs and badly predictable
# ones. Everything here runs on the frozen model at calibration time; only the
# per-sequence scoring is meant to be cheap enough for inference.
# ===========================================================================

# ---- numpy-only ranking metrics (no scipy/sklearn dependency) -------------

def _rankdata(a):
    """Average ranks (1-based); ties get the mean rank. Matches scipy."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auroc(scores, labels):
    """AUROC via Mann-Whitney U; label 1 = positive (OOD), higher score = OOD."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    r = _rankdata(scores)
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def _spearman(x, y):
    """Spearman rho over finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan')
    rx = _rankdata(x[m]) - (m.sum() + 1) / 2.0
    ry = _rankdata(y[m]) - (m.sum() + 1) / 2.0
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float('nan')


def _fpr_at_tpr(scores, labels, tpr_target=0.95):
    """FPR at the threshold that yields TPR >= tpr_target. Positive = OOD."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    thr = np.quantile(pos, 1.0 - tpr_target)
    return float((neg >= thr).mean())


# ---- reliability numeric primitives ---------------------------------------

# Reliability features are OOD signals only (no host-prob features): patch
# typicality (n-standardised), OOD-fraction, and aggregator-space typicality.
# This makes reliability a pure far-from-distribution safeguard -- a randomly
# high host prediction on a distant sequence cannot inflate it.
RELIABILITY_FEATURE_ORDER = ['z_typicality', 'ood_fraction', 'agg_typicality']


def _family_of(lineage):
    """Family-level prefix of a Phylum;Class;Order;Family;Genus lineage."""
    return ";".join(str(lineage).split(";")[:4])


def _reliability_feature_matrix(order, cols):
    """Build the (N, len(order)) feature matrix from a dict of named columns,
    in the given order. Single source of truth for fit and inference; the
    stored ``feature_order`` drives inference."""
    return np.stack(
        [np.atleast_1d(np.asarray(cols[name], dtype=float)) for name in order],
        axis=1)


def _weighted_quantile(values, quantiles, weights=None):
    """Weighted empirical quantiles. `quantiles` in [0,1]. With weights=None
    this matches np.quantile (linear). Used so the ID distance CDFs can be
    reweighted (e.g. to undo redundancy-biased selection of the calib set)."""
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if weights is None:
        return np.quantile(values, quantiles)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    # midpoint (Hazen-style) plotting positions for the weighted ECDF
    pos = (cw - 0.5 * w) / w.sum()
    return np.interp(quantiles, pos, v)


def _logistic_fit(X, y, l2=1.0, iters=200, tol=1e-8, sample_weight=None):
    """Newton-IRLS logistic regression with L2 (bias unpenalised) and optional
    per-sample weights. ``X`` should already be standardised. Returns ``(w, b)``."""
    N, Fdim = X.shape
    sw = np.ones(N) if sample_weight is None else np.asarray(sample_weight, float)
    Xb = np.hstack([X, np.ones((N, 1))])
    theta = np.zeros(Fdim + 1)
    reg = l2 * np.eye(Fdim + 1)
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ theta, -30, 30)))
        W = sw * p * (1 - p)
        grad = Xb.T @ (sw * (p - y)) + reg @ theta
        H = (Xb * W[:, None]).T @ Xb + reg
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        theta_new = theta - step
        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta[:-1], float(theta[-1])


def _cdf_interp(x, values, probs):
    """CDF(x) from a stored quantile function (values at probs). Clamped [0,1]."""
    return np.clip(np.interp(np.asarray(x, dtype=float), values, probs), 0.0, 1.0)


def _fit_sigma_n(n_arr, s_arr, min_per_bin=20, max_bins=10):
    """Fit sigma(n)=max(c*n^-beta, sigma_min) and mu0=mean(s) from native
    (n, score) pairs — the n-conditional ID reference, built without any
    subsampling (real within-genome patch correlation is already present)."""
    n_arr = np.asarray(n_arr, float)
    s_arr = np.asarray(s_arr, float)
    mu0 = float(s_arr.mean())
    global_std = float(s_arr.std() + 1e-9)
    fallback = dict(c=global_std, beta=0.0,
                    sigma_min=global_std * 0.3, mu0=mu0)
    if len(n_arr) < 3 * min_per_bin:
        return fallback
    order = np.argsort(n_arr)
    n_sorted, s_sorted = n_arr[order], s_arr[order]
    nbin = min(max_bins, max(1, len(n_arr) // min_per_bin))
    ncent, sstd = [], []
    for ix in np.array_split(np.arange(len(n_arr)), nbin):
        if len(ix) < min_per_bin:
            continue
        ncent.append(n_sorted[ix].mean())
        sstd.append(s_sorted[ix].std() + 1e-9)
    ncent, sstd = np.array(ncent), np.array(sstd)
    if len(ncent) < 3:
        return fallback
    A = np.vstack([np.log(ncent), np.ones_like(ncent)]).T
    slope, intercept = np.linalg.lstsq(A, np.log(sstd), rcond=None)[0]
    return dict(c=float(np.exp(intercept)), beta=float(-slope),
                sigma_min=float(max(sstd.min() * 0.5, global_std * 0.05)),
                mu0=mu0)


def _sigma_of_n(n, m):
    return np.maximum(m['c'] * np.power(np.asarray(n, float), -m['beta']),
                      m['sigma_min'])


# ---- lineage / shrinkage --------------------------------------------------

def _lineage_parent(host: str) -> str:
    """Family-level key for a host lineage string.

    Hosts are ``Phylum;Class;Order;Family;Genus``; the parent is everything
    but the last component. Falls back to a sentinel when there is no
    separator so such genera all shrink toward the grand mean.
    """
    if isinstance(host, str) and ';' in host:
        return ';'.join(host.split(';')[:-1])
    return '<root>'


def _hierarchical_shrink(raw_means, counts, parents, lam):
    """Shrink genus means toward family centroid, family toward grand mean.

    ``mu_k' = (n_k mu_k + lam mu_family') / (n_k + lam)`` with the family
    centroid itself shrunk toward the (patch-weighted) grand mean by the same
    ``lam``. ``lam`` is in units of equivalent patches: a genus with
    ``n_k >> lam`` keeps its own mean; ``n_k << lam`` mostly borrows its
    family's. Rows with count 0 are left untouched (the caller drops them).
    """
    K, D = raw_means.shape
    valid = counts > 0
    if valid.sum() == 0:
        return raw_means.copy()
    w = counts.astype(np.float64)
    grand = (raw_means[valid] * w[valid, None]).sum(0) / w[valid].sum()

    fam_sum, fam_cnt = {}, {}
    for k in range(K):
        if not valid[k]:
            continue
        p = parents[k]
        fam_sum[p] = fam_sum.get(p, 0.0) + raw_means[k] * w[k]
        fam_cnt[p] = fam_cnt.get(p, 0.0) + w[k]
    fam_shrunk = {}
    for p, s in fam_sum.items():
        Nf = fam_cnt[p]
        mu_f = s / Nf
        fam_shrunk[p] = (Nf * mu_f + lam * grand) / (Nf + lam)

    out = raw_means.copy()
    for k in range(K):
        if not valid[k]:
            continue
        mu_f = fam_shrunk[parents[k]]
        out[k] = (w[k] * raw_means[k] + lam * mu_f) / (w[k] + lam)
    return out


# ---- gathering per-patch embeddings from the frozen model -----------------

def _gather_patch_cache(model, loader, device, n_host, keep_bacterial=False):
    """Encode every sequence in ``loader`` into per-patch embeddings.

    Returns a list (one entry per sequence, in loader order) of dicts::

        emb     : (n_i, D) float16 numpy   — patch embeddings on CPU
        pooled  : (D_agg,) float32 numpy   — aggregator pooled seq embedding
        active  : int array                — host-genus indices (label > 0)
        n       : int                       — patch count
        is_bact : bool                      — extra (bacterial) class positive

    ``pooled`` is the aggregator's sequence embedding (the vector the
    classifier head decides from), computed in the same pass so the
    aggregator-space OOD detector needs no extra encode. Embeddings are held
    in float16 to keep the training-set cache manageable. Phage rows are
    always kept; bacterial rows only when ``keep_bacterial`` is set.
    """
    model.eval()
    cache = []
    with torch.no_grad():
        for patches, counts, labels in loader:
            counts_d = counts.to(device, non_blocking=True)
            per_seq = model.encode_patches(
                patches.to(device, non_blocking=True), counts_d)
            embs = torch.nn.utils.rnn.pad_sequence(per_seq, batch_first=True)
            _, pooled = model.aggregator(embs, counts_d, return_embedding=True)
            pooled = pooled.detach().to('cpu', torch.float32).numpy()
            labels = labels.cpu().numpy()
            for i, (emb, lab) in enumerate(zip(per_seq, labels)):
                is_bact = bool(lab[n_host:].any()) if lab.shape[0] > n_host else False
                if is_bact and not keep_bacterial:
                    continue
                active = np.where(lab[:n_host] > 0)[0].astype(np.int64)
                cache.append({
                    'emb': emb.detach().to('cpu', torch.float16).numpy(),
                    'pooled': pooled[i],
                    'active': active,
                    'n': int(emb.shape[0]),
                    'is_bact': is_bact,
                })
    return cache


# ---- fitting the class-conditional Mahalanobis model ----------------------

def fit_class_conditional_mahalanobis(cache, n_host, hosts,
                                      shrinkage_lambda=200.0,
                                      cov_jitter_rel=1e-6, normalize=False):
    """Fit per-genus means (shared covariance) from a patch-embedding cache.

    Two exact passes over the cache (no subsampling): pass 1 accumulates
    per-genus sums to form raw means (each patch contributes to every host
    genus positive on its parent sequence — multi-label aware); means are
    then hierarchically shrunk. Pass 2 accumulates the pooled within-class
    scatter of each patch around the average of its assigned (shrunk) means,
    giving one shared precision.

    Genera with no training patches are dropped from the candidate set
    entirely, so they cannot act as spurious low-distance attractors.

    Returns a dict with numpy arrays:
        means     : (Kv, D)  shrunk means of the Kv genera that have support
        cholL     : (D, D)   cholesky(precision), lower-triangular
        valid_idx : (Kv,)    original host-class indices of those means
        patch_counts / seq_counts : (n_host,) support per genus
    """
    if not cache:
        raise ValueError("empty embedding cache — nothing to fit OOD model on")
    D = cache[0]['emb'].shape[1]
    parents = [_lineage_parent(h) for h in hosts[:n_host]]

    sums = np.zeros((n_host, D), dtype=np.float64)
    patch_counts = np.zeros(n_host, dtype=np.float64)
    seq_counts = np.zeros(n_host, dtype=np.float64)
    for item in cache:
        active = item['active']
        if len(active) == 0:
            continue
        e = item['emb'].astype(np.float64)
        if normalize:
            e = _l2norm(e)
        s = e.sum(0)
        for k in active:
            sums[k] += s
            patch_counts[k] += e.shape[0]
            seq_counts[k] += 1

    valid = patch_counts > 0
    raw_means = np.zeros((n_host, D), dtype=np.float64)
    raw_means[valid] = sums[valid] / patch_counts[valid, None]
    means = _hierarchical_shrink(raw_means, patch_counts, parents,
                                 shrinkage_lambda)

    W = np.zeros((D, D), dtype=np.float64)
    n_total = 0
    for item in cache:
        active = item['active']
        if len(active) == 0:
            continue
        e = item['emb'].astype(np.float64)
        if normalize:
            e = _l2norm(e)
        mu = means[active].mean(0)
        dev = e - mu
        W += dev.T @ dev
        n_total += e.shape[0]

    dof = max(n_total - int(valid.sum()), 1)
    cov = W / dof
    cov += cov_jitter_rel * float(np.mean(np.diag(cov))) * np.eye(D)
    precision = np.linalg.inv(cov)
    cholL = np.linalg.cholesky(precision)

    valid_idx = np.where(valid)[0]
    logger.info(f"  OOD fit: {int(valid.sum())}/{n_host} genera with support "
                f"(dropped {n_host - int(valid.sum())} empty), "
                f"{n_total:,} patches, D={D}, lambda={shrinkage_lambda}")
    logger.info(f"    genus support: min={int(seq_counts[valid].min())} "
                f"median={int(np.median(seq_counts[valid]))} "
                f"max={int(seq_counts[valid].max())} sequences")
    return dict(means=means[valid_idx].astype(np.float32),
                cholL=cholL.astype(np.float32),
                valid_idx=valid_idx.astype(np.int64),
                patch_counts=patch_counts, seq_counts=seq_counts)


def _l2norm(X):
    """Row-wise L2 normalization to the unit sphere (zero-safe)."""
    X = np.asarray(X, dtype=np.float64)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def _min_maha_np(X, means, cholL, normalize=False):
    """Per-row min-over-genus Mahalanobis distance (numpy).

    ``dist_k^2 = || (x - mu_k) @ L ||^2`` with ``L = cholesky(precision)``.
    ``X`` (n, D), ``means`` (Kv, D), ``cholL`` (D, D) -> (n,) distances.
    When ``normalize`` is set, query rows are L2-normalized first (the means
    were fit on normalized vectors), turning this into a whitened distance on
    the unit sphere — directional rather than magnitude-sensitive.
    """
    X = np.asarray(X, dtype=np.float64)
    if normalize:
        X = _l2norm(X)
    Xw = X @ cholL                                   # (n, D)
    Mw = means @ cholL                               # (Kv, D)
    d2 = ((Xw * Xw).sum(1)[:, None]
          - 2.0 * Xw @ Mw.T + (Mw * Mw).sum(1)[None, :])
    return np.sqrt(np.clip(d2.min(1), 0.0, None))


def _per_patch_distances(cache, means, cholL, normalize=False):
    """List (per sequence, in cache order) of per-patch min-genus distances."""
    means = means.astype(np.float64)
    cholL = cholL.astype(np.float64)
    return [_min_maha_np(item['emb'].astype(np.float64), means, cholL,
                         normalize=normalize)
            for item in cache]


class ReliabilityScorer:
    """Inference-side folded reliability score (loaded from calibration).

    Given a sequence's per-patch embeddings, the aggregator pooled embedding,
    and its calibrated host probabilities, returns the raw patch OOD distance,
    the patch OOD metrics (mean typicality, OOD-fraction), the aggregator-space
    typicality, and the final folded reliability in [0, 1]. All numpy.
    """

    def __init__(self, means, cholL, rel, agg_means=None, agg_cholL=None,
                 normalize=False):
        self.means = np.asarray(means, dtype=np.float64)
        self.cholL = np.asarray(cholL, dtype=np.float64)
        self.normalize = bool(normalize)
        self.Fp_values = np.asarray(rel['patch_dist_quantiles']['values'])
        self.Fp_probs = np.asarray(rel['patch_dist_quantiles']['probs'])
        self.tau = float(rel['tau'])
        self.sm = rel['sigma_model']
        self.z_values = np.asarray(rel['z_quantiles']['values'])
        self.z_probs = np.asarray(rel['z_quantiles']['probs'])
        self.n0 = float(rel['n0'])
        self.logistic = rel.get('logistic')  # may be None
        # aggregator-space model (optional; present on newer calibrations)
        self.agg_means = None if agg_means is None else np.asarray(agg_means, np.float64)
        self.agg_cholL = None if agg_cholL is None else np.asarray(agg_cholL, np.float64)
        aq = rel.get('agg_dist_quantiles')
        self.Fa_values = np.asarray(aq['values']) if aq else None
        self.Fa_probs = np.asarray(aq['probs']) if aq else None

    @classmethod
    def from_calibration(cls, model_dir, calib):
        """Build from a model dir + parsed calibration.json, or None if the
        run has no OOD/reliability block (backward compatible)."""
        ood = calib.get('ood')
        if not ood or 'reliability' not in ood:
            return None
        npz_path = os.path.join(model_dir, ood.get('sidecar',
                                                    'ood_mahalanobis.npz'))
        if not os.path.exists(npz_path):
            logger.warning(f"OOD sidecar {npz_path} missing; reliability off.")
            return None
        d = np.load(npz_path)
        agg_means = d['agg_means'] if 'agg_means' in d.files else None
        agg_cholL = d['agg_cholL'] if 'agg_cholL' in d.files else None
        normalize = bool(d['normalize']) if 'normalize' in d.files else False
        return cls(d['means'], d['cholL'], ood['reliability'],
                   agg_means=agg_means, agg_cholL=agg_cholL,
                   normalize=normalize)

    def score(self, emb, host_probs, agg_emb=None):
        """emb: (n, D) patch embeddings. agg_emb: (D_agg,) pooled embedding.
        host_probs: (n_host,) calibrated. Returns a dict with the raw distance,
        both patch OOD metrics, the aggregator typicality, and reliability."""
        emb = np.asarray(emb, dtype=np.float64)
        n = emb.shape[0]
        d = _min_maha_np(emb, self.means, self.cholL,
                         normalize=self.normalize)         # (n,)
        raw = float(d.mean())
        q = _cdf_interp(d, self.Fp_values, self.Fp_probs)         # (n,)
        typicality = float(q.mean())
        ood_fraction = float((q > self.tau).mean())

        sigma = float(_sigma_of_n(n, self.sm))
        z = (typicality - self.sm['mu0']) / max(sigma, 1e-9)

        # aggregator-space distance + typicality (if model + embedding present)
        agg_distance = float('nan')
        agg_typicality = float('nan')
        if self.agg_means is not None and agg_emb is not None:
            ad = _min_maha_np(np.asarray(agg_emb, np.float64)[None, :],
                              self.agg_means, self.agg_cholL,
                              normalize=self.normalize)[0]
            agg_distance = float(ad)
            if self.Fa_values is not None:
                agg_typicality = float(_cdf_interp(ad, self.Fa_values,
                                                   self.Fa_probs))

        g = n / (n + self.n0) if self.n0 > 0 else 1.0

        if self.logistic is None:
            rel_typ = 1.0 - float(_cdf_interp(z, self.z_values, self.z_probs))
            reliability = rel_typ * g
        else:
            # OOD-only features; build whatever the stored feature_order asks
            # for (host-prob features are no longer part of the model).
            available = {
                'z_typicality': z,
                'ood_fraction': ood_fraction,
                'agg_typicality': (agg_typicality
                                   if np.isfinite(agg_typicality) else 0.0),
            }
            order = self.logistic.get('feature_order', RELIABILITY_FEATURE_ORDER)
            cols = {name: available[name] for name in order}
            feat = _reliability_feature_matrix(order, cols)
            fmean = np.asarray(self.logistic['feat_mean'])
            fstd = np.asarray(self.logistic['feat_std'])
            xs = (feat[0] - fmean) / fstd
            w = np.asarray(self.logistic['w'])
            b = float(self.logistic['b'])
            p = 1.0 / (1.0 + np.exp(-np.clip(xs @ w + b, -30, 30)))
            reliability = float(p) * g
        return dict(ood_distance=raw, ood_typicality=typicality,
                    ood_fraction=ood_fraction, ood_agg_distance=agg_distance,
                    ood_agg_typicality=agg_typicality,
                    reliability=float(np.clip(reliability, 0.0, 1.0)))


def save_ood_model(path, ood, hosts, n_host, score_agg='mean', agg=None,
                   normalize=False):
    """Persist the fitted detector(s) as a sidecar .npz (not inlined in JSON).

    Always stores the patch-space model (``means``/``cholL``); when ``agg`` is
    given, also stores the aggregator-space model (``agg_means``/``agg_cholL``).
    ``normalize`` records whether embeddings were L2-normalized before fitting
    (predict must apply the same transform).
    """
    valid_hosts = np.asarray(hosts[:n_host])[ood['valid_idx']]
    data = dict(
        means=ood['means'], cholL=ood['cholL'],
        valid_idx=ood['valid_idx'],
        valid_hosts=valid_hosts.astype(str),
        patch_counts=ood['patch_counts'], seq_counts=ood['seq_counts'],
        score_agg=np.array(score_agg), normalize=np.array(bool(normalize)),
    )
    if agg is not None:
        agg_hosts = np.asarray(hosts[:n_host])[agg['valid_idx']]
        data.update(
            agg_means=agg['means'], agg_cholL=agg['cholL'],
            agg_valid_idx=agg['valid_idx'],
            agg_valid_hosts=agg_hosts.astype(str),
        )
    np.savez(path, **data)
    logger.info(f"  OOD model saved -> {path}"
                + ("  (patch + aggregator spaces)" if agg is not None else ""))


def _score_vs_tani(label, score, npatch, phage, tani, tani_cutoff,
                   scramble_score=None, n_patch_bins=(1, 2, 4, 8, 1_000_000)):
    """Compact OOD go/no-go for one score (used per embedding space).

    Logs ID typicality summary, scramble floor AUROC/FPR@95, Spearman vs tANI
    and near-OOD AUROC binned by patch count. Returns a summary sub-dict.
    """
    ph_score = score[phage]
    ph_n = npatch[phage]
    sub = {}
    logger.info(f"    [{label}] ID typicality: median={np.median(ph_score):.3f} "
                f"p95={np.quantile(ph_score, 0.95):.3f}")
    if scramble_score is not None and len(scramble_score):
        y = np.concatenate([np.ones(len(scramble_score)), np.zeros(len(ph_score))])
        s = np.concatenate([scramble_score, ph_score])
        sub['scramble_auroc'] = _auroc(s, y)
        sub['scramble_fpr95'] = _fpr_at_tpr(s, y, 0.95)
        logger.info(f"    [{label}] scramble floor vs ID: "
                    f"AUROC={sub['scramble_auroc']:.4f} "
                    f"FPR@95={sub['scramble_fpr95']:.4f}")
    if tani is not None:
        t = np.asarray(tani, float)[:phage.sum()]
        finite = np.isfinite(t)
        rho = _spearman(ph_score[finite], t[finite])
        sub['tani_spearman'] = rho
        logger.info(f"    [{label}] Spearman(typicality, tANI)={rho:+.4f}")
        near = finite & (t <= tani_cutoff)
        far = finite & (t > tani_cutoff)
        if near.sum() and far.sum():
            binauc = {}
            edges = list(n_patch_bins)
            for a, b in zip(edges[:-1], edges[1:]):
                m = (ph_n >= a) & (ph_n < b)
                nn, ff = near & m, far & m
                if nn.sum() >= 5 and ff.sum() >= 5:
                    ss = np.concatenate([ph_score[nn], ph_score[ff]])
                    yy = np.concatenate([np.ones(nn.sum()), np.zeros(ff.sum())])
                    au = _auroc(ss, yy)
                    binauc[f"{a}-{b if b < 1e6 else 'inf'}"] = au
                    logger.info(f"      [{label}] patches "
                                f"[{a},{b if b < 1e6 else 'inf'}): AUROC={au:.4f} "
                                f"(near={nn.sum()} far={ff.sum()})")
            sub['near_ood_auroc_by_patchbin'] = binauc
    return sub


def evaluate_ood(ood_score, npatch, phage_mask, tani=None, tani_cutoff=0.5,
                 scramble_score=None, seq_counts=None, active_lists=None,
                 rare_seq_threshold=20, n_patch_bins=(1, 2, 4, 8, 1_000_000),
                 reliability=None, correct=None, target_reject_rate=None):
    """Log the go/no-go diagnostics for the OOD / reliability score.

    ``ood_score`` (per-patch typicality aggregated per sequence), ``npatch``
    and ``phage_mask`` are aligned per-sequence arrays over the validation
    loader. The synthetic floor is ``scramble_score`` (typicality of scrambled
    val phages) — bacterial is deliberately *not* used, since the model was
    trained on bacterial sequence and their embeddings can sit inside the ID
    manifold. The meaningful test is genetic distance: if ``tani`` is given
    (aligned to the phage rows) we report Spearman(score, tANI) and near-OOD
    AUROC binned by patch count so distance is not confounded with contig
    length.

    When ``reliability`` and top-1 ``correct`` are given, we additionally tie
    the flag to *host-call correctness*: a confusion matrix at the reject
    operating point (flag precision / recall, retained error), a risk-coverage
    curve over reliability (retained error as coverage drops), and the retained
    error split by tANI bin. Returns a summary dict.
    """
    phage = np.asarray(phage_mask, dtype=bool)
    ph_score = ood_score[phage]
    ph_n = npatch[phage]
    summary = {}

    logger.info("  --- OOD / reliability evaluation (validation) ---")
    logger.info(f"    ID phage typicality: median={np.median(ph_score):.3f} "
                f"p95={np.quantile(ph_score, 0.95):.3f} (n={phage.sum()})")

    if scramble_score is not None and len(scramble_score):
        y = np.concatenate([np.ones(len(scramble_score)), np.zeros(len(ph_score))])
        s = np.concatenate([scramble_score, ph_score])
        summary['scramble_auroc'] = _auroc(s, y)
        summary['scramble_fpr95'] = _fpr_at_tpr(s, y, 0.95)
        logger.info(f"    scrambled-phage floor vs ID: "
                    f"AUROC={summary['scramble_auroc']:.4f} "
                    f"FPR@95={summary['scramble_fpr95']:.4f} "
                    f"(scramble median={np.median(scramble_score):.3f})")

    if tani is not None:
        tani = np.asarray(tani, dtype=float)[:phage.sum()]
        finite = np.isfinite(tani)
        rho = _spearman(ph_score[finite], tani[finite])
        summary['tani_spearman'] = rho
        logger.info(f"    Spearman(typicality, tANI) = {rho:+.4f} "
                    f"(expect negative: distant phages score higher; "
                    f"n={finite.sum()})")

        near = finite & (tani <= tani_cutoff)
        far = finite & (tani > tani_cutoff)
        if near.sum() and far.sum():
            logger.info(f"    near-OOD (tANI<={tani_cutoff}) vs ID, binned by patch count:")
            bin_aurocs = {}
            edges = list(n_patch_bins)
            for a, b in zip(edges[:-1], edges[1:]):
                m = (ph_n >= a) & (ph_n < b)
                nn, ff = near & m, far & m
                if nn.sum() >= 5 and ff.sum() >= 5:
                    s = np.concatenate([ph_score[nn], ph_score[ff]])
                    yy = np.concatenate([np.ones(nn.sum()), np.zeros(ff.sum())])
                    au = _auroc(s, yy)
                    bin_aurocs[f"{a}-{b if b < 1e6 else 'inf'}"] = au
                    logger.info(f"      patches [{a},{b if b < 1e6 else 'inf'}): "
                                f"AUROC={au:.4f}  (near={nn.sum()} far={ff.sum()})")
            summary['near_ood_auroc_by_patchbin'] = bin_aurocs

    if reliability is not None and correct is not None:
        rel_ph = reliability[phage]
        cor_ph = correct[phage]
        typ_ph = ph_score
        tani_ph = np.asarray(tani, float)[:phage.sum()] if tani is not None else None
        valid = np.isfinite(cor_ph) & np.isfinite(rel_ph)
        if valid.sum() >= 20 and 0 < cor_ph[valid].sum() < valid.sum():
            relv = rel_ph[valid]
            wrong = ~cor_ph[valid].astype(bool)          # True = wrong host call
            n = int(valid.sum())
            base_err = float(wrong.mean())

            au = _auroc(relv, (~wrong).astype(float))
            summary['reliability_vs_correct_auroc'] = au
            logger.info(f"    reliability vs top-1 correctness: AUROC={au:.4f} "
                        f"(n={n}, base correct rate={1-base_err:.3f})")

            # confusion matrix at the OOD reject operating point
            if target_reject_rate is not None and 0 < target_reject_rate < 1:
                flag_thr = float(np.quantile(typ_ph, 1.0 - target_reject_rate))
                flagged = typ_ph[valid] >= flag_thr      # flagged = OOD-rejected
                fw = int((flagged & wrong).sum())
                fr = int((flagged & ~wrong).sum())
                uw = int((~flagged & wrong).sum())
                ur = int((~flagged & ~wrong).sum())
                flag_prec = fw / max(fw + fr, 1)
                flag_recall = fw / max(fw + uw, 1)
                retained_err = uw / max(uw + ur, 1)
                logger.info(f"    OOD flag @ reject {target_reject_rate:.0%} "
                            f"(typicality>={flag_thr:.3f}), on {n} labelled ID phages:")
                logger.info(f"      flagged&wrong={fw}  flagged&right={fr}  "
                            f"unflagged&wrong={uw}  unflagged&right={ur}")
                logger.info(f"      flag precision={flag_prec:.3f} (of flagged, "
                            f"share truly wrong)  flag recall={flag_recall:.3f} "
                            f"(of wrong calls, share caught)")
                logger.info(f"      retained error={retained_err:.3f} vs "
                            f"base error={base_err:.3f} "
                            f"(error among kept predictions)")
                summary['flag'] = dict(
                    threshold=flag_thr, flagged_wrong=fw, flagged_right=fr,
                    unflagged_wrong=uw, unflagged_right=ur,
                    precision=flag_prec, recall=flag_recall,
                    retained_error=retained_err, base_error=base_err)

                if tani_ph is not None:
                    tv = tani_ph[valid]
                    kept = ~flagged
                    for name, msk in (
                            (f'near tANI<={tani_cutoff}',
                             kept & np.isfinite(tv) & (tv <= tani_cutoff)),
                            (f'far  tANI>{tani_cutoff}',
                             kept & np.isfinite(tv) & (tv > tani_cutoff))):
                        if msk.sum():
                            logger.info(f"      retained error [{name}]: "
                                        f"{wrong[msk].mean():.3f} "
                                        f"(n={int(msk.sum())})")

            # risk-coverage: retain highest-reliability first, error of kept set
            order = np.argsort(-relv)
            cum_err = np.cumsum(wrong[order]) / np.arange(1, n + 1)
            rc = {}
            logger.info("    risk-coverage (retain highest-reliability first):")
            for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
                k = max(1, int(round(c * n)))
                rc[f"{c:.1f}"] = float(cum_err[k - 1])
                logger.info(f"      coverage {c:.0%}: retained error="
                            f"{cum_err[k-1]:.3f} (n={k})")
            summary['risk_coverage'] = rc
            summary['aurc'] = float(cum_err.mean())
            logger.info(f"      AURC={summary['aurc']:.4f} (lower=better; "
                        f"flat vs base error {base_err:.3f} => score ranks "
                        f"distance, not error)")

    if seq_counts is not None and active_lists is not None:
        rare_flag = np.array([
            (len(a) > 0 and min(seq_counts[k] for k in a) < rare_seq_threshold)
            for a in active_lists], dtype=bool)
        thr = np.quantile(ph_score, 0.95)
        for name, mask in (('rare-host', rare_flag[phage]),
                           ('common-host', ~rare_flag[phage])):
            if mask.sum():
                rej = (ph_score[mask] >= thr).mean()
                logger.info(f"    ID rejection @p95 for {name}: {rej:.3f} "
                            f"(n={mask.sum()})")
        summary['rare_host_frac'] = float(rare_flag[phage].mean())
    return summary


def save_calibration(path, temperature, hosts, model_config,
                     threshold=0.5, fdr_thresholds=None,
                     blocked_classes=None, stride=None,
                     temperature_host=None, temperature_bacterial=None,
                     ood=None):
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
    if stride is not None:
        data['stride'] = stride
    if fdr_thresholds:
        data['fdr_thresholds'] = {
            f"fdr_{int(fdr*100):02d}": round(t, 5)
            for fdr, t in fdr_thresholds.items()
        }
    if blocked_classes is not None:
        data['blocked_classes'] = blocked_classes
    if ood is not None:
        data['ood'] = ood
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"  calibration saved -> {path}")


def _gather_logits_labels(model, loader, device, unpack_fn):
    """Run the model over the loader and return concatenated (logits, labels)
    on CPU, without fitting any temperature."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            lo, la = unpack_fn(model, batch, device)
            all_logits.append(lo.cpu())
            all_labels.append(la.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def _run_ood(model, fit_loader, val_loader, device, n_host, hosts, run_dir,
             target_reject_rate, shrinkage_lambda, score_agg,
             tani, tani_cutoff, val_logits, val_labels, host_temperature,
             scramble_loader=None, tau=0.95, n0=0.0, logistic_l2=1.0,
             min_logistic_fit=100, reliability_features=None,
             normalize=False, true_families=None, id_weights=None,
             ood_eval_loader=None):
    """Fit the OOD detector(s) + folded reliability model at calibration time.

    Two Mahalanobis detectors are fit on training phages: patch-space (per-patch
    encoder embeddings) and aggregator-space (the pooled per-sequence embedding
    the classifier decides from). Pipeline:
      1. fit both class-conditional Mahalanobis models on training phages;
      2. score validation -> patch typicality/OOD-fraction and aggregator
         typicality (each via its own ID distance CDF);
      3. n-conditional standardisation of patch typicality (mu0, sigma(n));
      4. logistic P(family-correct) over OOD-only features [z_typicality,
         ood_fraction, agg_typicality] -- no host-prob features, so a random
         high host prediction on a distant sequence cannot raise reliability;
         fit on discard_id + same-family OOD (family-level correctness);
      5. final reliability = P_correct (optionally x n/(n+n0) if n0>0; the
         count floor is off by default so short genomes aren't penalised).

    ``val_logits`` / ``val_labels`` are aligned row-for-row with ``val_loader``.
    Writes both detectors to ``ood_mahalanobis.npz`` and returns the ``ood``
    block (with a nested ``reliability`` sub-block).
    """
    raw = getattr(model, '_orig_mod', model)
    logger.info("  --- OOD detectors (class-conditional Mahalanobis) ---")
    logger.info("  Gathering training embeddings for the fit "
                "(full coverage, no subsampling) …")
    fit_cache = _gather_patch_cache(raw, fit_loader, device, n_host,
                                    keep_bacterial=False)
    # patch-space: one Mahalanobis over per-patch encoder embeddings
    ood = fit_class_conditional_mahalanobis(
        fit_cache, n_host, hosts, shrinkage_lambda=shrinkage_lambda,
        normalize=normalize)
    means, cholL = ood['means'], ood['cholL']
    # aggregator-space: one Mahalanobis over the per-sequence pooled embedding
    logger.info("  Fitting aggregator-space detector (pooled seq embedding) …")
    agg_items = [{'emb': it['pooled'][None, :], 'active': it['active'],
                  'n': 1, 'is_bact': it['is_bact']} for it in fit_cache]
    agg = fit_class_conditional_mahalanobis(
        agg_items, n_host, hosts, shrinkage_lambda=shrinkage_lambda,
        normalize=normalize)
    agg_means, agg_cholL = agg['means'], agg['cholL']
    del fit_cache, agg_items
    gc.collect()
    if normalize:
        logger.info("  (embeddings L2-normalized before fit/scoring)")

    def _agg_distances(cache_items):
        P = np.stack([it['pooled'] for it in cache_items]).astype(np.float64)
        return _min_maha_np(P, agg_means.astype(np.float64),
                            agg_cholL.astype(np.float64), normalize=normalize)

    logger.info("  Scoring validation sequences …")
    val_cache = _gather_patch_cache(raw, val_loader, device, n_host,
                                    keep_bacterial=True)
    vl = (val_logits.float().cpu().numpy() if torch.is_tensor(val_logits)
          else np.asarray(val_logits, dtype=np.float64))
    vy = (val_labels.float().cpu().numpy() if torch.is_tensor(val_labels)
          else np.asarray(val_labels, dtype=np.float64))
    if len(val_cache) != vl.shape[0]:
        raise RuntimeError(
            f"val cache ({len(val_cache)}) and logits ({vl.shape[0]}) "
            f"misaligned — the loader must be iterated with shuffle=False")

    dists = _per_patch_distances(val_cache, means, cholL, normalize=normalize)
    npatch = np.array([len(d) for d in dists], dtype=int)
    raw_d = np.array([float(d.mean()) for d in dists])
    is_bact = np.array([it['is_bact'] for it in val_cache], dtype=bool)
    active_lists = [it['active'] for it in val_cache]
    phage = ~is_bact

    # F_patch: ID (phage) per-patch distance CDF
    probs_grid = np.linspace(0.0, 1.0, 201)
    id_patch_d = np.concatenate([dists[i] for i in np.where(phage)[0]])
    Fp_values = np.quantile(id_patch_d, probs_grid)

    def _typ_frac(d):
        q = np.clip(np.interp(d, Fp_values, probs_grid), 0.0, 1.0)
        return float(q.mean()), float((q > tau).mean())

    typ = np.empty(len(dists))
    frac = np.empty(len(dists))
    for i, d in enumerate(dists):
        typ[i], frac[i] = _typ_frac(d)

    # aggregator-space per-sequence distance -> typicality via ID CDF
    agg_d = _agg_distances(val_cache)
    Fa_values = np.quantile(agg_d[phage], probs_grid)
    agg_typ = np.clip(np.interp(agg_d, Fa_values, probs_grid), 0.0, 1.0)

    # Reference ID raw-distance ranges (compare inference distances to these to
    # catch a calibration/model mismatch — inference >> these p95 means the
    # OOD model was fit on a different model's embeddings).
    logger.info(f"  ID raw distance (patch):      median="
                f"{np.median(raw_d[phage]):.2f} p95="
                f"{np.quantile(raw_d[phage], 0.95):.2f} "
                f"max={raw_d[phage].max():.2f}")
    logger.info(f"  ID raw distance (aggregator): median="
                f"{np.median(agg_d[phage]):.2f} p95="
                f"{np.quantile(agg_d[phage], 0.95):.2f} "
                f"max={agg_d[phage].max():.2f}")

    # raw reject threshold kept for reference / simple gating
    reject_threshold = float(np.quantile(raw_d[phage], 1.0 - target_reject_rate))
    id_score_quantiles = np.quantile(
        raw_d[phage], np.linspace(0, 1, 101)).astype(float).tolist()

    # n-conditional typicality standardisation (fit on native ID phages)
    sm = _fit_sigma_n(npatch[phage], typ[phage])
    z_all = (typ - sm['mu0']) / _sigma_of_n(npatch, sm)
    z_values = np.quantile(z_all[phage], probs_grid)

    # Top-1 predicted genus per row (argmax of host logits; monotone under
    # sigmoid/temperature, so no host-prob magnitude is used anywhere).
    pred_arg = vl[:, :n_host].argmax(1)

    # Family-level correctness. For each row we need the set of TRUE host
    # families: supplied by the caller (trained_hosts for discard_id,
    # novel_hosts for same-family OOD). Fallback to genus-correct from the
    # multi-hot labels when true_families isn't provided (legacy val path).
    host_families = np.array([_family_of(h) for h in hosts[:n_host]])
    pred_family = host_families[pred_arg]
    correct = np.full(len(dists), np.nan)
    if true_families is not None:
        for i, fams in enumerate(true_families):
            if fams:
                correct[i] = 1.0 if pred_family[i] in fams else 0.0
    else:
        has_label = vy[:, :n_host].sum(1) > 0
        lab_rows = np.where(has_label)[0]
        correct[lab_rows] = (vy[lab_rows, pred_arg[lab_rows]] > 0).astype(float)

    # OOD-only features (no host-prob features).
    feat_cols = {'z_typicality': z_all, 'ood_fraction': frac,
                 'agg_typicality': agg_typ}
    feat_order = RELIABILITY_FEATURE_ORDER
    feats = _reliability_feature_matrix(feat_order, feat_cols)

    # Per-row fit weights: sampling_weight for discard_id (undo redundancy
    # skew), 1.0 for same-family OOD; 1.0 everywhere if not supplied.
    weights = (np.ones(len(dists)) if id_weights is None
               else np.asarray(id_weights, dtype=float))

    fit_mask = np.isfinite(correct)
    logistic = None
    reliability = np.full(len(dists), np.nan)
    floor = (npatch / (npatch + n0)) if n0 > 0 else np.ones(len(dists))
    y = correct[fit_mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    logger.info(f"  Reliability logistic: features {feat_order}; "
                f"fit rows={int(fit_mask.sum())} "
                f"(family-correct positives={n_pos}, negatives={n_neg})")
    if fit_mask.sum() >= min_logistic_fit and 0 < n_pos < len(y):
        Xf = feats[fit_mask]
        wf = weights[fit_mask]
        fmean = Xf.mean(0)
        fstd = Xf.std(0) + 1e-9
        w, b = _logistic_fit((Xf - fmean) / fstd, y, l2=logistic_l2,
                             sample_weight=wf)
        logistic = dict(w=w.tolist(), b=float(b),
                        feat_mean=fmean.tolist(), feat_std=fstd.tolist(),
                        feature_order=feat_order)
        xs_all = (feats - fmean) / fstd
        p_all = 1.0 / (1.0 + np.exp(-np.clip(xs_all @ w + b, -30, 30)))
        reliability = p_all * floor
        logger.info(f"    base family-correct rate="
                    f"{np.average(y, weights=wf):.3f} (weighted)")
        logger.info("    weights " + ", ".join(
            f"{k}={v:+.3f}" for k, v in zip(feat_order, w)))
    else:
        logger.warning("  Not enough labelled ID phages for the correctness "
                       "logistic; reliability falls back to typicality only.")
        rel_typ = 1.0 - _cdf_interp(z_all, z_values, probs_grid)
        reliability = rel_typ * floor

    # scrambled-phage floor (synthetic OOD) — both spaces
    scramble_typ = None
    scramble_agg_typ = None
    if scramble_loader is not None:
        logger.info("  Scoring scrambled val phages (synthetic OOD floor) …")
        sc_cache = _gather_patch_cache(raw, scramble_loader, device, n_host,
                                       keep_bacterial=True)
        sc_dists = _per_patch_distances(sc_cache, means, cholL,
                                        normalize=normalize)
        scramble_typ = np.array([_typ_frac(d)[0] for d in sc_dists])
        sc_agg_d = _agg_distances(sc_cache)
        scramble_agg_typ = np.clip(np.interp(sc_agg_d, Fa_values, probs_grid),
                                   0.0, 1.0)
        del sc_cache
        gc.collect()

    logger.info("  --- patch-space vs aggregator-space OOD ---")
    summary = evaluate_ood(
        typ, npatch, phage, tani=tani, tani_cutoff=tani_cutoff,
        scramble_score=scramble_typ, seq_counts=ood['seq_counts'],
        active_lists=active_lists, reliability=reliability, correct=correct,
        target_reject_rate=target_reject_rate)
    # parallel OOD go/no-go for the aggregator space
    summary['aggregator'] = _score_vs_tani(
        'aggregator', agg_typ, npatch, phage, tani, tani_cutoff,
        scramble_score=scramble_agg_typ)

    # Distant-OOD evaluation set (e.g. beyond-family discard_ood): should read
    # clearly more atypical than the ID-calibration set. This is the "completely
    # far away" case the reliability score is meant to suppress.
    if ood_eval_loader is not None:
        ev_cache = _gather_patch_cache(raw, ood_eval_loader, device, n_host,
                                       keep_bacterial=False)
        if ev_cache:
            ev_dists = _per_patch_distances(ev_cache, means, cholL,
                                            normalize=normalize)
            ev_typ = np.array([_typ_frac(d)[0] for d in ev_dists])
            ev_agg = np.clip(np.interp(_agg_distances(ev_cache), Fa_values,
                                       probs_grid), 0.0, 1.0)
            id_typ, id_agg = typ[phage], agg_typ[phage]
            ev_sum = {}
            for nm, ev, idv in (('patch', ev_typ, id_typ),
                                ('agg', ev_agg, id_agg)):
                yy = np.concatenate([np.ones(len(ev)), np.zeros(len(idv))])
                ss = np.concatenate([ev, idv])
                au = _auroc(ss, yy)
                ev_sum[f'{nm}_auroc'] = au
                ev_sum[f'{nm}_median'] = float(np.median(ev))
                logger.info(f"  distant-OOD vs ID-calib [{nm}]: AUROC={au:.4f} "
                            f"(distant median={np.median(ev):.3f} "
                            f"ID median={np.median(idv):.3f})")
            summary['distant_ood'] = ev_sum
        del ev_cache
        gc.collect()

    save_ood_model(os.path.join(run_dir, 'ood_mahalanobis.npz'),
                   ood, hosts, n_host, score_agg=score_agg, agg=agg,
                   normalize=normalize)
    logger.info(f"  OOD raw reject threshold @ {target_reject_rate:.0%} ID "
                f"rejection = {reject_threshold:.4f}")
    del val_cache
    gc.collect()

    reliability_block = {
        'patch_dist_quantiles': {'probs': probs_grid.tolist(),
                                 'values': Fp_values.astype(float).tolist()},
        'agg_dist_quantiles': {'probs': probs_grid.tolist(),
                               'values': Fa_values.astype(float).tolist()},
        'tau': float(tau),
        'sigma_model': {k: float(v) for k, v in sm.items()},
        'z_quantiles': {'probs': probs_grid.tolist(),
                        'values': z_values.astype(float).tolist()},
        'n0': float(n0),
        'host_temperature': float(host_temperature),
        'logistic': logistic,
    }
    return {
        'method': 'class_conditional_mahalanobis',
        'sidecar': 'ood_mahalanobis.npz',
        'spaces': ['patch', 'aggregator'],
        'normalize': bool(normalize),
        'score_agg': score_agg,
        'shrinkage_lambda': float(shrinkage_lambda),
        'target_reject_rate': float(target_reject_rate),
        'reject_threshold': reject_threshold,
        'id_score_quantiles': id_score_quantiles,
        'n_genera_with_support': int(len(ood['valid_idx'])),
        'n_genera_with_support_agg': int(len(agg['valid_idx'])),
        'reliability': reliability_block,
        'eval': summary,
    }


# logit-space calibration keys owned by the validation model (spliced in when
# calibrating a production model that has no held-out val for these).
_LOGIT_CALIB_KEYS = ('temperature', 'temperature_host', 'temperature_bacterial',
                     'fdr_thresholds', 'threshold')


def _assert_calib_compatible(ext, hosts, model_config, stride):
    """Hard-fail if the external (validation) calibration was fit on a model
    whose head/vocab/tiling differs from the production model, since the FDR
    thresholds are class-indexed and the manifold is stride-specific."""
    if list(ext.get('hosts', [])) != list(hosts):
        raise ValueError(
            "--merge_ood_from: 'hosts' differ between the validation and "
            "production models; FDR thresholds are class-indexed and would "
            "misalign. Refusing to merge.")
    if ext.get('model_config') != model_config:
        raise ValueError(
            "--merge_ood_from: 'model_config' differs between the two models; "
            "incompatible heads. Refusing to merge.")
    if stride is not None and ext.get('stride') not in (None, stride):
        raise ValueError(
            f"--merge_ood_from: 'stride' differs "
            f"({ext.get('stride')} vs {stride}); the OOD manifold is "
            f"stride-specific. Refusing to merge.")


def _assemble_calibration_dict(hosts, model_config, ood_json, eval_threshold,
                               stride, ext=None):
    """Build calibration.json: production OOD block + (optional) validation
    logit-space keys spliced in verbatim."""
    data = {
        'hosts': hosts.tolist() if hasattr(hosts, 'tolist') else list(hosts),
        'model_config': model_config,
        'threshold': eval_threshold,
        'blocked_classes': [],
        'ood': ood_json,
    }
    if stride is not None:
        data['stride'] = stride
    if ext is not None:
        for k in _LOGIT_CALIB_KEYS:
            if ext.get(k) is not None:
                data[k] = ext[k]
    data.setdefault('temperature', 1.0)
    return data


def run_ood_only_calibration(model, fit_loader, ood_calib_loader, device,
                             unpack_fn, run_dir, hosts, model_config,
                             eval_threshold, n_extra_classes=0, stride=None,
                             merge_ood_from=None,
                             ood_calib_true_families=None, ood_calib_weights=None,
                             ood_eval_loader=None, ood_target_reject_rate=0.05,
                             ood_shrinkage_lambda=200.0, ood_score_agg='mean',
                             scramble_loader=None, ood_tau=0.95, ood_n0=0.0,
                             ood_normalize=False):
    """Production-model calibration: fit the OOD/reliability block on the
    discard calibration set, and splice the logit-space calibration
    (temperature/FDR) from a validation-model calibration.json rather than
    recomputing it on memorized data. Writes calibration.json + the sidecar.
    """
    if fit_loader is None or ood_calib_loader is None:
        raise ValueError(
            "OOD-only calibration needs a fit loader (train phages) and the "
            "discard calibration set; neither may be None.")
    calib_logits, calib_labels = _gather_logits_labels(
        model, ood_calib_loader, device, unpack_fn)
    n_host = calib_logits.shape[1] - n_extra_classes

    ext = None
    host_T = 1.0
    if merge_ood_from is not None:
        with open(merge_ood_from) as fh:
            ext = json.load(fh)
        _assert_calib_compatible(ext, hosts, model_config, stride)
        host_T = ext.get('temperature_host') or ext.get('temperature') or 1.0
        logger.info(f"  Merging logit-space calibration from {merge_ood_from} "
                    f"(temperature + FDR thresholds).")
    else:
        logger.warning("  No --merge_ood_from given: writing OOD-only "
                       "calibration (default threshold, no temperature/FDR). "
                       "predict will use raw sigmoid + eval_threshold.")

    ood_json = _run_ood(
        model, fit_loader, ood_calib_loader, device, n_host, hosts, run_dir,
        target_reject_rate=ood_target_reject_rate,
        shrinkage_lambda=ood_shrinkage_lambda, score_agg=ood_score_agg,
        tani=None, tani_cutoff=0.5,
        val_logits=calib_logits, val_labels=calib_labels,
        host_temperature=host_T, scramble_loader=scramble_loader,
        tau=ood_tau, n0=ood_n0, normalize=ood_normalize,
        true_families=ood_calib_true_families, id_weights=ood_calib_weights,
        ood_eval_loader=ood_eval_loader)

    data = _assemble_calibration_dict(hosts, model_config, ood_json,
                                      eval_threshold, stride, ext=ext)
    with open(os.path.join(run_dir, 'calibration.json'), 'w') as fh:
        json.dump(data, fh, indent=2)
    logger.info(f"  calibration saved -> {os.path.join(run_dir, 'calibration.json')}")
    return data


def run_calibration(model, loader, device, unpack_fn, run_dir, hosts,
                    model_config, eval_threshold, min_val_precision,
                    min_val_support, stride=None, n_extra_classes=0,
                    tani=None, tani_cutoff=0.5, block_max_fdr=0.5,
                    fit_loader=None, fit_ood=True,
                    ood_target_reject_rate=0.05, ood_shrinkage_lambda=200.0,
                    ood_score_agg='mean', scramble_loader=None,
                    ood_tau=0.95, ood_n0=0.0, reliability_features='full',
                    ood_normalize=False, ood_calib_loader=None,
                    ood_calib_true_families=None, ood_calib_weights=None,
                    ood_eval_loader=None):
    """Temperature + FDR-threshold calibration, then OOD-detector calibration.

    Temperature (single or split host/bacterial) and FDR thresholds are fit
    exactly as before. Per-class *blocking* has been replaced by an
    out-of-distribution detector: a class-conditional Mahalanobis model is fit
    on the training-phage patch-embedding manifold (via ``fit_loader``), its
    rejection threshold is calibrated on the validation set to a target ID
    rejection rate, and its behaviour vs genetic distance is reported using
    ``tani``. The idea is that sequence-level rejection subsumes what blocking
    was for — silencing untrustworthy outputs — without permanently killing a
    class. ``blocked_classes`` is still written (empty) for backward
    compatibility with older ``predict.py``.

    If ``tani`` (per-phage max tANI to training, aligned to the leading phage
    rows) is given, the HOST temperature and FDR thresholds are computed on the
    distant phage subset (tANI <= ``tani_cutoff``); the bacterial temperature is
    fit on the full set. ``min_val_precision`` / ``min_val_support`` /
    ``block_max_fdr`` are retained for signature compatibility but no longer
    drive any blocking.
    """
    T_host = None
    T_bact = None

    if tani is not None:
        logits, labels = _gather_logits_labels(model, loader, device, unpack_fn)
        n_host = logits.shape[1] - n_extra_classes

        tani_arr = np.asarray(tani, dtype=float)
        n_phage = len(tani_arr)
        n_nan = int((~np.isfinite(tani_arr)).sum())
        keep = np.isfinite(tani_arr) & (tani_arr <= tani_cutoff)
        if keep.sum() == 0:
            logger.warning("  No val phages pass the tANI cutoff; using all "
                           "phage rows for calibration.")
            keep = np.isfinite(tani_arr)
            if keep.sum() == 0:
                keep = np.ones(n_phage, dtype=bool)
        n_keep = int(keep.sum())
        logger.info(f"  Distance-stratified calibration: {n_keep}/{n_phage} "
                    f"val phages with tANI <= {tani_cutoff} "
                    f"({n_nan} had no tANI)")

        keep_idx = torch.from_numpy(np.nonzero(keep)[0])
        sub_logits = logits[:n_phage][keep_idx]      # distant phage rows
        sub_labels = labels[:n_phage][keep_idx]

        # Host temperature on the distant phage subset (host columns).
        T_host = _optimize_temperature(
            sub_logits[:, :n_host], sub_labels[:, :n_host])

        if n_extra_classes > 0:
            # Bacterial temperature on the FULL set — phages are its negatives.
            T_bact = _optimize_temperature(logits[:, n_host:], labels[:, n_host:])
            n_bact_pos = int((labels[:, n_host:] > 0).sum().item())
            logger.info(f"  T_host={T_host:.4f} (on {n_keep} distant phages)  "
                        f"T_bact={T_bact:.4f} (full set: {n_bact_pos} bacterial "
                        f"positives + {labels.shape[0]-n_bact_pos} phage "
                        f"negatives)")
            T = torch.full((logits.shape[1],), T_host)
            T[n_host:] = T_bact
        else:
            logger.info(f"  T={T_host:.4f} (on {n_keep} distant phages)")
            T = T_host

        fdr_thresholds = find_fdr_thresholds(sub_logits, sub_labels, T)
    else:
        # ---- global (no-tANI) temperature + FDR path ---------------------
        if n_extra_classes > 0:
            T_host, T_bact, T_vec, logits, labels = calibrate_temperature_split(
                model, loader, device, unpack_fn,
                n_extra_classes=n_extra_classes)
            T = T_vec
        else:
            T, logits, labels = calibrate_temperature(
                model, loader, device, unpack_fn)
        n_host = logits.shape[1] - n_extra_classes
        fdr_thresholds = find_fdr_thresholds(logits, labels, T)

    # ---- OOD detector replaces class blocking ----------------------------
    ood_json = None
    if fit_ood and fit_loader is not None:
        host_T = T_host if T_host is not None else (
            T if isinstance(T, float) else float(T[0]))
        # ID-calibration set for the OOD/reliability fit: the discard-based set
        # (discard_id + same-family OOD) when provided, else the val set.
        # Temperature/FDR above stay on the val set regardless.
        if ood_calib_loader is not None:
            calib_loader = ood_calib_loader
            calib_logits, calib_labels = _gather_logits_labels(
                model, ood_calib_loader, device, unpack_fn)
            logger.info(f"  OOD calibration on discard set "
                        f"({calib_logits.shape[0]} rows) with family-level "
                        f"correctness; temperature/FDR stay on val.")
        else:
            calib_loader, calib_logits, calib_labels = loader, logits, labels

        ood_json = _run_ood(
            model, fit_loader, calib_loader, device, n_host, hosts, run_dir,
            target_reject_rate=ood_target_reject_rate,
            shrinkage_lambda=ood_shrinkage_lambda, score_agg=ood_score_agg,
            tani=(tani if ood_calib_loader is None else None),
            tani_cutoff=tani_cutoff,
            val_logits=calib_logits, val_labels=calib_labels,
            host_temperature=host_T,
            scramble_loader=scramble_loader, tau=ood_tau, n0=ood_n0,
            reliability_features=reliability_features, normalize=ood_normalize,
            true_families=ood_calib_true_families, id_weights=ood_calib_weights,
            ood_eval_loader=ood_eval_loader)
    elif fit_ood:
        logger.warning("  fit_ood requested but no fit_loader given; "
                       "skipping OOD detector.")

    T_scalar = T_host if T_host is not None else (
        T if isinstance(T, float) else float(T[0]))
    save_calibration(
        os.path.join(run_dir, 'calibration.json'),
        temperature=T_scalar, hosts=hosts, model_config=model_config,
        threshold=eval_threshold, fdr_thresholds=fdr_thresholds,
        blocked_classes=[], stride=stride,
        temperature_host=T_host, temperature_bacterial=T_bact,
        ood=ood_json,
    )
