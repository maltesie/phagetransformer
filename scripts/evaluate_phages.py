#!/usr/bin/env python3
"""Evaluate a trained PhageTransformer model on phage host prediction.

Standalone script — not part of the installed package.

Usage:
    python evaluate_phages.py --model_dir ./models/my_training_run
"""

import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phagetransformer.model import CodonTokenizer
from phagetransformer.predict import load_model_and_calibration
from phagetransformer.dataset import (
    load_phage_host_merged, load_phage_host_test,
    PatchSequenceDataset, sequence_collate_fn,
)
from eval_utils import (
    COLORS, METRIC_COLORS, LEVEL_NAMES, FONT_SIZES, PRESENTATION_MODE,
    setup_style, _add_panel_letters, _output_path, _suptitle,
    enable_presentation_mode,
    collect_logits, evaluate_all_levels, compute_real_support,
    parse_lineages, aggregate_to_level,
)

logger = logging.getLogger(__name__)


def _save_figure(fig, out_path: str, dpi: int = 200):
    """Save figure as PNG, PDF, and SVG."""
    fig.savefig(out_path, dpi=dpi)
    logger.info(f"Figure saved: {out_path}")
    root = os.path.splitext(out_path)[0]
    for ext in ('.pdf', '.svg'):
        path = root + ext
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved: {path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def load_training_history(run_dir: str) -> Optional[pd.DataFrame]:
    """Load metrics.csv from the run's log directory."""
    log_dir = os.path.join(run_dir, 'logs')
    csv_path = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(csv_path):
        logger.warning(f"  metrics.csv not found at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if col not in ('phase',):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    'encoder':      COLORS['tertiary'],
    'frame_stats':  COLORS['secondary'],
    'aggregator':   COLORS['primary'],
}
PHASE_LABELS = {
    'encoder':      'Encoder',
    'frame_stats':  'Frame Stats',
    'aggregator':   'Aggregator',
}


def plot_loss_curves(ax, history: pd.DataFrame):
    """Training and validation loss across phases."""
    for phase in history['phase'].unique():
        ph = history[history['phase'] == phase]
        c = PHASE_COLORS.get(phase, COLORS['text_light'])
        label = PHASE_LABELS.get(phase, phase)
        ax.plot(ph['epoch'], ph['train_loss'], color=c,
                linewidth=1.5, label=f'{label} train')
        if 'val_loss' in ph.columns and ph['val_loss'].notna().any():
            ax.plot(ph['epoch'], ph['val_loss'], color=c,
                    linewidth=1.5, linestyle='--', alpha=0.7,
                    label=f'{label} val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & validation loss')
    ax.legend(fontsize=FONT_SIZES['legend_small'], ncol=2, loc='upper right')


def plot_f1_curves(ax, history: pd.DataFrame):
    """Validation micro/macro F1 across phases."""
    for phase in history['phase'].unique():
        ph = history[history['phase'] == phase]
        c = PHASE_COLORS.get(phase, COLORS['text_light'])
        label = PHASE_LABELS.get(phase, phase)
        if 'val_micro_f1' in ph.columns and ph['val_micro_f1'].notna().any():
            ax.plot(ph['epoch'], ph['val_micro_f1'], color=c,
                    linewidth=1.5, label=f'{label} micro')
        if 'val_macro_f1' in ph.columns and ph['val_macro_f1'].notna().any():
            ax.plot(ph['epoch'], ph['val_macro_f1'], color=c,
                    linewidth=1.5, linestyle='--', alpha=0.7,
                    label=f'{label} macro')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 score')
    ax.set_title('Validation F1')
    ax.legend(fontsize=FONT_SIZES['legend_small'], ncol=2, loc='lower right')


def plot_calibration(ax, logits: torch.Tensor, labels: torch.Tensor,
                     temperature: float, n_bins: int = 15):
    """Reliability diagram for calibrated probabilities."""
    probs = torch.sigmoid(logits / temperature).numpy().ravel()
    truth = labels.numpy().ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_acc, bin_conf, bin_count = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_acc.append(truth[mask].mean())
            bin_conf.append(probs[mask].mean())
            bin_count.append(mask.sum())
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_count.append(0)

    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_count = np.array(bin_count)
    valid = ~np.isnan(bin_acc)

    ax.plot([0, 1], [0, 1], '--', color=COLORS['grid'], linewidth=1,
            label='Perfect', zorder=1)
    ax.plot(bin_conf[valid], bin_acc[valid], 'o-',
            color=COLORS['primary'], markersize=4, linewidth=1.5,
            label=f'T={temperature:.2f}', zorder=3)

    ax2 = ax.twinx()
    ax2.bar(bin_centers, bin_count, width=1 / n_bins * 0.8,
            alpha=0.15, color=COLORS['quaternary'], zorder=0)
    ax2.set_ylabel('Count', fontsize=FONT_SIZES['secondary_axis'],
                   color=COLORS['text_light'])
    ax2.tick_params(axis='y', labelsize=FONT_SIZES['secondary_axis'],
                    colors=COLORS['text_light'])
    ax2.set_ylim(0, max(bin_count) * 4)

    weights = bin_count[valid] / bin_count[valid].sum()
    ece = (weights * np.abs(bin_acc[valid] - bin_conf[valid])).sum()
    ax.text(0.05, 0.92, f'ECE = {ece:.4f}', transform=ax.transAxes,
            fontsize=FONT_SIZES['annotation'], color=COLORS['text'])

    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Calibration')
    ax.legend(fontsize=FONT_SIZES['legend_small'], loc='lower right')


def plot_taxonomic_metrics(ax, level_results: Dict[str, Dict]):
    """Grouped bar chart of metrics across taxonomic levels."""
    levels = list(level_results.keys())
    metrics = ['micro_f1', 'macro_f1', 'micro_p', 'micro_r']
    labels = ['Micro F1', 'Macro F1', 'Precision', 'Recall']
    colors = [METRIC_COLORS[m] for m in metrics]

    x = np.arange(len(levels))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [level_results[lvl][metric] for lvl in levels]
        bars = ax.bar(x + offsets[i], values, width, label=label,
                      color=color, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=FONT_SIZES['bar_value'], color=COLORS['text_light'])

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=FONT_SIZES['tick_small'])
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title('Metrics by taxonomic rank')
    ax.legend(fontsize=FONT_SIZES['legend_small'], ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, 1.0))


def plot_f1_distribution(ax, level_results: Dict[str, Dict]):
    """Violin plots of per-category F1 at each taxonomic rank."""
    levels = list(level_results.keys())
    palette = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
               COLORS['quaternary'], COLORS['quinary'], COLORS['senary']]

    data, positions, colors_list = [], [], []
    for i, lvl in enumerate(levels):
        f1 = level_results[lvl]['per_f1']
        support = level_results[lvl]['support']
        valid = support > 0
        if valid.any():
            data.append(f1[valid])
            positions.append(i)
            colors_list.append(palette[i % len(palette)])

    if data:
        parts = ax.violinplot(data, positions=positions, widths=0.6,
                              showmeans=False, showmedians=True,
                              showextrema=False)
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(colors_list[i])
            body.set_edgecolor(colors_list[i])
            body.set_alpha(0.45)
        parts['cmedians'].set_color(COLORS['text'])
        parts['cmedians'].set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for i, (d, pos) in enumerate(zip(data, positions)):
        n_pts = min(len(d), 2000)
        idx = rng.choice(len(d), n_pts, replace=False) if len(d) > 2000 \
            else np.arange(len(d))
        jitter = rng.uniform(-0.15, 0.15, n_pts)
        ax.scatter(pos + jitter, d[idx], s=4, alpha=0.3,
                   color=colors_list[i], zorder=3, linewidths=0)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=FONT_SIZES['tick_small'])
    ax.set_ylabel('Per-category F1')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('F1 distribution per taxonomic rank')


def plot_support_vs_f1(ax, level_results: Dict[str, Dict],
                       real_support: Optional[Dict[str, np.ndarray]] = None):
    """Scatter plot of class support vs. per-class F1."""
    genus = level_results.get('Genus', {})
    if not genus:
        ax.text(0.5, 0.5, 'No genus data', transform=ax.transAxes,
                ha='center', va='center', color=COLORS['text_light'])
        return

    per_f1 = genus['per_f1']
    has = genus['support'] > 0

    if real_support and 'Genus' in real_support:
        sup_all = real_support['Genus']
        xlabel = 'Host category support (genomes)'
    else:
        sup_all = genus['support'] * 5.0
        xlabel = 'Estimated support (val \u00d7 5)'

    f1_vals = per_f1[has]
    sup_vals = sup_all[has]

    ax.scatter(sup_vals, f1_vals, s=12, alpha=0.45,
               color=COLORS['primary'], edgecolors='none', zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Per-category F1')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Host category support vs. F1 (genus)')

    # Add a trend line (moving median in log-spaced bins)
    log_sup = np.log10(sup_vals.clip(min=1))
    bin_edges = np.linspace(log_sup.min() - 0.01, log_sup.max() + 0.01, 12)
    bin_centers, bin_medians = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (log_sup >= lo) & (log_sup < hi)
        if mask.sum() >= 3:
            bin_centers.append(10 ** ((lo + hi) / 2))
            bin_medians.append(np.median(f1_vals[mask]))
    if len(bin_centers) >= 2:
        ax.plot(bin_centers, bin_medians, 'o-',
                color=COLORS['secondary'], markersize=4, linewidth=1.5,
                label='Median F1', zorder=4)
        ax.legend(fontsize=FONT_SIZES['legend_small'], loc='lower right')

    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Distance-to-training panels
# ---------------------------------------------------------------------------

def _compute_any_pred_correctness(raw_logits, labels, temperature, pcut):
    """Per-sample prediction flags: correct if ANY above-threshold pred is true.

    Returns (has_prediction, is_correct) boolean tensors of shape (N,).
    """
    probs = torch.sigmoid(raw_logits / temperature)
    labels_bool = labels.bool()

    has_prediction = (probs >= pcut).any(dim=1)
    is_correct = ((probs >= pcut) & labels_bool).any(dim=1)

    return has_prediction, is_correct


def _plot_distance_bar(ax, distance_values, has_prediction, is_correct,
                       xlabel, title, pcut, binsize=0.1):
    """Stacked bar chart of prediction rate binned by a distance metric."""
    bins = [(round(i, 2), round(i + binsize, 2))
            for i in np.arange(0, 1, binsize)]

    bin_labels, pred_ratios, correct_ratios = [], [], []
    for lo, hi in bins:
        sel = (lo <= distance_values) & (distance_values <= hi)
        n = int(sel.sum())
        lo_pct = int(round(lo * 100))
        hi_pct = int(round(hi * 100))
        bin_labels.append(f'{lo_pct}\u2013{hi_pct}\n{n}')
        if n == 0:
            pred_ratios.append(0.0)
            correct_ratios.append(0.0)
        else:
            pred_ratios.append(float(has_prediction[sel].sum()) / n)
            correct_ratios.append(float(is_correct[sel].sum()) / n)

    x = np.arange(len(bins))
    ax.bar(x, pred_ratios, color=COLORS['primary'],
           label=f'Has prediction (score \u2265 {pcut:.2f})')
    ax.bar(x, correct_ratios, color=COLORS['secondary'],
           label=f'Correct prediction (score \u2265 {pcut:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=FONT_SIZES['tick_tiny'])
    ax.set_ylabel('Ratio of test genomes')
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(fontsize=FONT_SIZES['legend_small'], loc='lower right')
    ax.grid(alpha=0.3)


def plot_distance_panels(ax_prot, ax_ani, metadata_df,
                         raw_logits, labels, temperature, pcut,
                         binsize=0.1):
    """Performance vs. distance to training data on two axes."""
    has_prediction, is_correct = _compute_any_pred_correctness(
        raw_logits, labels, temperature, pcut)

    n_unique = np.array(metadata_df['n_unique_proteins'], dtype=int)
    n_total = np.array(metadata_df['n_total_proteins'], dtype=int)
    ratio_shared = 1.0 - n_unique / n_total
    max_ani = np.array(metadata_df['max_pt_train_tani'], dtype=float)

    _plot_distance_bar(
        ax_prot, ratio_shared, has_prediction, is_correct,
        xlabel='Proteins close to PT training, 95% AAI (in %)',
        title='Performance vs. shared proteins',
        pcut=pcut, binsize=binsize)

    _plot_distance_bar(
        ax_ani, max_ani, has_prediction, is_correct,
        xlabel='Max. tANI to PT training (in %)',
        title='Performance vs. max tANI',
        pcut=pcut, binsize=binsize)


# ---------------------------------------------------------------------------
# Standard genetic code
# ---------------------------------------------------------------------------

CODON_TO_AA = {
    'TTT': 'Phe', 'TTC': 'Phe', 'TTA': 'Leu', 'TTG': 'Leu',
    'CTT': 'Leu', 'CTC': 'Leu', 'CTA': 'Leu', 'CTG': 'Leu',
    'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile', 'ATG': 'Met',
    'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val',
    'TCT': 'Ser', 'TCC': 'Ser', 'TCA': 'Ser', 'TCG': 'Ser',
    'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    'GCT': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    'TAT': 'Tyr', 'TAC': 'Tyr', 'TAA': 'Stop', 'TAG': 'Stop',
    'CAT': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
    'AAT': 'Asn', 'AAC': 'Asn', 'AAA': 'Lys', 'AAG': 'Lys',
    'GAT': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
    'TGT': 'Cys', 'TGC': 'Cys', 'TGA': 'Stop', 'TGG': 'Trp',
    'CGT': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
    'AGT': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
    'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
}

# 20 amino acids + Stop -> distinct colors
_AA_PALETTE = {
    'Ala': '#E41A1C', 'Arg': '#377EB8', 'Asn': '#4DAF4A', 'Asp': '#984EA3',
    'Cys': '#FF7F00', 'Gln': '#A65628', 'Glu': '#F781BF', 'Gly': '#999999',
    'His': '#66C2A5', 'Ile': '#FC8D62', 'Leu': '#8DA0CB', 'Lys': '#E78AC3',
    'Met': '#A6D854', 'Phe': '#FFD92F', 'Pro': '#E5C494', 'Ser': '#B3B3B3',
    'Thr': '#1B9E77', 'Trp': '#D95F02', 'Tyr': '#7570B3', 'Val': '#E7298A',
    'Stop': '#2D2D2D',
}


def _codon_gc(codon: str) -> float:
    """GC fraction of a 3-letter codon."""
    return sum(1 for c in codon if c in 'GC') / 3.0


def silhouette_score(X: np.ndarray, labels: np.ndarray,
                     D: Optional[np.ndarray] = None) -> float:
    """Mean silhouette score measuring how well points cluster by label.

    For each point i with label L:
      a(i) = mean distance to other points with the same label L
      b(i) = min over other labels L' of mean distance to points with label L'
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Returns mean s(i) over all points.  Range: [-1, 1], higher = better
    label-based clustering.  Points in singleton groups get s = 0.

    If ``D`` (precomputed distance matrix) is provided, ``X`` is ignored.
    """
    from scipy.spatial.distance import cdist
    if D is None:
        D = cdist(X, X, metric='euclidean')
    unique_labels = np.unique(labels)
    n = len(labels)
    sil = np.zeros(n)

    for i in range(n):
        same = labels == labels[i]
        n_same = same.sum()
        if n_same <= 1:
            sil[i] = 0.0
            continue
        a_i = D[i, same].sum() / (n_same - 1)

        b_i = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other = labels == lbl
            if other.sum() == 0:
                continue
            b_i = min(b_i, D[i, other].mean())

        sil[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)

    return float(sil.mean())


def silhouette_with_pvalue(X: np.ndarray, labels: np.ndarray,
                           n_perm: int = 1000, seed: int = 42) -> tuple:
    """Silhouette score with permutation-test p-value.

    Shuffles labels ``n_perm`` times and computes the fraction of
    permuted scores >= observed score.

    Returns (score, p_value).
    """
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    D = cdist(X, X, metric='euclidean')  # precompute once
    observed = silhouette_score(X, labels, D)

    count_ge = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        if silhouette_score(X, perm_labels, D) >= observed:
            count_ge += 1

    p = (count_ge + 1) / (n_perm + 1)   # conservative estimate
    return observed, float(p)


# ---------------------------------------------------------------------------
# Figure 3 — Codon embedding PCA
# ---------------------------------------------------------------------------

def make_embedding_figure(model, out_path: str, dpi: int = 200,
                          show_mds: bool = False):
    """PCA (and optionally MDS cosine) of learned codon embeddings."""
    setup_style()
    from scipy.spatial.distance import cdist as _cdist

    # Extract embedding weights (skip PAD=0 and UNK=1)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    emb_weight = raw_model.patch_encoder.frame_cnn.embedding.weight.detach().cpu()
    codon_embs = emb_weight[2:].numpy()   # (64, embed_dim)

    # Build codon list in same order as tokenizer
    nucs = ['A', 'C', 'G', 'T']
    codons = [a + b + c for a in nucs for b in nucs for c in nucs]
    assert len(codons) == 64

    # PCA via SVD
    centered = codon_embs - codon_embs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc = centered @ Vt[:2].T                       # (64, 2)
    var_explained = S[:2] ** 2 / (S ** 2).sum()

    # MDS on cosine distances (optional)
    mds, stress = None, None
    if show_mds:
        D_cos = _cdist(codon_embs, codon_embs, metric='cosine')
        D_cos = np.clip(D_cos, 0, None)
        D2 = D_cos ** 2
        n = D2.shape[0]
        H_mat = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H_mat @ D2 @ H_mat
        eigvals, eigvecs = np.linalg.eigh(B)
        idx_top = np.argsort(eigvals)[::-1][:2]
        mds = eigvecs[:, idx_top] * np.sqrt(np.maximum(eigvals[idx_top], 0))
        D_mds = _cdist(mds, mds, metric='euclidean')
        stress = np.sqrt(
            ((D_cos - D_mds) ** 2).sum() / (D_cos ** 2).sum())

    # Amino acid labels
    aa_labels = [CODON_TO_AA[c] for c in codons]
    aa_label_arr = np.array(aa_labels)
    unique_aa = sorted(set(aa_labels), key=lambda x: (x == 'Stop', x))

    # Silhouette score — sweep over 1..5 PCs, keep the maximum
    sil_scores = []
    max_pcs = min(5, centered.shape[1])
    for k in range(2, max_pcs + 1):
        pc_k = centered @ Vt[:k].T
        sil_scores.append(silhouette_score(pc_k, aa_label_arr))
    best_k = int(np.argmax(sil_scores)) + 1
    # Compute p-value only for the best
    pc_best = centered @ Vt[:best_k].T
    sil, sil_p = silhouette_with_pvalue(pc_best, aa_label_arr)

    # Purine fraction per codon: fraction of A/G nucleotides (0, 1/3, 2/3, 1)
    purine_fraction = np.array([sum(1 for nt in c if nt in 'AG') / 3.0
                                for c in codons])

    # Shared plotting helpers
    def _plot_aa(ax, coords):
        for aa in unique_aa:
            mask = [i for i, a in enumerate(aa_labels) if a == aa]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=_AA_PALETTE.get(aa, '#888888'),
                       s=50, alpha=0.85, edgecolors='white', linewidths=0.4,
                       label=aa, zorder=3)
        for i, codon in enumerate(codons):
            ax.annotate(codon, (coords[i, 0], coords[i, 1]),
                        fontsize=FONT_SIZES['codon'], ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        color=COLORS['text_light'])

    def _plot_purine(ax, coords):
        """Plot codons colored by purine fraction (continuous)."""
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=purine_fraction, cmap='RdYlBu_r',
                        vmin=0, vmax=1,
                        s=50, alpha=0.85,
                        edgecolors='white', linewidths=0.4, zorder=3)
        for i, codon in enumerate(codons):
            ax.annotate(codon, (coords[i, 0], coords[i, 1]),
                        fontsize=FONT_SIZES['codon'], ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        color=COLORS['text_light'])
        return sc

    # ---- Compute property–PC correlations for summary panel ----
    from scipy.stats import spearmanr as _spearmanr

    n_summary_pcs = min(10, centered.shape[1])
    all_var = S ** 2 / (S ** 2).sum()
    pcs_all = centered @ Vt[:n_summary_pcs].T

    prop_matrix, prop_names = get_curated_properties(codons)
    n_props = len(prop_names)

    # For each PC find the most-correlated property
    best_prop_idx = []
    best_rho = []
    best_pval = []
    for j in range(n_summary_pcs):
        top_idx, top_r, top_p = 0, 0.0, 1.0
        for i in range(n_props):
            if prop_matrix[:, i].std() < 1e-12:
                continue
            r, p = _spearmanr(prop_matrix[:, i], pcs_all[:, j])
            if abs(r) > abs(top_r):
                top_idx, top_r, top_p = i, r, p
        best_prop_idx.append(top_idx)
        best_rho.append(top_r)
        best_pval.append(top_p)

    # --- Layout: 1×3 (or 2×3 with MDS) ---
    nrows = 2 if show_mds else 1
    fig = plt.figure(figsize=(22, 6.5 * nrows))
    gs = gridspec.GridSpec(nrows, 3, figure=fig, hspace=0.22,
                           wspace=0.38, left=0.06, right=0.97,
                           top=0.90, bottom=0.08 if show_mds else 0.10)

    # --- Panel A: summary bar chart ---
    ax_summary = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(n_summary_pcs)
    bar_colors = [PROPERTY_GROUP_COLORS.get(prop_group(prop_names[idx]), '#999')
                  for idx in best_prop_idx]
    bars = ax_summary.barh(y_pos, [abs(r) for r in best_rho],
                           color=bar_colors, edgecolor='white',
                           linewidth=0.5, height=0.7)
    # Sign indicator: positive or negative correlation
    for yi, (r, pv) in enumerate(zip(best_rho, best_pval)):
        sign = '+' if r > 0 else '−'
        stars = '***' if pv < 0.001 else ('**' if pv < 0.01
                                          else ('*' if pv < 0.05 else ''))
        ax_summary.text(abs(r) + 0.01, yi,
                        f'{sign} {prop_names[best_prop_idx[yi]]}{stars}',
                        va='center', fontsize=FONT_SIZES['bar_label'],
                        color=COLORS['text'])
    ax_summary.set_yticks(y_pos)
    ax_summary.set_yticklabels(
        [f'PC{j+1} ({all_var[j]:.1%})' for j in range(n_summary_pcs)],
        fontsize=FONT_SIZES['tick_small'])
    ax_summary.invert_yaxis()
    ax_summary.set_xlabel('|Spearman ρ|', fontsize=FONT_SIZES['label'])
    ax_summary.set_xlim(0, 1.0)
    ax_summary.set_title('Top correlated property per PC',
                         fontsize=FONT_SIZES['title'], fontweight='bold')
    ax_summary.grid(alpha=0.3, axis='x')

    # Mini legend for property groups (only groups actually present)
    from matplotlib.patches import Patch as _Patch
    seen_groups = sorted(set(prop_group(prop_names[idx])
                             for idx in best_prop_idx))
    grp_patches = [_Patch(facecolor=PROPERTY_GROUP_COLORS.get(g, '#999'), label=g)
                   for g in seen_groups]
    ax_summary.legend(handles=grp_patches,
                      fontsize=FONT_SIZES['bar_label'], loc='lower right',
                      frameon=True, framealpha=0.9, edgecolor=COLORS['grid'])

    # --- Panel B: PCA — amino acid ---
    ax_pca_aa = fig.add_subplot(gs[0, 1])
    _plot_aa(ax_pca_aa, pc)
    ax_pca_aa.set_xlabel(f'PC1 ({var_explained[0]:.1%} var.)')
    ax_pca_aa.set_ylabel(f'PC2 ({var_explained[1]:.1%} var.)')
    ax_pca_aa.set_title('PCA — amino acid')
    p_str = f'{sil_p:.1e}' if sil_p < 0.001 else f'{sil_p:.3f}'
    ax_pca_aa.text(0.97, 0.97,
                   f'Silhouette = {sil:.3f} ({best_k} PCs)\np = {p_str}',
                   transform=ax_pca_aa.transAxes,
                   fontsize=FONT_SIZES['annotation'], color=COLORS['text'],
                   va='top', ha='right')

    # --- Panel C: PCA — purine fraction ---
    ax_pca_pur = fig.add_subplot(gs[0, 2])
    sc = _plot_purine(ax_pca_pur, pc)
    ax_pca_pur.set_xlabel(f'PC1 ({var_explained[0]:.1%} var.)')
    ax_pca_pur.set_ylabel(f'PC2 ({var_explained[1]:.1%} var.)')
    ax_pca_pur.set_title('PCA — purine fraction')
    cbar = fig.colorbar(sc, ax=ax_pca_pur, shrink=0.7, pad=0.02)
    cbar.set_label('Purine fraction', fontsize=FONT_SIZES['colorbar'])

    all_axes = [ax_summary, ax_pca_aa, ax_pca_pur]

    # Row 1: MDS (cosine) — optional
    if show_mds:
        ax_mds_blank = fig.add_subplot(gs[1, 0])
        ax_mds_blank.set_axis_off()  # no summary equivalent for MDS row

        ax_mds_aa = fig.add_subplot(gs[1, 1])
        _plot_aa(ax_mds_aa, mds)
        ax_mds_aa.set_xlabel('MDS dim 1')
        ax_mds_aa.set_ylabel('MDS dim 2')
        ax_mds_aa.set_title('MDS (cosine) — amino acid')
        ax_mds_aa.text(0.03, 0.97, f'Stress-1 = {stress:.3f}',
                       transform=ax_mds_aa.transAxes,
                       fontsize=FONT_SIZES['annotation'],
                       color=COLORS['text'], va='top')

        ax_mds_pur = fig.add_subplot(gs[1, 2])
        sc_mds = _plot_purine(ax_mds_pur, mds)
        ax_mds_pur.set_xlabel('MDS dim 1')
        ax_mds_pur.set_ylabel('MDS dim 2')
        ax_mds_pur.set_title('MDS (cosine) — purine fraction')
        fig.colorbar(sc_mds, ax=ax_mds_pur, shrink=0.7, pad=0.02).set_label(
            'Purine fraction', fontsize=FONT_SIZES['colorbar'])

        all_axes += [ax_mds_aa, ax_mds_pur]

    # Panel labels
    _add_panel_letters(all_axes, 'ABCDE', x=-0.06, y=1.06)

    # Shared amino acid legend — right of panel B
    handles, labels = ax_pca_aa.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FONT_SIZES['legend_small'], ncol=1,
               loc='center left',
               bbox_to_anchor=(0.67, 0.5 if not show_mds else 0.75),
               frameon=True, framealpha=0.9, edgecolor=COLORS['grid'],
               handletextpad=0.4)

    embed_dim = codon_embs.shape[1]
    title = (f'Learned codon embeddings  ({embed_dim}-dim,  '
             f'AA silhouette = {sil:.3f} @ {best_k} PCs, p = {p_str}')
    if show_mds:
        title += f',  MDS cosine stress = {stress:.3f}'
    title += ')'
    _suptitle(fig, title, y=0.97)

    _save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Codon property matrix & PC correlation analysis
# ---------------------------------------------------------------------------

# Amino acid physicochemical properties (standard scales)
_AA_HYDROPHOBICITY = {  # Kyte-Doolittle
    'Ala': 1.8, 'Arg': -4.5, 'Asn': -3.5, 'Asp': -3.5, 'Cys': 2.5,
    'Gln': -3.5, 'Glu': -3.5, 'Gly': -0.4, 'His': -3.2, 'Ile': 4.5,
    'Leu': 3.8, 'Lys': -3.9, 'Met': 1.9, 'Phe': 2.8, 'Pro': -1.6,
    'Ser': -0.8, 'Thr': -0.7, 'Trp': -0.9, 'Tyr': -1.3, 'Val': 4.2,
    'Stop': 0.0,
}
_AA_MW = {  # molecular weight (Da)
    'Ala': 89.09, 'Arg': 174.20, 'Asn': 132.12, 'Asp': 133.10, 'Cys': 121.16,
    'Gln': 146.15, 'Glu': 147.13, 'Gly': 75.03, 'His': 155.16, 'Ile': 131.17,
    'Leu': 131.17, 'Lys': 146.19, 'Met': 149.21, 'Phe': 165.19, 'Pro': 115.13,
    'Ser': 105.09, 'Thr': 119.12, 'Trp': 204.23, 'Tyr': 181.19, 'Val': 117.15,
    'Stop': 0.0,
}
_AA_VOLUME = {  # van der Waals volume (Å³)
    'Ala': 67, 'Arg': 148, 'Asn': 96, 'Asp': 91, 'Cys': 86,
    'Gln': 114, 'Glu': 109, 'Gly': 48, 'His': 118, 'Ile': 124,
    'Leu': 124, 'Lys': 135, 'Met': 124, 'Phe': 135, 'Pro': 90,
    'Ser': 73, 'Thr': 93, 'Trp': 163, 'Tyr': 141, 'Val': 105,
    'Stop': 0,
}
_AA_PI = {  # isoelectric point
    'Ala': 6.00, 'Arg': 10.76, 'Asn': 5.41, 'Asp': 2.77, 'Cys': 5.07,
    'Gln': 5.65, 'Glu': 3.22, 'Gly': 5.97, 'His': 7.59, 'Ile': 6.02,
    'Leu': 5.98, 'Lys': 9.74, 'Met': 5.74, 'Phe': 5.48, 'Pro': 6.30,
    'Ser': 5.68, 'Thr': 5.60, 'Trp': 5.89, 'Tyr': 5.66, 'Val': 5.96,
    'Stop': 7.0,
}
_AA_CHARGE_PH7 = {  # net charge at pH 7
    'Arg': 1.0, 'Lys': 1.0, 'His': 0.1, 'Asp': -1.0, 'Glu': -1.0,
}
_AA_POLARITY = {  # Zimmerman polarity
    'Ala': 0.0, 'Arg': 52.0, 'Asn': 3.38, 'Asp': 49.7, 'Cys': 1.48,
    'Gln': 3.53, 'Glu': 49.9, 'Gly': 0.0, 'His': 51.6, 'Ile': 0.13,
    'Leu': 0.13, 'Lys': 49.5, 'Met': 1.43, 'Phe': 0.35, 'Pro': 1.58,
    'Ser': 1.67, 'Thr': 1.66, 'Trp': 2.10, 'Tyr': 1.61, 'Val': 0.13,
    'Stop': 0.0,
}

# Codon degeneracy: how many codons encode each amino acid
_AA_DEGENERACY = {
    'Ala': 4, 'Arg': 6, 'Asn': 2, 'Asp': 2, 'Cys': 2,
    'Gln': 2, 'Glu': 2, 'Gly': 4, 'His': 2, 'Ile': 3,
    'Leu': 6, 'Lys': 2, 'Met': 1, 'Phe': 2, 'Pro': 4,
    'Ser': 6, 'Thr': 4, 'Trp': 1, 'Tyr': 2, 'Val': 4,
    'Stop': 3,
}

# Curated property whitelist — used consistently across all figures
CURATED_PROPERTIES = [
    # Sequence composition
    'GC_content', 'purine_fraction',
    'GC_pos1', 'GC_pos2', 'GC_pos3',
    'purine_pos1', 'purine_pos2', 'purine_pos3',
    # Amino acid physicochemistry
    'aa_hydrophobicity', 'aa_charge_pH7',
    # Amino acid categories
    'aa_aromatic', 'aa_aliphatic', 'aa_sulfur',
    # Coding
    'degeneracy', 'is_stop',
]

PROPERTY_GROUPS = {
    'GC_content': 'Sequence', 'purine_fraction': 'Sequence',
    'GC_pos1': 'Sequence', 'GC_pos2': 'Sequence', 'GC_pos3': 'Sequence',
    'purine_pos1': 'Sequence', 'purine_pos2': 'Sequence',
    'purine_pos3': 'Sequence',
    'aa_hydrophobicity': 'Amino acid', 'aa_charge_pH7': 'Amino acid',
    'aa_aromatic': 'Amino acid', 'aa_aliphatic': 'Amino acid',
    'aa_sulfur': 'Amino acid',
    'degeneracy': 'Degeneracy', 'is_stop': 'Degeneracy',
}

PROPERTY_GROUP_COLORS = {
    'Sequence':   '#4E79A7',
    'Amino acid': '#E15759',
    'Degeneracy': '#59A14F',
}


def get_curated_properties(codons: List[str]):
    """Return (matrix, names) for the curated subset of codon properties."""
    full_matrix, full_names = build_codon_property_matrix(codons)
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    keep = [n for n in CURATED_PROPERTIES if n in name_to_idx]
    indices = [name_to_idx[n] for n in keep]
    return full_matrix[:, indices], keep


def prop_group(name: str) -> str:
    """Return the display group for a property name."""
    return PROPERTY_GROUPS.get(name, 'Other')


def build_codon_property_matrix(codons: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build a (64, N_properties) matrix of biologically meaningful codon features.

    Returns (matrix, property_names).
    """
    props = {}  # name -> array of length 64

    # --- Nucleotide composition ---
    props['GC_content'] = np.array([_codon_gc(c) for c in codons])
    props['purine_fraction'] = np.array(
        [sum(1 for nt in c if nt in 'AG') / 3.0 for c in codons])
    for pos in range(3):
        for nuc in 'ACGT':
            props[f'pos{pos+1}_{nuc}'] = np.array(
                [1.0 if c[pos] == nuc else 0.0 for c in codons])
    # GC at each position
    for pos in range(3):
        props[f'GC_pos{pos+1}'] = np.array(
            [1.0 if c[pos] in 'GC' else 0.0 for c in codons])
    # Purine (AG) vs pyrimidine (TC) at each position
    for pos in range(3):
        props[f'purine_pos{pos+1}'] = np.array(
            [1.0 if c[pos] in 'AG' else 0.0 for c in codons])

    # --- Dinucleotide content ---
    for di in ['CG', 'TA', 'GC', 'AT', 'TG', 'CA', 'AG', 'GA',
               'TC', 'CT', 'AC', 'GT', 'AA', 'TT', 'CC', 'GG']:
        vals = []
        for c in codons:
            count = int(c[0:2] == di) + int(c[1:3] == di)
            vals.append(float(count))
        props[f'dinuc_{di}'] = np.array(vals)

    # --- Wobble classification ---
    props['wobble_purine'] = np.array(
        [1.0 if c[2] in 'AG' else 0.0 for c in codons])
    props['wobble_transition_from_T'] = np.array(
        [1.0 if c[2] in 'TC' else 0.0 for c in codons])

    # --- Amino acid identity features ---
    aa_labels = [CODON_TO_AA[c] for c in codons]
    props['is_stop'] = np.array([1.0 if aa == 'Stop' else 0.0 for aa in aa_labels])
    props['is_start'] = np.array([1.0 if c == 'ATG' else 0.0 for c in codons])

    # Amino acid physicochemical properties
    props['aa_hydrophobicity'] = np.array(
        [_AA_HYDROPHOBICITY[aa] for aa in aa_labels])
    props['aa_mol_weight'] = np.array(
        [_AA_MW[aa] for aa in aa_labels])
    props['aa_volume'] = np.array(
        [float(_AA_VOLUME[aa]) for aa in aa_labels])
    props['aa_pI'] = np.array(
        [_AA_PI[aa] for aa in aa_labels])
    props['aa_charge_pH7'] = np.array(
        [_AA_CHARGE_PH7.get(aa, 0.0) for aa in aa_labels])
    props['aa_polarity'] = np.array(
        [_AA_POLARITY[aa] for aa in aa_labels])

    # Amino acid categorical features
    props['aa_aromatic'] = np.array(
        [1.0 if aa in ('Phe', 'Trp', 'Tyr') else 0.0 for aa in aa_labels])
    props['aa_aliphatic'] = np.array(
        [1.0 if aa in ('Ala', 'Val', 'Leu', 'Ile') else 0.0 for aa in aa_labels])
    props['aa_sulfur'] = np.array(
        [1.0 if aa in ('Cys', 'Met') else 0.0 for aa in aa_labels])
    props['aa_hydroxyl'] = np.array(
        [1.0 if aa in ('Ser', 'Thr', 'Tyr') else 0.0 for aa in aa_labels])
    props['aa_charged'] = np.array(
        [1.0 if aa in ('Arg', 'Lys', 'His', 'Asp', 'Glu') else 0.0
         for aa in aa_labels])
    props['aa_positive'] = np.array(
        [1.0 if aa in ('Arg', 'Lys', 'His') else 0.0 for aa in aa_labels])
    props['aa_negative'] = np.array(
        [1.0 if aa in ('Asp', 'Glu') else 0.0 for aa in aa_labels])
    props['aa_polar_uncharged'] = np.array(
        [1.0 if aa in ('Ser', 'Thr', 'Asn', 'Gln', 'Cys') else 0.0
         for aa in aa_labels])
    props['aa_small'] = np.array(
        [1.0 if aa in ('Gly', 'Ala', 'Ser', 'Pro') else 0.0
         for aa in aa_labels])
    props['aa_tiny'] = np.array(
        [1.0 if aa in ('Gly', 'Ala', 'Ser') else 0.0 for aa in aa_labels])

    # --- Degeneracy ---
    props['degeneracy'] = np.array(
        [float(_AA_DEGENERACY[aa]) for aa in aa_labels])
    _fourfold_prefixes = {'GC', 'GG', 'CC', 'AC', 'GT', 'TC', 'CT', 'CG'}
    props['fourfold_degenerate'] = np.array(
        [1.0 if c[:2] in _fourfold_prefixes else 0.0 for c in codons])
    props['twofold_degenerate'] = np.array(
        [1.0 if _AA_DEGENERACY[aa] == 2 else 0.0 for aa in aa_labels])

    # --- Molecular structure of codon ---
    _hbonds = {'A': 2, 'T': 2, 'G': 3, 'C': 3}
    props['total_hbonds'] = np.array(
        [float(sum(_hbonds[nt] for nt in c)) for c in codons])
    props['purine_purine_steps'] = np.array(
        [float((c[0] in 'AG' and c[1] in 'AG') +
               (c[1] in 'AG' and c[2] in 'AG'))
         for c in codons])

    # Build matrix
    names = list(props.keys())
    matrix = np.column_stack([props[n] for n in names])
    return matrix, names


def make_codon_property_correlation(model, out_path: str,
                                    tsv_path: Optional[str] = None,
                                    n_pcs: int = 30, dpi: int = 200):
    """Correlate codon properties with learned embedding PCs.

    Single heatmap of Spearman correlations (clustered properties x PCs)
    with variance-explained bars on top.  Nucleotide-level properties
    are excluded; only amino acid, degeneracy, and sequence-structure
    properties are shown.  P-values are FDR-corrected (Benjamini-Hochberg).
    """
    from scipy.stats import spearmanr
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    setup_style()

    # Extract embeddings
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    emb_weight = raw_model.patch_encoder.frame_cnn.embedding.weight.detach().cpu()
    codon_embs = emb_weight[2:].numpy()  # (64, embed_dim)

    nucs = ['A', 'C', 'G', 'T']
    codons = [a + b + c for a in nucs for b in nucs for c in nucs]

    # PCA
    centered = codon_embs - codon_embs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_pcs = min(n_pcs, centered.shape[1])
    pcs = centered @ Vt[:n_pcs].T  # (64, n_pcs)
    var_explained = S[:n_pcs] ** 2 / (S ** 2).sum()

    # Use curated property subset
    prop_matrix, prop_names = get_curated_properties(codons)
    n_props = len(prop_names)
    logger.info(f"  {n_props} curated properties")

    # Remove constant properties
    nonconstant = [i for i in range(n_props) if prop_matrix[:, i].std() > 1e-12]
    prop_matrix = prop_matrix[:, nonconstant]
    prop_names = [prop_names[i] for i in nonconstant]
    n_props = len(prop_names)

    # Compute Spearman correlations: (n_props, n_pcs)
    rho_matrix = np.zeros((n_props, n_pcs))
    pval_matrix = np.ones((n_props, n_pcs))
    for i in range(n_props):
        for j in range(n_pcs):
            rho, pval = spearmanr(prop_matrix[:, i], pcs[:, j])
            rho_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    # FDR correction (Benjamini-Hochberg) across all tests
    pvals_flat = pval_matrix.ravel()
    n_tests = len(pvals_flat)
    sort_idx = np.argsort(pvals_flat)
    ranks = np.empty_like(sort_idx)
    ranks[sort_idx] = np.arange(1, n_tests + 1)
    fdr_flat = np.minimum(1.0, pvals_flat * n_tests / ranks)
    # Enforce monotonicity
    fdr_sorted = fdr_flat[sort_idx]
    for k in range(n_tests - 2, -1, -1):
        fdr_sorted[k] = min(fdr_sorted[k], fdr_sorted[k + 1])
    fdr_flat[sort_idx] = fdr_sorted
    fdr_matrix = fdr_flat.reshape(n_props, n_pcs)

    # Export TSV
    if tsv_path:
        rows = []
        for i, pname in enumerate(prop_names):
            for j in range(n_pcs):
                rows.append({
                    'property': pname,
                    'PC': j + 1,
                    'var_explained': f'{var_explained[j]:.5f}',
                    'spearman_rho': f'{rho_matrix[i, j]:.5f}',
                    'p_value': f'{pval_matrix[i, j]:.2e}',
                    'fdr_q': f'{fdr_matrix[i, j]:.2e}',
                })
        df = pd.DataFrame(rows)
        df.to_csv(tsv_path, sep='\t', index=False)
        logger.info(f"Codon property correlations: {tsv_path}")

    # Cluster properties by correlation profile
    if n_props > 2:
        row_dist = pdist(rho_matrix, metric='correlation')
        row_dist = np.nan_to_num(row_dist, nan=1.0)
        row_link = linkage(row_dist, method='average')
        row_order = leaves_list(row_link)
    else:
        row_order = np.arange(n_props)

    rho_ordered = rho_matrix[row_order]
    fdr_ordered = fdr_matrix[row_order]
    names_ordered = [prop_names[i] for i in row_order]

    # --- Figure: variance bars on top + heatmap below ---
    fig_height = max(6, n_props * 0.32 + 2.5)
    fig = plt.figure(figsize=(max(12, n_pcs * 0.45 + 3), fig_height))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 6],
                           hspace=0.05, left=0.22, right=0.92,
                           top=0.93, bottom=0.06)

    # Variance explained bar chart (top)
    ax_var = fig.add_subplot(gs[0])
    ax_var.bar(range(n_pcs), var_explained * 100, color=COLORS['primary'],
               alpha=0.7, width=0.8, edgecolor='none')
    ax_var.set_xlim(-0.5, n_pcs - 0.5)
    ax_var.set_ylabel('% var.', fontsize=FONT_SIZES['secondary_axis'])
    ax_var.tick_params(axis='x', labelbottom=False)
    ax_var.tick_params(axis='y', labelsize=FONT_SIZES['tick_tiny'])
    ax_var.set_title('Codon properties vs. learned embedding PCs',
                     fontsize=FONT_SIZES['title'], fontweight='bold', pad=8)

    # Heatmap (bottom)
    ax_heat = fig.add_subplot(gs[1])
    im = ax_heat.imshow(rho_ordered, aspect='auto', cmap='RdBu_r',
                        vmin=-1, vmax=1, interpolation='nearest')

    ax_heat.set_xticks(range(n_pcs))
    ax_heat.set_xticklabels([f'PC{j+1}' for j in range(n_pcs)],
                            fontsize=FONT_SIZES['bar_value'], rotation=90)
    ax_heat.set_yticks(range(n_props))
    ax_heat.set_yticklabels(names_ordered, fontsize=FONT_SIZES['bar_label'])

    # Color y-tick labels by group
    for yi, name in enumerate(names_ordered):
        grp = prop_group(name)
        ax_heat.get_yticklabels()[yi].set_color(
            PROPERTY_GROUP_COLORS.get(grp, '#333'))

    # Significance stars (FDR-corrected)
    for i in range(n_props):
        for j in range(n_pcs):
            q = fdr_ordered[i, j]
            if q >= 0.05:
                continue
            stars = '***' if q < 0.001 else ('**' if q < 0.01 else '*')
            color = 'white' if abs(rho_ordered[i, j]) > 0.6 else 'black'
            ax_heat.text(j, i, stars, ha='center', va='center',
                        fontsize=4, color=color)

    ax_heat.set_xlabel('Principal component')

    # Colorbar
    cbar_ax = fig.add_axes([0.935, 0.06, 0.012, 0.55])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Spearman \u03c1', fontsize=FONT_SIZES['colorbar'])

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=PROPERTY_GROUP_COLORS[g], label=g)
                      for g in PROPERTY_GROUP_COLORS]
    fig.legend(handles=legend_patches, fontsize=FONT_SIZES['bar_label'],
               loc='lower left', bbox_to_anchor=(0.01, 0.01), ncol=3,
               frameon=True, framealpha=0.9, edgecolor=COLORS['grid'])

    _suptitle(fig,
        f'Spearman \u03c1  (* q<.05  ** q<.01  *** q<.001, BH-corrected)',
        y=0.98, fontsize=FONT_SIZES['annotation'])

    _save_figure(fig, out_path, dpi=dpi)

    # Export property matrix
    if tsv_path:
        prop_tsv = tsv_path.replace('_correlations.tsv', '_matrix.tsv')
        prop_df = pd.DataFrame(prop_matrix, columns=prop_names,
                                index=codons)
        prop_df.index.name = 'codon'
        aa_col = [CODON_TO_AA[c] for c in codons]
        prop_df.insert(0, 'amino_acid', aa_col)
        prop_df.to_csv(prop_tsv, sep='\t')
        logger.info(f"Codon property matrix: {prop_tsv}")

    return rho_matrix, fdr_matrix, prop_names, var_explained



# ---------------------------------------------------------------------------
# Figure 4 — Dataset overview
# ---------------------------------------------------------------------------

def plot_support_rank_curves(ax, level_results: Dict[str, Dict],
                             real_support: Optional[Dict[str, np.ndarray]] = None):
    """Normalized support vs. normalized rank for all taxonomic levels."""
    palette = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
               COLORS['quaternary'], COLORS['quinary'], COLORS['senary']]

    for i, (level_name, metrics) in enumerate(level_results.items()):
        if real_support and level_name in real_support:
            support = real_support[level_name]
            has = support > 0
        else:
            support = metrics['support']
            has = support > 0
        if not has.any():
            continue
        sup = np.sort(support[has])[::-1]
        n = len(sup)
        rank_norm = np.arange(1, n + 1) / n
        sup_norm = sup / sup.max()
        ax.plot(rank_norm, sup_norm,
                color=palette[i % len(palette)], linewidth=1.5,
                label=f'{level_name} (n={n})')

    ax.set_xlabel('Normalized rank')
    ax.set_ylabel('Normalized support (fraction of max)')
    ax.set_title('Host category support')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=FONT_SIZES['legend_small'], loc='upper right')
    ax.grid(alpha=0.3)


def plot_top_genera(ax, level_results: Dict[str, Dict],
                    real_support: Optional[Dict[str, np.ndarray]] = None,
                    top_n: int = 20):
    """Horizontal bar chart of the top-N genera by support."""
    genus = level_results.get('Genus', {})
    if not genus:
        ax.text(0.5, 0.5, 'No genus data', transform=ax.transAxes,
                ha='center', va='center', color=COLORS['text_light'])
        return

    if real_support and 'Genus' in real_support:
        support = real_support['Genus']
    else:
        support = genus['support'] * 5.0
    taxa = genus['taxon_names']

    # Sort descending, take top N
    order = np.argsort(support)[::-1][:top_n]
    top_sup = support[order]
    # Extract short genus name (last element of lineage)
    top_names = [taxa[i].split(';')[-1] if ';' in taxa[i] else taxa[i]
                 for i in order]

    y = np.arange(len(top_names))
    ax.barh(y, top_sup, color=COLORS['primary'], edgecolor='white',
            linewidth=0.5, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(top_names, fontsize=FONT_SIZES['legend_small'])
    ax.invert_yaxis()
    ax.set_xlabel('Support (genomes)')
    ax.set_title(f'Top {top_n} genera by support')

    # Value labels
    for yi, val in zip(y, top_sup):
        ax.text(val + top_sup.max() * 0.01, yi, f'{val:.0f}',
                va='center', fontsize=FONT_SIZES['tick_tiny'],
                color=COLORS['text_light'])

    ax.grid(alpha=0.3, axis='x')


def plot_unique_proteins_vs_support(ax, full_metadata_df: pd.DataFrame,
                                    hosts: List[str],
                                    level_results: Dict[str, Dict],
                                    real_support: Optional[Dict[str, np.ndarray]] = None):
    """Scatter median unique proteins per genus vs. class support."""
    genus = level_results.get('Genus', {})
    if not genus:
        ax.text(0.5, 0.5, 'No genus data', transform=ax.transAxes,
                ha='center', va='center', color=COLORS['text_light'])
        return

    if real_support and 'Genus' in real_support:
        support = real_support['Genus']
    else:
        support = genus['support'] * 5.0
    taxa = genus['taxon_names']
    taxa_set = {t: i for i, t in enumerate(taxa)}

    # Parse metadata: for each phage, find its host genera and n_unique_proteins
    genus_proteins = defaultdict(list)  # genus_lineage -> [n_unique_proteins]

    col = 'host_genus_lineage'
    if col not in full_metadata_df.columns:
        ax.text(0.5, 0.5, f'Column {col!r} not found in metadata',
                transform=ax.transAxes, ha='center', va='center',
                color=COLORS['text_light'])
        return

    for _, row in full_metadata_df.iterrows():
        try:
            n_uniq = int(row['n_unique_proteins'])
        except (ValueError, KeyError):
            continue
        lineages = str(row[col]).split('|')
        for lin in lineages:
            if lin in taxa_set:
                genus_proteins[lin].append(n_uniq)

    # Compute median per genus
    sup_vals, prot_vals = [], []
    for taxon, prot_list in genus_proteins.items():
        idx = taxa_set[taxon]
        sup_vals.append(support[idx])
        prot_vals.append(np.median(prot_list))

    sup_vals = np.array(sup_vals)
    prot_vals = np.array(prot_vals)

    ax.scatter(sup_vals, prot_vals, s=14, alpha=0.5,
               color=COLORS['primary'], edgecolors='none', zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel('Host category support (genomes)')
    ax.set_ylabel('Median unique proteins per phage')
    ax.set_title('Unique proteins')
    ax.grid(alpha=0.3)


def make_dataset_figure(level_results: Dict[str, Dict],
                        full_metadata_df: Optional[pd.DataFrame],
                        hosts: List[str],
                        real_support: Optional[Dict[str, np.ndarray]] = None,
                        out_path: str = 'dataset.png', dpi: int = 200):
    """Support distributions, top genera, and unique proteins vs support."""
    setup_style()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_support_rank_curves(ax1, level_results, real_support)
    plot_top_genera(ax2, level_results, real_support, top_n=20)

    if full_metadata_df is not None and len(full_metadata_df) > 0:
        plot_unique_proteins_vs_support(
            ax3, full_metadata_df, hosts, level_results, real_support)
    else:
        ax3.text(0.5, 0.5, 'No metadata available',
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=FONT_SIZES['fallback_msg'],
                 color=COLORS['text_light'])
        ax3.set_axis_off()

    _add_panel_letters([ax1, ax2, ax3], 'ABC', x=-0.08, y=1.06)

    _suptitle(fig, 'Dataset overview', y=1.01)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 1 - Training
# ---------------------------------------------------------------------------

def make_training_figure(history: Optional[pd.DataFrame],
                         logits: torch.Tensor, labels: torch.Tensor,
                         temperature: float, threshold: float,
                         out_path: str, dpi: int = 200):
    """Loss curves, validation F1, and calibration diagram."""
    setup_style()

    has_history = history is not None and len(history) > 0
    ncols = 3 if has_history else 1

    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    col = 0
    if has_history:
        plot_loss_curves(axes[0], history)
        plot_f1_curves(axes[1], history)
        col = 2

    plot_calibration(axes[col], logits, labels, temperature)

    _add_panel_letters(axes, 'ABC', x=-0.08, y=1.06)

    _suptitle(fig,
              f'Training summary  (T = {temperature:.3f}, '
              f'threshold = {threshold:.3f})',
              y=1.01)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 2 - Evaluation
# ---------------------------------------------------------------------------

def make_evaluation_figure(level_results: Dict[str, Dict],
                           metadata_df: Optional[pd.DataFrame],
                           logits: torch.Tensor, labels: torch.Tensor,
                           temperature: float, threshold: float,
                           real_support: Optional[Dict[str, np.ndarray]] = None,
                           out_path: str = 'evaluation.png', dpi: int = 200):
    """Taxonomic metrics, F1 distributions, support vs F1, distance panel."""
    setup_style()

    has_distance = (metadata_df is not None and len(metadata_df) > 0)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)
    ax_tax  = fig.add_subplot(gs[0, 0])
    ax_box  = fig.add_subplot(gs[0, 1])
    ax_sup  = fig.add_subplot(gs[1, 0])
    ax_ani  = fig.add_subplot(gs[1, 1])

    plot_taxonomic_metrics(ax_tax, level_results)
    plot_f1_distribution(ax_box, level_results)
    plot_support_vs_f1(ax_sup, level_results, real_support)

    if has_distance:
        has_prediction, is_correct = _compute_any_pred_correctness(
            logits, labels, temperature, threshold)
        max_ani = np.array(metadata_df['max_pt_train_tani'], dtype=float)
        _plot_distance_bar(
            ax_ani, max_ani, has_prediction, is_correct,
            xlabel='Max. tANI to PT training (in %)',
            title='Performance vs. max tANI',
            pcut=threshold)
    else:
        ax_ani.text(0.5, 0.5, 'No distance metadata available',
                    transform=ax_ani.transAxes, ha='center', va='center',
                    fontsize=FONT_SIZES['fallback_msg'],
                    color=COLORS['text_light'])
        ax_ani.set_axis_off()

    genus = level_results.get('Genus', {})
    title = (f'Evaluation summary  \u2014  '
             f'Genus: micro-F1={genus.get("micro_f1", 0):.3f}  '
             f'macro-F1={genus.get("macro_f1", 0):.3f}  '
             f'({genus.get("n_taxa", 0)} host categories)')
    _suptitle(fig, title, y=0.98)

    _add_panel_letters(fig.axes, 'ABCD', x=-0.06, y=1.06)

    _save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Per-class TSV export
# ---------------------------------------------------------------------------

def export_per_class_results(level_results: Dict[str, Dict], out_path: str):
    """Export per-class metrics at each taxonomic level to a TSV."""
    rows = []
    for level_name, metrics in level_results.items():
        taxa = metrics['taxon_names']
        per_f1 = metrics['per_f1']
        per_p = metrics['per_p']
        per_r = metrics['per_r']
        support = metrics['support']
        for i, taxon in enumerate(taxa):
            rows.append({
                'level': level_name,
                'taxon': taxon,
                'f1': f'{per_f1[i]:.5f}',
                'precision': f'{per_p[i]:.5f}',
                'recall': f'{per_r[i]:.5f}',
                'support': int(support[i]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep='\t', index=False)
    logger.info(f"Per-class results: {out_path}")


# ---------------------------------------------------------------------------
# Figure — Precision-Recall curves
# ---------------------------------------------------------------------------

RANK_COLORS = {
    'Phylum': COLORS['primary'],
    'Class':  COLORS['tertiary'],
    'Order':  COLORS['quaternary'],
    'Family': COLORS['quinary'],
    'Genus':  COLORS['secondary'],
}


def _compute_pr_curve(probs, labels, n_thresholds=200):
    """Compute micro- and macro-averaged PR curves by threshold sweep.

    Returns (thresholds, micro_p, micro_r, macro_p, macro_r) arrays.
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    micro_p = np.zeros(n_thresholds)
    micro_r = np.zeros(n_thresholds)
    macro_p = np.zeros(n_thresholds)
    macro_r = np.zeros(n_thresholds)

    # Support mask: classes with at least one positive
    support = labels.sum(0)
    has_support = support > 0

    for ti, t in enumerate(thresholds):
        preds = (probs >= t).float()
        tp = (preds * labels).sum(0)
        fp = (preds * (1 - labels)).sum(0)
        fn = ((1 - preds) * labels).sum(0)

        # Micro
        tp_sum = tp.sum().item()
        fp_sum = fp.sum().item()
        fn_sum = fn.sum().item()
        micro_p[ti] = tp_sum / max(tp_sum + fp_sum, 1e-12)
        micro_r[ti] = tp_sum / max(tp_sum + fn_sum, 1e-12)

        # Macro (only classes with support)
        per_p = tp / (tp + fp).clamp(min=1e-12)
        per_r = tp / (tp + fn).clamp(min=1e-12)
        # Set precision to 1 when no predictions (tp+fp=0) and no positives
        no_pred = (tp + fp) == 0
        per_p[no_pred & ~has_support] = 0
        per_p[no_pred & has_support] = 1.0  # conservative: if we predict nothing, precision=1

        if has_support.any():
            macro_p[ti] = per_p[has_support].mean().item()
            macro_r[ti] = per_r[has_support].mean().item()

    return thresholds, micro_p, micro_r, macro_p, macro_r


def _compute_auprc(precision, recall):
    """Compute area under the PR curve (trapezoidal)."""
    # Sort by recall ascending
    order = np.argsort(recall)
    r_sorted = recall[order]
    p_sorted = precision[order]
    return np.trapezoid(p_sorted, r_sorted)


def _plot_pr_panel(ax, probs_per_level, labels_per_level, level_names,
                   title, temperature, averaging='micro', n_thresholds=200):
    """Plot PR curves for all taxonomic levels on one axis."""
    for level_name in level_names:
        if level_name not in probs_per_level:
            continue
        probs = probs_per_level[level_name]
        labels = labels_per_level[level_name]
        _, micro_p, micro_r, macro_p, macro_r = _compute_pr_curve(
            probs, labels, n_thresholds)

        if averaging == 'micro':
            p, r = micro_p, micro_r
        else:
            p, r = macro_p, macro_r

        auprc = _compute_auprc(p, r)
        color = RANK_COLORS.get(level_name, '#999')
        ax.plot(r, p, color=color, linewidth=1.5, alpha=0.85,
                label=f'{level_name} (AUPRC={auprc:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.legend(fontsize=FONT_SIZES['legend_small'], loc='lower left',
              frameon=True, framealpha=0.9, edgecolor=COLORS['grid'])
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')


def make_pr_figure(logits: torch.Tensor, labels: torch.Tensor,
                   hosts: List[str], temperature: float,
                   blocked_classes: Optional[List[int]] = None,
                   out_path: str = 'precision_recall.png',
                   dpi: int = 200):
    """Precision-recall curves at each taxonomic level.

    Three panels:
      A: Micro-averaged PR curves per rank
      B: Macro-averaged PR curves per rank (all classes)
      C: Macro-averaged PR curves per rank (blocked classes excluded)
    """
    setup_style()

    probs = torch.sigmoid(logits / temperature)
    max_depth, lineages = parse_lineages(hosts)

    # Aggregate to each level
    probs_per_level = {}
    labels_per_level = {}
    level_names_present = []
    for level in range(max_depth):
        name = LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f'Level_{level}'
        if level == max_depth - 1:
            probs_per_level[name] = probs
            labels_per_level[name] = labels
        else:
            agg_p, agg_l, _ = aggregate_to_level(probs, labels, lineages, level)
            probs_per_level[name] = agg_p
            labels_per_level[name] = agg_l
        level_names_present.append(name)

    # Blocked-class version: zero out blocked columns at genus level
    has_blocked = blocked_classes and len(blocked_classes) > 0
    probs_blocked = {}
    labels_blocked = {}
    if has_blocked:
        keep_mask = torch.ones(probs.shape[1], dtype=torch.bool)
        keep_mask[blocked_classes] = False
        probs_kept = probs[:, keep_mask]
        labels_kept = labels[:, keep_mask]
        hosts_kept = [h for i, h in enumerate(hosts) if keep_mask[i]]

        _, lineages_kept = parse_lineages(hosts_kept)
        max_depth_kept = max(len(lin) for lin in lineages_kept)

        for level in range(max_depth_kept):
            name = LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f'Level_{level}'
            if level == max_depth_kept - 1:
                probs_blocked[name] = probs_kept
                labels_blocked[name] = labels_kept
            else:
                agg_p, agg_l, _ = aggregate_to_level(
                    probs_kept, labels_kept, lineages_kept, level)
                probs_blocked[name] = agg_p
                labels_blocked[name] = agg_l

    # --- Figure: 1×3 ---
    n_panels = 3 if has_blocked else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6),
                             squeeze=False)
    axes = axes[0]

    _plot_pr_panel(axes[0], probs_per_level, labels_per_level,
                   level_names_present, 'Micro-averaged precision–recall',
                   temperature, averaging='micro')

    _plot_pr_panel(axes[1], probs_per_level, labels_per_level,
                   level_names_present, 'Macro-averaged precision–recall',
                   temperature, averaging='macro')

    if has_blocked:
        n_blocked = len(blocked_classes)
        n_kept = int(keep_mask.sum())
        _plot_pr_panel(axes[2], probs_blocked, labels_blocked,
                       level_names_present,
                       f'Macro-averaged PR\n({n_blocked} blocked classes removed, '
                       f'{n_kept} kept)',
                       temperature, averaging='macro')

    letters = 'ABC'[:n_panels]
    _add_panel_letters(axes, letters, x=-0.08, y=1.06)

    fig.tight_layout()
    _save_figure(fig, out_path, dpi=dpi)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PhageTransformer model. '
                    'Produces figures in <model_dir>/plots/',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory (contains calibration.json and checkpoints/)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Override dataset directory')
    parser.add_argument('--testset_dir', type=str, default=None,
                        help='Override testset directory')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed score threshold (overrides --fdr)')
    parser.add_argument('--fdr', type=float, default=0.1,
                        help='FDR level for threshold from calibration')
    parser.add_argument('--max_patches', type=int, default=512)
    parser.add_argument('--eval_stride', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--mds', action='store_true',
                        help='Include MDS panels in embedding figure')
    parser.add_argument('--presentation', action='store_true',
                        help='Increase font sizes for presentations and '
                             'remove panel letters')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- presentation mode -----------------------------------------------
    if args.presentation:
        enable_presentation_mode()
        logger.info("Presentation mode enabled")

    # ---- output directory -------------------------------------------------
    plot_dir = os.path.join(args.model_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # ---- load model and calibration --------------------------------------
    model, calib = load_model_and_calibration(args.model_dir, args.checkpoint, device)
    temperature = calib['temperature']
    hosts = np.array(calib['hosts'])
    patch_nt_len = calib['model_config']['patch_nt_len']
    blocked_classes = calib.get('blocked_classes', [])

    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Threshold (fixed): {threshold:.4f}")
    else:
        fdr_key = f"fdr_{int(args.fdr * 100):02d}"
        fdr_thresholds = calib.get('fdr_thresholds', {})
        if fdr_key in fdr_thresholds:
            threshold = fdr_thresholds[fdr_key]
            logger.info(f"Threshold (FDR {args.fdr*100:.0f}%): {threshold:.4f}")
        else:
            threshold = calib.get('threshold', 0.5)
            logger.warning(f"FDR key '{fdr_key}' not in calibration, "
                          f"using default threshold {threshold:.4f}")

    logger.info(f"Threshold: {threshold:.4f}  Temperature: {temperature:.4f}")

    # ---- load data -------------------------------------------------------
    here = os.path.dirname(os.path.realpath(__file__))
    ds_path = args.dataset_dir
    ts_path = args.testset_dir

    logger.info("Loading data ...")
    _, _, val_seqs, val_labels, hosts_from_data = load_phage_host_merged(ds_path)
    test_seqs, test_labels = load_phage_host_test(ts_path, hosts_from_data)
    logger.info(f"  val={len(val_seqs)}  test={len(test_seqs)}  "
                f"classes={len(hosts)}")

    # ---- metadata for distance panels (optional) -------------------------
    metadata_path = os.path.join(ds_path, 'phages_hosts_tani.csv')
    metadata_df = None
    full_metadata_df = None
    if os.path.exists(metadata_path):
        full_metadata_df = pd.read_csv(metadata_path, sep=',', dtype=str)
        metadata_df = full_metadata_df[
            full_metadata_df['in_testset'] == '1'
        ].reset_index(drop=True)
        logger.info(f"  metadata loaded: {len(full_metadata_df)} total, "
                    f"{len(metadata_df)} testset samples")
    else:
        # Try phages_hosts.csv as fallback for dataset figure
        alt_path = os.path.join(ds_path, 'phages_hosts.csv')
        if os.path.exists(alt_path):
            full_metadata_df = pd.read_csv(alt_path, sep=',', dtype=str)
            logger.info(f"  metadata (phages_hosts.csv): "
                        f"{len(full_metadata_df)} samples")
        else:
            logger.info(f"  No metadata CSV found, skipping distance panels")

    # ---- build val loader ------------------------------------------------
    tokenizer = CodonTokenizer()
    eval_stride = args.eval_stride or patch_nt_len // 2
    val_ds = PatchSequenceDataset(
        val_seqs, val_labels, tokenizer,
        patch_nt_len=patch_nt_len, max_patches=args.max_patches,
        is_train=False, eval_stride=eval_stride,
    )
    ldr_kw = dict(num_workers=args.num_workers, pin_memory=True,
                  persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=sequence_collate_fn, **ldr_kw)

    # ---- inference -------------------------------------------------------
    logger.info("Running inference on validation set ...")
    logits, labels = collect_logits(model, val_loader, device)
    logger.info(f"  logits: {logits.shape}  labels: {labels.shape}")

    # Strip bacterial_fragment class if present — the model outputs an extra
    # column for it, but val/test labels are phage-only so dimensions mismatch.
    if len(hosts) > labels.shape[1] and hosts[-1] == 'bacterial_fragment':
        n_core = len(hosts) - 1
        logits = logits[:, :n_core]
        hosts = hosts[:n_core]
        logger.info(f"  Stripped bacterial_fragment → "
                    f"logits: {logits.shape}  hosts: {len(hosts)}")

    # ---- evaluate at every taxonomic level -------------------------------
    logger.info("Computing metrics at each taxonomic level ...")
    level_results = evaluate_all_levels(
        logits, labels, list(hosts), temperature, threshold)

    # ---- load training history -------------------------------------------
    history = load_training_history(args.model_dir)

    # ---- compute real class support from full metadata -------------------
    real_support = compute_real_support(
        full_metadata_df, level_results, list(hosts))
    if real_support:
        n_genus = len(real_support.get('Genus', []))
        logger.info(f"  Real support computed for {len(real_support)} levels "
                    f"({n_genus} genera)")

    # ---- Figure 1: Training ----------------------------------------------
    logger.info("Generating training figure ...")
    make_training_figure(
        history, logits, labels, temperature, threshold,
        out_path=_output_path(os.path.join(plot_dir, 'training.png')),
        dpi=args.dpi)

    # ---- Figure 2: Evaluation --------------------------------------------
    logger.info("Generating evaluation figure ...")
    make_evaluation_figure(
        level_results, metadata_df, logits, labels, temperature, threshold,
        real_support=real_support,
        out_path=_output_path(os.path.join(plot_dir, 'evaluation.png')),
        dpi=args.dpi)

    # ---- Figure 3: Codon embedding PCA -----------------------------------
    logger.info("Generating embedding PCA figure ...")
    make_embedding_figure(
        model,
        out_path=_output_path(os.path.join(plot_dir, 'embeddings.png')),
        dpi=args.dpi, show_mds=args.mds)

    # ---- Figure 3b: Codon property × PC correlation (supplementary) ------
    logger.info("Generating codon property–PC correlation figure ...")
    make_codon_property_correlation(
        model,
        out_path=_output_path(os.path.join(plot_dir,
                                            'codon_property_pc_correlation.png')),
        tsv_path=os.path.join(plot_dir, 'codon_property_pc_correlations.tsv'),
        n_pcs=30, dpi=args.dpi)

    # ---- Figure 4: Dataset overview --------------------------------------
    logger.info("Generating dataset overview figure ...")
    make_dataset_figure(
        level_results, full_metadata_df, list(hosts),
        real_support=real_support,
        out_path=_output_path(os.path.join(plot_dir, 'dataset.png')),
        dpi=args.dpi)

    # ---- Per-class TSV ---------------------------------------------------
    export_per_class_results(
        level_results,
        out_path=os.path.join(plot_dir, 'per_class_metrics.tsv'))

    # ---- Figure 5: Precision–Recall curves --------------------------------
    logger.info("Generating precision–recall figure ...")
    make_pr_figure(
        logits, labels, list(hosts), temperature,
        blocked_classes=blocked_classes,
        out_path=_output_path(os.path.join(plot_dir, 'precision_recall.png')),
        dpi=args.dpi)

    logger.info(f"All outputs saved to {plot_dir}/")
    logger.info("Done.")


if __name__ == '__main__':
    main()
