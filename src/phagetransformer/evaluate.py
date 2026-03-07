#!/usr/bin/env python3
"""Evaluate a trained PhageTransformer model and produce figures.

Usage:
    python evaluate.py --model_dir ./models/my_training_run
"""

import argparse
import gzip
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.utils.data import DataLoader

from .model import CodonTokenizer
from .predict import load_model_and_calibration
from .dataset import (
    load_phage_host_merged, load_phage_host_test,
    PatchSequenceDataset, sequence_collate_fn,
)
from .train import _unpack_sequence_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color theme
# ---------------------------------------------------------------------------

COLORS = {
    'primary':    '#2C5F8A',
    'secondary':  '#D4563E',
    'tertiary':   '#3A8A6E',
    'quaternary': '#E8983E',
    'quinary':    '#7B5EA7',
    'senary':     '#C47BA0',
    #'light_bg':   '#F5F5F2',
    'light_bg':   '#FFFFFF',
    'grid':       '#D0CFC8',
    'text':       '#2D2D2D',
    'text_light': '#6B6B6B',
}

METRIC_COLORS = {
    'micro_f1': COLORS['primary'],
    'macro_f1': COLORS['secondary'],
    'micro_p':  COLORS['tertiary'],
    'micro_r':  COLORS['quaternary'],
    'macro_p':  COLORS['quinary'],
    'macro_r':  COLORS['senary'],
}

LEVEL_NAMES = ['Phylum', 'Class', 'Order', 'Family', 'Genus']

# ---------------------------------------------------------------------------
# Central font-size definitions
# ---------------------------------------------------------------------------

FONT_SIZES = {
    'base':           11,    # mpl font.size
    'suptitle':       12,    # figure suptitle
    'title':          11,    # axes title
    'label':          11,    # axes label
    'tick':           10,    # standard tick labels
    'tick_small':      9,    # smaller tick labels (grouped bars, box plots)
    'tick_tiny':       7.5,  # very small tick labels (distance bins)
    'legend':         10,    # standard legend
    'legend_small':    8,    # compact legends inside dense panels
    'annotation':      8,    # in-plot text (ECE, silhouette, etc.)
    'bar_value':       6,    # tiny value labels on bars
    'bar_label':       7,    # n= labels below grouped bars
    'panel_letter':   13,    # A, B, C, D panel labels
    'codon':           5.5,  # codon annotations on embedding plots
    'colorbar':        9,    # colorbar label
    'secondary_axis':  7,    # twin-axis labels and ticks
    'fallback_msg':   10,    # "no data available" placeholder text
}

PRESENTATION_MODE = False


def setup_style():
    """Configure matplotlib for publication-quality output."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': FONT_SIZES['base'],
        'axes.titlesize': FONT_SIZES['title'],
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_SIZES['label'],
        'axes.labelcolor': COLORS['text'],
        'axes.edgecolor': COLORS['grid'],
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'xtick.color': COLORS['text_light'],
        'ytick.color': COLORS['text_light'],
        'legend.fontsize': FONT_SIZES['legend'],
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': COLORS['grid'],
        'figure.facecolor': COLORS['light_bg'],
        'savefig.facecolor': COLORS['light_bg'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })


def _add_panel_letters(axes, letters='ABCDEFGH', x=-0.06, y=1.06):
    """Add panel letters (A, B, C, ...) unless in presentation mode."""
    if PRESENTATION_MODE:
        return
    for ax, letter in zip(axes, letters):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=FONT_SIZES['panel_letter'], fontweight='bold',
                color=COLORS['text'], va='top', ha='right')


def _output_path(base_path: str) -> str:
    """Adjust output filename for presentation mode."""
    if not PRESENTATION_MODE:
        return base_path
    root, ext = os.path.splitext(base_path)
    return f'{root}_presentation{ext}'


def _suptitle(fig, title: str, **kwargs):
    """Add figure suptitle unless in presentation mode."""
    if PRESENTATION_MODE:
        return
    kwargs.setdefault('fontsize', FONT_SIZES['suptitle'])
    kwargs.setdefault('fontweight', 'bold')
    kwargs.setdefault('color', COLORS['text'])
    fig.suptitle(title, **kwargs)


# ---------------------------------------------------------------------------
# Model + data loading
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_logits(model, loader, device):
    """Run model on loader, return (logits, labels) on CPU."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        logits, labels = _unpack_sequence_batch(model, batch, device)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def load_test_with_ids(ts_path: str, hosts: np.ndarray
                       ) -> Tuple[List[str], List[str], List[str],
                                  np.ndarray]:
    """Load the external test set preserving sequence IDs and dataset labels.

    Returns
    -------
    seq_ids  : list of FASTA record IDs
    seqs     : list of nucleotide sequences
    datasets : list of dataset labels (cow_fecal, human_gut, …)
    labels   : (N, C) multi-hot label matrix
    """
    fasta_path = os.path.join(ts_path, 'combined.fna.gz')
    csv_path = os.path.join(ts_path, 'combined_lineage.csv')

    # Read FASTA with IDs
    seq_ids, seqs = [], []
    with gzip.open(fasta_path, 'rt') as fh:
        for rec in SeqIO.parse(fh, 'fasta'):
            seq_ids.append(rec.id)
            seqs.append(str(rec.seq))

    # Read lineage metadata
    df = pd.read_csv(csv_path, delimiter=',', dtype=str)

    # Build label matrix
    genus_index = {x: i for i, x in enumerate(hosts)}
    labels = np.zeros((len(df), len(genus_index)), dtype=np.float32)
    for i, hs in enumerate(df['host_genus_lineage']):
        for h in hs.split('|'):
            h_trunc = ';'.join(h.split(';')[1:])  # drop domain prefix
            if h_trunc in genus_index:
                labels[i, genus_index[h_trunc]] = 1

    datasets = df['dataset'].tolist()
    return seq_ids, seqs, datasets, labels


def predict_test_for_comparison(
        model, seqs: List[str], seq_ids: List[str],
        hosts: np.ndarray, temperature: float,
        tokenizer, device: torch.device,
        patch_nt_len: int = 4096, max_patches: int = 512,
        eval_stride: Optional[int] = None,
        batch_size: int = 8, num_workers: int = 4,
) -> pd.DataFrame:
    """Run model inference on test sequences and return predictions DataFrame.

    Returns DataFrame with columns: sequence_id, genus, score, is_bacterial.
    ``is_bacterial`` is True when the bacterial_fragment class probability
    exceeds the best genus probability (model thinks the input is bacterial
    DNA, not a phage).  One row per sequence (top-1 genus prediction).
    """
    # Build dummy labels (not needed for prediction, but DataLoader expects them)
    dummy_labels = np.zeros((len(seqs), len(hosts)), dtype=np.float32)

    stride = eval_stride or patch_nt_len // 2
    ds = PatchSequenceDataset(
        seqs, dummy_labels, tokenizer,
        patch_nt_len=patch_nt_len, max_patches=max_patches,
        is_train=False, eval_stride=stride,
    )
    ldr_kw = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=sequence_collate_fn, **ldr_kw)

    logger.info(f"  Running PT inference on {len(seqs)} test sequences ...")
    logits, _ = collect_logits(model, loader, device)
    all_probs = torch.sigmoid(logits / temperature)

    # Check if model has a bacterial_fragment class (last column beyond hosts)
    n_core = len(hosts)
    has_bact_class = all_probs.shape[1] > n_core

    # Core genus probabilities (excluding bacterial_fragment)
    core_probs = all_probs[:, :n_core]
    best_scores, best_indices = core_probs.max(dim=1)

    rows = []
    for i in range(len(seqs)):
        is_bact = False
        if has_bact_class:
            is_bact = all_probs[i, n_core].item() > best_scores[i].item()
        rows.append({
            'sequence_id': seq_ids[i],
            'genus': hosts[best_indices[i].item()],
            'score': best_scores[i].item(),
            'is_bacterial': is_bact,
        })

    df = pd.DataFrame(rows)
    n_bact = df['is_bacterial'].sum()
    if n_bact:
        logger.info(f"  {n_bact} / {len(df)} sequences flagged as "
                    f"bacterial_fragment")
    return df


# ---------------------------------------------------------------------------
# Taxonomic aggregation
# ---------------------------------------------------------------------------

def parse_lineages(hosts: List[str]) -> Tuple[int, List[List[str]]]:
    """Parse semicolon-delimited lineage strings."""
    lineages = []
    max_depth = 0
    for h in hosts:
        parts = h.split(';')
        lineages.append(parts)
        max_depth = max(max_depth, len(parts))
    return max_depth, lineages


def aggregate_to_level(probs: torch.Tensor, labels: torch.Tensor,
                       lineages: List[List[str]], level: int
                       ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Aggregate genus-level predictions/labels to a higher taxonomic level."""
    taxon_to_indices = defaultdict(list)
    for idx, lin in enumerate(lineages):
        if level < len(lin):
            key = ';'.join(lin[:level + 1])
        else:
            key = ';'.join(lin)
        taxon_to_indices[key].append(idx)

    taxon_names = sorted(taxon_to_indices.keys())
    n_taxa = len(taxon_names)
    N = probs.shape[0]

    agg_probs = torch.zeros(N, n_taxa)
    agg_labels = torch.zeros(N, n_taxa)

    for ti, taxon in enumerate(taxon_names):
        indices = taxon_to_indices[taxon]
        agg_probs[:, ti] = probs[:, indices].max(dim=1).values
        agg_labels[:, ti] = labels[:, indices].max(dim=1).values

    return agg_probs, agg_labels, taxon_names


def compute_level_metrics(probs: torch.Tensor, labels: torch.Tensor,
                          threshold: float) -> Dict[str, float]:
    """Compute micro and macro metrics from probabilities and binary labels."""
    preds = (probs >= threshold).float()
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    fn = ((1 - preds) * labels).sum(0)

    tp_s, fp_s, fn_s = tp.sum(), fp.sum(), fn.sum()
    micro_p = (tp_s / (tp_s + fp_s).clamp(min=1)).item()
    micro_r = (tp_s / (tp_s + fn_s).clamp(min=1)).item()
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)

    has = (tp + fn) > 0
    per_p = tp / (tp + fp).clamp(min=1)
    per_r = tp / (tp + fn).clamp(min=1)
    per_f1 = 2 * per_p * per_r / (per_p + per_r).clamp(min=1e-8)

    macro_p = per_p[has].mean().item() if has.any() else 0.0
    macro_r = per_r[has].mean().item() if has.any() else 0.0
    macro_f1 = per_f1[has].mean().item() if has.any() else 0.0

    return {
        'micro_f1': micro_f1, 'micro_p': micro_p, 'micro_r': micro_r,
        'macro_f1': macro_f1, 'macro_p': macro_p, 'macro_r': macro_r,
        'n_taxa': int(has.sum()),
        'per_f1': per_f1.numpy(), 'per_p': per_p.numpy(), 'per_r': per_r.numpy(),
        'support': (tp + fn).numpy(),
    }


def evaluate_all_levels(logits: torch.Tensor, labels: torch.Tensor,
                        hosts: List[str], temperature: float,
                        threshold: float) -> Dict[str, Dict]:
    """Compute metrics at every taxonomic level."""
    probs = torch.sigmoid(logits / temperature)
    max_depth, lineages = parse_lineages(hosts)

    results = {}
    for level in range(max_depth):
        name = LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f'Level_{level}'
        if level == max_depth - 1:
            metrics = compute_level_metrics(probs, labels, threshold)
            metrics['taxon_names'] = list(hosts)
        else:
            agg_p, agg_l, taxa = aggregate_to_level(probs, labels, lineages, level)
            metrics = compute_level_metrics(agg_p, agg_l, threshold)
            metrics['taxon_names'] = taxa
        results[name] = metrics
        logger.info(f"  {name:8s}: n_taxa={metrics['n_taxa']:4d}  "
                    f"micro-F1={metrics['micro_f1']:.4f}  "
                    f"macro-F1={metrics['macro_f1']:.4f}  "
                    f"P={metrics['micro_p']:.4f}  R={metrics['micro_r']:.4f}")

    return results


def compute_real_support(full_metadata_df: pd.DataFrame,
                         level_results: Dict[str, Dict],
                         hosts: List[str]) -> Dict[str, np.ndarray]:
    """Count actual per-class support from the full metadata CSV.

    Returns a dict mapping level name -> np.ndarray of the same length as
    the taxon list in level_results, with real sample counts.
    """
    col = 'host_genus_lineage'
    if full_metadata_df is None or col not in full_metadata_df.columns:
        return {}

    # Count per-genus from metadata
    genus_counts = defaultdict(int)
    for lineages_str in full_metadata_df[col].dropna():
        for lin in lineages_str.split('|'):
            genus_counts[lin] += 1

    # Build real support for genus level
    genus_res = level_results.get('Genus', {})
    if not genus_res:
        return {}

    taxa = genus_res['taxon_names']
    genus_support = np.array([genus_counts.get(t, 0) for t in taxa],
                             dtype=float)

    real_support = {'Genus': genus_support}

    # Aggregate to higher levels using the same taxon groupings
    _, lineages = parse_lineages(hosts)
    for level_name, metrics in level_results.items():
        if level_name == 'Genus':
            continue
        level_taxa = metrics['taxon_names']
        # Find which level index this is
        level_idx = LEVEL_NAMES.index(level_name) if level_name in LEVEL_NAMES \
            else None
        if level_idx is None:
            continue

        taxon_to_genera = defaultdict(set)
        for idx, lin in enumerate(lineages):
            if level_idx < len(lin):
                key = ';'.join(lin[:level_idx + 1])
            else:
                key = ';'.join(lin)
            taxon_to_genera[key].add(idx)

        level_sup = np.zeros(len(level_taxa))
        taxa_lookup = {t: i for i, t in enumerate(taxa)}
        for ti, taxon in enumerate(level_taxa):
            genus_indices = taxon_to_genera.get(taxon, set())
            for gi in genus_indices:
                if gi < len(taxa):
                    level_sup[ti] += genus_support[gi]
        real_support[level_name] = level_sup

    return real_support


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

def _compute_best_pred_correctness(raw_logits, labels, temperature, pcut):
    """Per-sample prediction flags for distance analysis."""
    logits = raw_logits / temperature
    probs = torch.sigmoid(logits)
    best_preds = probs.argmax(dim=1)
    best_scores = probs[range(len(best_preds)), best_preds]

    true_probs = [
        p[l] for p, l in zip(
            probs.detach().cpu().numpy(),
            np.array(labels.detach().cpu().numpy(), dtype=bool),
        )
    ]

    has_prediction = best_scores >= pcut
    is_best = torch.tensor([
        any(bs == ts for ts in tss)
        for bs, tss in zip(best_scores.detach().cpu().numpy(), true_probs)
    ])
    is_correct = has_prediction & is_best
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
    has_prediction, is_correct = _compute_best_pred_correctness(
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

    # Wobble (3rd position) classification: pyrimidine (U/C) vs purine (A/G)
    wobble_is_purine = np.array([c[2] in 'AG' for c in codons])  # True=purine

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

    def _plot_wobble(ax, coords):
        """Plot codons colored by wobble position: UC (pyrimidine) vs AG (purine)."""
        color_pur = COLORS['primary']    # AG — purine
        color_pyr = COLORS['secondary']  # UC — pyrimidine
        mask_pur = wobble_is_purine
        mask_pyr = ~wobble_is_purine
        ax.scatter(coords[mask_pyr, 0], coords[mask_pyr, 1],
                   c=color_pyr, s=50, alpha=0.85,
                   edgecolors='white', linewidths=0.4, zorder=3,
                   label='UC (pyrimidine)')
        ax.scatter(coords[mask_pur, 0], coords[mask_pur, 1],
                   c=color_pur, s=50, alpha=0.85,
                   edgecolors='white', linewidths=0.4, zorder=3,
                   label='AG (purine)')
        for i, codon in enumerate(codons):
            ax.annotate(codon, (coords[i, 0], coords[i, 1]),
                        fontsize=FONT_SIZES['codon'], ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        color=COLORS['text_light'])

    # ---- Compute property–PC correlations for summary panel ----
    from scipy.stats import spearmanr as _spearmanr

    n_summary_pcs = min(10, centered.shape[1])
    all_var = S ** 2 / (S ** 2).sum()
    pcs_all = centered @ Vt[:n_summary_pcs].T

    prop_matrix, prop_names = build_codon_property_matrix(codons)
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

    # Property group colors for the summary bars
    _prop_group_colors = {
        'Nucleotide': '#4E79A7', 'Dinucleotide': '#F28E2B',
        'Sequence': '#76B7B2', 'Amino acid': '#E15759',
        'Degeneracy': '#59A14F', 'Other': '#999999',
    }

    def _prop_group(name):
        if name.startswith('pos') or name.startswith('GC_pos') \
                or name.startswith('purine_pos'):
            return 'Nucleotide'
        if name.startswith('dinuc_'):
            return 'Dinucleotide'
        if name.startswith('wobble') or name in ('GC_content', 'total_hbonds',
                                                  'purine_purine_steps'):
            return 'Sequence'
        if name.startswith('aa_') or name in ('is_stop', 'is_start'):
            return 'Amino acid'
        if 'degenerate' in name or name == 'degeneracy':
            return 'Degeneracy'
        return 'Other'

    # --- Layout: 1×3 (or 2×3 with MDS) ---
    nrows = 2 if show_mds else 1
    fig = plt.figure(figsize=(22, 6.5 * nrows))
    gs = gridspec.GridSpec(nrows, 3, figure=fig, hspace=0.22,
                           wspace=0.38, left=0.06, right=0.97,
                           top=0.90, bottom=0.08 if show_mds else 0.10)

    # --- Panel A: summary bar chart ---
    ax_summary = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(n_summary_pcs)
    bar_colors = [_prop_group_colors.get(_prop_group(prop_names[idx]), '#999')
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
    seen_groups = sorted(set(_prop_group(prop_names[idx])
                             for idx in best_prop_idx))
    grp_patches = [_Patch(facecolor=_prop_group_colors[g], label=g)
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

    # --- Panel C: PCA — wobble ---
    ax_pca_wob = fig.add_subplot(gs[0, 2])
    _plot_wobble(ax_pca_wob, pc)
    ax_pca_wob.set_xlabel(f'PC1 ({var_explained[0]:.1%} var.)')
    ax_pca_wob.set_ylabel(f'PC2 ({var_explained[1]:.1%} var.)')
    ax_pca_wob.set_title('PCA — wobble nucleotide (3rd pos.)')
    ax_pca_wob.legend(fontsize=FONT_SIZES['legend_small'], loc='best',
                      frameon=True, framealpha=0.9, edgecolor=COLORS['grid'])

    all_axes = [ax_summary, ax_pca_aa, ax_pca_wob]

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

        ax_mds_wob = fig.add_subplot(gs[1, 2])
        _plot_wobble(ax_mds_wob, mds)
        ax_mds_wob.set_xlabel('MDS dim 1')
        ax_mds_wob.set_ylabel('MDS dim 2')
        ax_mds_wob.set_title('MDS (cosine) — wobble nucleotide')
        ax_mds_wob.legend(fontsize=FONT_SIZES['legend_small'], loc='best',
                          frameon=True, framealpha=0.9,
                          edgecolor=COLORS['grid'])

        all_axes += [ax_mds_aa, ax_mds_wob]

    # Panel labels
    _add_panel_letters(all_axes, 'ABCDE', x=-0.06, y=1.06)

    # Shared amino acid legend — right of panel B
    handles, labels = ax_pca_aa.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FONT_SIZES['legend_small'], ncol=1,
               loc='center left',
               bbox_to_anchor=(0.62, 0.5 if not show_mds else 0.75),
               frameon=True, framealpha=0.9, edgecolor=COLORS['grid'],
               handletextpad=0.4)

    embed_dim = codon_embs.shape[1]
    title = (f'Learned codon embeddings  ({embed_dim}-dim,  '
             f'AA silhouette = {sil:.3f} @ {best_k} PCs, p = {p_str}')
    if show_mds:
        title += f',  MDS cosine stress = {stress:.3f}'
    title += ')'
    _suptitle(fig, title, y=0.97)

    fig.savefig(out_path, dpi=dpi)
    logger.info(f"Figure saved: {out_path}")
    plt.close(fig)


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


def build_codon_property_matrix(codons: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build a (64, N_properties) matrix of biologically meaningful codon features.

    Returns (matrix, property_names).
    """
    props = {}  # name -> array of length 64

    # --- Nucleotide composition ---
    props['GC_content'] = np.array([_codon_gc(c) for c in codons])
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
                                    n_pcs: int = 50, dpi: int = 200):
    """Correlate codon properties with learned embedding PCs.

    Produces a heatmap of Spearman correlations (properties × PCs) and
    optionally exports a TSV of all correlations.
    """
    from scipy.stats import spearmanr
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

    # Build property matrix
    prop_matrix, prop_names = build_codon_property_matrix(codons)
    n_props = len(prop_names)

    # Compute Spearman correlations: (n_props, n_pcs)
    rho_matrix = np.zeros((n_props, n_pcs))
    pval_matrix = np.zeros((n_props, n_pcs))
    for i in range(n_props):
        if prop_matrix[:, i].std() < 1e-12:
            continue
        for j in range(n_pcs):
            rho, pval = spearmanr(prop_matrix[:, i], pcs[:, j])
            rho_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    # --- Export TSV ---
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
                })
        df = pd.DataFrame(rows)
        df.to_csv(tsv_path, sep='\t', index=False)
        logger.info(f"Codon property correlations: {tsv_path}")

    # --- Cluster properties by correlation profile for better heatmap ---
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    nonconstant = [i for i in range(n_props)
                   if prop_matrix[:, i].std() > 1e-12]
    rho_nc = rho_matrix[nonconstant]
    names_nc = [prop_names[i] for i in nonconstant]

    if len(nonconstant) > 2:
        row_dist = pdist(rho_nc, metric='correlation')
        row_dist = np.nan_to_num(row_dist, nan=1.0)
        row_link = linkage(row_dist, method='average')
        row_order = leaves_list(row_link)
    else:
        row_order = np.arange(len(nonconstant))

    rho_ordered = rho_nc[row_order]
    names_ordered = [names_nc[i] for i in row_order]

    # Group properties for annotation
    def _get_group(n):
        if n.startswith('pos') or n.startswith('GC_pos') \
                or n.startswith('purine_pos'):
            return 'Nucleotide'
        if n.startswith('dinuc_'):
            return 'Dinucleotide'
        if n.startswith('wobble') or n in ('GC_content', 'total_hbonds',
                                            'purine_purine_steps'):
            return 'Sequence'
        if n.startswith('aa_') or n in ('is_stop', 'is_start'):
            return 'Amino acid'
        if 'degenerate' in n or n == 'degeneracy':
            return 'Degeneracy'
        return 'Other'

    group_colors = {
        'Nucleotide': '#4E79A7', 'Dinucleotide': '#F28E2B',
        'Sequence': '#76B7B2', 'Amino acid': '#E15759',
        'Degeneracy': '#59A14F', 'Other': '#999999',
    }

    # --- Figure: top PCs heatmap + full heatmap ---
    n_top = min(20, n_pcs)
    fig = plt.figure(figsize=(18, max(12, len(names_ordered) * 0.28 + 3)))
    gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2.2],
                                wspace=0.35, left=0.18, right=0.96,
                                top=0.93, bottom=0.06)

    ax_top = fig.add_subplot(gs_main[0, 0])
    im_top = ax_top.imshow(rho_ordered[:, :n_top], aspect='auto',
                            cmap='RdBu_r', vmin=-1, vmax=1,
                            interpolation='nearest')
    ax_top.set_xticks(range(n_top))
    ax_top.set_xticklabels(
        [f'PC{j+1}\n({var_explained[j]:.1%})' for j in range(n_top)],
        fontsize=FONT_SIZES['bar_value'], rotation=90)
    ax_top.set_yticks(range(len(names_ordered)))
    ax_top.set_yticklabels(names_ordered, fontsize=FONT_SIZES['bar_label'])
    ax_top.set_title(f'Top {n_top} PCs', fontsize=FONT_SIZES['title'],
                     fontweight='bold')

    for yi, name in enumerate(names_ordered):
        grp = _get_group(name)
        ax_top.get_yticklabels()[yi].set_color(group_colors.get(grp, '#333'))

    for i in range(len(names_ordered)):
        for j in range(n_top):
            orig_i = nonconstant[row_order[i]]
            if pval_matrix[orig_i, j] < 0.001:
                ax_top.text(j, i, '***', ha='center', va='center',
                           fontsize=4, color='black'
                           if abs(rho_ordered[i, j]) < 0.6 else 'white')
            elif pval_matrix[orig_i, j] < 0.01:
                ax_top.text(j, i, '**', ha='center', va='center',
                           fontsize=4, color='black'
                           if abs(rho_ordered[i, j]) < 0.6 else 'white')
            elif pval_matrix[orig_i, j] < 0.05:
                ax_top.text(j, i, '*', ha='center', va='center',
                           fontsize=4, color='black'
                           if abs(rho_ordered[i, j]) < 0.6 else 'white')

    ax_all = fig.add_subplot(gs_main[0, 1])
    im_all = ax_all.imshow(rho_ordered, aspect='auto',
                            cmap='RdBu_r', vmin=-1, vmax=1,
                            interpolation='nearest')
    tick_step = max(1, n_pcs // 10)
    ax_all.set_xticks(range(0, n_pcs, tick_step))
    ax_all.set_xticklabels(
        [f'PC{j+1}' for j in range(0, n_pcs, tick_step)],
        fontsize=FONT_SIZES['bar_value'])
    ax_all.set_yticks([])
    ax_all.set_title(f'All {n_pcs} PCs', fontsize=FONT_SIZES['title'],
                     fontweight='bold')

    ax_var = ax_all.twiny()
    ax_var.bar(range(n_pcs), var_explained * 100, color=COLORS['primary'],
               alpha=0.5, width=0.8)
    ax_var.set_xlim(-0.5, n_pcs - 0.5)
    ax_var.set_ylabel('% var.', fontsize=FONT_SIZES['secondary_axis'])
    ax_var.tick_params(labelsize=FONT_SIZES['codon'])

    cbar_ax = fig.add_axes([0.97, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im_all, cax=cbar_ax)
    cb.set_label('Spearman ρ', fontsize=FONT_SIZES['colorbar'])

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=group_colors[g], label=g)
                      for g in ['Nucleotide', 'Dinucleotide', 'Sequence',
                                'Amino acid', 'Degeneracy']]
    fig.legend(handles=legend_patches, fontsize=FONT_SIZES['bar_label'],
               loc='lower left', bbox_to_anchor=(0.01, 0.01), ncol=5,
               frameon=True, framealpha=0.9, edgecolor=COLORS['grid'])

    _suptitle(fig,
        f'Codon properties vs. learned embedding PCs  '
        f'(Spearman ρ, * p<.05  ** p<.01  *** p<.001)',
        y=0.97)

    fig.savefig(out_path, dpi=dpi)
    logger.info(f"Figure saved: {out_path}")
    plt.close(fig)

    # --- Also export the property matrix itself ---
    if tsv_path:
        prop_tsv = tsv_path.replace('_correlations.tsv', '_matrix.tsv')
        prop_df = pd.DataFrame(prop_matrix, columns=prop_names,
                                index=codons)
        prop_df.index.name = 'codon'
        aa_col = [CODON_TO_AA[c] for c in codons]
        prop_df.insert(0, 'amino_acid', aa_col)
        prop_df.to_csv(prop_tsv, sep='\t')
        logger.info(f"Codon property matrix: {prop_tsv}")

    return rho_matrix, pval_matrix, prop_names, var_explained


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
    fig.savefig(out_path, dpi=dpi)
    logger.info(f"Figure saved: {out_path}")
    plt.close(fig)


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
    fig.savefig(out_path, dpi=dpi)
    logger.info(f"Figure saved: {out_path}")
    plt.close(fig)


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
        has_prediction, is_correct = _compute_best_pred_correctness(
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

    fig.savefig(out_path, dpi=dpi)
    logger.info(f"Figure saved: {out_path}")
    plt.close(fig)


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
# Figure 5 - Tool comparison
# ---------------------------------------------------------------------------

RANK_COLORS = {
    'Phylum': COLORS['primary'],     # blue
    'Class':  COLORS['tertiary'],    # green
    'Order':  COLORS['quaternary'],  # orange
    'Family': COLORS['quinary'],     # purple
    'Genus':  COLORS['secondary'],   # red
}

DATASET_LABELS = {
    'cow_fecal':    'Cow fecal',
    'human_gut':    'Human gut',
    'waste_water':  'Waste water',
    'refseq':       'RefSeq',
}


def _parse_lineage_target(lin: str) -> List[Optional[str]]:
    """Parse target host_genus_lineage: Domain;Phylum;Class;Order;Family;Genus."""
    if pd.isna(lin) or lin == '':
        return [None] * 5
    parts = lin.split(';')
    return [parts[i] if i < len(parts) and parts[i] else None
            for i in range(1, 6)]


def _parse_lineage_gtdb(lin: str) -> List[Optional[str]]:
    """Parse GTDB-style lineage: d__;p__;c__;o__;f__;g__ → strip prefixes."""
    if pd.isna(lin) or lin == '':
        return [None] * 5
    parts = lin.split(';')
    cleaned = [p.split('__', 1)[1] if '__' in p else p for p in parts]
    return [cleaned[i] if i < len(cleaned) and cleaned[i] else None
            for i in range(1, 6)]


def _parse_lineage_pt(lin: str) -> List[Optional[str]]:
    """Parse PT lineage: Phylum;Class;Order;Family;Genus (no domain prefix)."""
    if pd.isna(lin) or lin == '':
        return [None] * 5
    parts = lin.split(';')
    return [parts[i] if i < len(parts) and parts[i] else None
            for i in range(0, 5)]


def load_comparison_data(compare_dir: str, pt_predictions: pd.DataFrame,
                         pt_threshold: float
                         ) -> Tuple[pd.DataFrame, List[str]]:
    """Load external tool predictions and return a unified stats DataFrame.

    Expects these files in *compare_dir*:
      - target_lineages.csv          (Phage, dataset, host_genus_lineage)
      - iphop_predictions_gtdb226.csv
      - cherry_final_prediction_gtdb.csv

    PhageTransformer predictions are passed directly as *pt_predictions*
    (DataFrame with columns: sequence_id, genus, score) from live inference.

    Parameters
    ----------
    compare_dir : path to the eval_combined directory
    pt_predictions : DataFrame from predict_test_for_comparison()
    pt_threshold : PhageTransformer score cutoff (FDR 10% from calibration)

    Returns
    -------
    stats : DataFrame with columns dataset, rank, tool, total, has_pred,
            correct, ratio_correct, ratio_wrong
    datasets : ordered list of dataset names found
    """
    target = pd.read_csv(os.path.join(compare_dir, 'target_lineages.csv'))
    iphop  = pd.read_csv(os.path.join(compare_dir,
                                       'iphop_predictions_gtdb226.csv'))
    cherry = pd.read_csv(os.path.join(compare_dir,
                                       'cherry_final_prediction_gtdb.csv'))

    # ---- exclusion: remove sequences where iPHoP predicts to genus level ----
    has_genus = iphop['host_genus_lineage'].str.contains('g__', na=False)
    excluded_ids = set(iphop.loc[has_genus, 'Virus'])
    logger.info(f"  Comparison: excluding {len(excluded_ids)} sequences "
                f"(iPHoP genus-level)")

    # ---- filter & deduplicate each tool ------------------------------------
    iphop = iphop[~iphop['Virus'].isin(excluded_ids)].copy()
    iphop = (iphop.sort_values('Confidence score', ascending=False)
                  .drop_duplicates(subset='Virus', keep='first'))

    pt = pt_predictions[~pt_predictions['sequence_id'].isin(excluded_ids)].copy()
    pt = (pt.sort_values('score', ascending=False)
            .drop_duplicates(subset='sequence_id', keep='first'))
    # Keep sequences that pass the genus score threshold OR are flagged as
    # bacterial (the model made a confident decision — just not a genus one).
    has_bact_col = 'is_bacterial' in pt.columns
    if has_bact_col:
        pt = pt[(pt['score'] > pt_threshold) | pt['is_bacterial']]
    else:
        pt = pt[pt['score'] > pt_threshold]
    logger.info(f"  PT predictions after FDR threshold ({pt_threshold:.4f}): "
                f"{len(pt)}"
                + (f" ({pt['is_bacterial'].sum()} bacterial)"
                   if has_bact_col else ''))

    cherry = cherry[~cherry['contig_name'].isin(excluded_ids)].copy()
    cherry = (cherry.sort_values('Score_1', ascending=False)
                    .drop_duplicates(subset='contig_name', keep='first'))

    target = target[~target['Phage'].isin(excluded_ids)].copy()
    logger.info(f"  Target sequences after exclusion: {len(target)}")

    # ---- parse lineages into per-rank columns ------------------------------
    for i, lvl in enumerate(LEVEL_NAMES):
        target[f'true_{lvl}'] = target['host_genus_lineage'].apply(
            lambda x, _i=i: _parse_lineage_target(x)[_i])
        iphop[f'pred_{lvl}'] = iphop['host_genus_lineage'].apply(
            lambda x, _i=i: _parse_lineage_gtdb(x)[_i])
        pt[f'pred_{lvl}'] = pt['genus'].apply(
            lambda x, _i=i: _parse_lineage_pt(x)[_i])
        cherry[f'pred_{lvl}'] = cherry['Top_1_label_lineage'].apply(
            lambda x, _i=i: _parse_lineage_gtdb(x)[_i])

    # For bacterial-flagged PT sequences, overwrite lineage predictions with a
    # sentinel so they count as "has prediction" but never match the target.
    if has_bact_col:
        bact_mask = pt['is_bacterial']
        for lvl in LEVEL_NAMES:
            pt.loc[bact_mask, f'pred_{lvl}'] = '__bacterial__'

    # ---- merge predictions onto target -------------------------------------
    true_cols = ['Phage', 'dataset'] + [f'true_{l}' for l in LEVEL_NAMES]
    pred_cols = lambda col: [col] + [f'pred_{l}' for l in LEVEL_NAMES]

    iphop_m  = target[true_cols].merge(
        iphop[pred_cols('Virus')],
        left_on='Phage', right_on='Virus', how='left')
    # PT merge: include is_bacterial flag
    pt_merge_cols = pred_cols('sequence_id')
    if has_bact_col:
        pt_merge_cols = pt_merge_cols + ['is_bacterial']
    pt_m     = target[true_cols].merge(
        pt[pt_merge_cols],
        left_on='Phage', right_on='sequence_id', how='left')
    if has_bact_col:
        pt_m['is_bacterial'] = pt_m['is_bacterial'].fillna(False)
    cherry_m = target[true_cols].merge(
        cherry[pred_cols('contig_name')],
        left_on='Phage', right_on='contig_name', how='left')

    # ---- compute per-dataset, per-rank accuracy ----------------------------
    datasets = [ds for ds in ['cow_fecal', 'human_gut', 'waste_water', 'refseq']
                if ds in target['dataset'].values]

    def _stats(merged_df, tool_name):
        rows = []
        has_bact = 'is_bacterial' in merged_df.columns
        for ds in datasets:
            sub = merged_df[merged_df['dataset'] == ds]
            total = len(sub)
            n_bacterial = int(sub['is_bacterial'].sum()) if has_bact else 0
            for lvl in LEVEL_NAMES:
                has_pred = int(sub[f'pred_{lvl}'].notna().sum())
                correct  = int((sub[f'pred_{lvl}'] == sub[f'true_{lvl}']).sum())
                rows.append({
                    'dataset': ds, 'rank': lvl, 'tool': tool_name,
                    'total': total, 'has_pred': has_pred, 'correct': correct,
                    'ratio_correct': correct / total if total else 0,
                    'ratio_wrong': (has_pred - correct) / total if total else 0,
                    'ratio_bacterial': n_bacterial / total if total else 0,
                })
        return pd.DataFrame(rows)

    stats = pd.concat([
        _stats(iphop_m,  'iPHoP'),
        _stats(pt_m,     'PT'),
        _stats(cherry_m, 'CHERRY'),
    ], ignore_index=True)

    return stats, datasets


def plot_tool_comparison(axes, stats: pd.DataFrame, datasets: List[str]):
    """Grouped bar chart comparing tool accuracy across datasets.

    One subplot per dataset, with grouped bars for each tool × taxonomic rank.
    Solid bars = correct, translucent extensions = incorrect predictions.
    For PT, a hatched overlay within the wrong portion shows sequences
    classified as bacterial_fragment.
    """
    tools = ['iPHoP', 'PT', 'CHERRY']
    n_levels = len(LEVEL_NAMES)
    bar_width = 0.08
    rank_cols = [RANK_COLORS[r] for r in LEVEL_NAMES]
    has_bact = 'ratio_bacterial' in stats.columns

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        ds_stats = stats[stats['dataset'] == ds]

        for t_idx, tool in enumerate(tools):
            ts = ds_stats[ds_stats['tool'] == tool]
            offsets = np.arange(n_levels) - (n_levels - 1) / 2
            x_pos = t_idx + offsets * (bar_width + 0.01)

            for j in range(n_levels):
                c = ts.iloc[j]['ratio_correct']
                w = ts.iloc[j]['ratio_wrong']
                ax.bar(x_pos[j], c, width=bar_width, color=rank_cols[j],
                       edgecolor='white', linewidth=0.5,
                       label=LEVEL_NAMES[j]
                             if (t_idx == 0 and ax_idx == 0) else None)
                ax.bar(x_pos[j], w, bottom=c, width=bar_width,
                       color=rank_cols[j], edgecolor='white', linewidth=0.5,
                       alpha=0.35)

                # Bacterial overlay: hatched region at top of wrong bar (PT only)
                if has_bact and tool == 'PT':
                    b = ts.iloc[j]['ratio_bacterial']
                    if b > 0:
                        ax.bar(x_pos[j], b, bottom=c + w - b,
                               width=bar_width, facecolor='none',
                               edgecolor='#333333', linewidth=0.6,
                               hatch='///', zorder=4)

        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(tools, fontsize=FONT_SIZES['tick'],
                           fontweight='bold')
        n_seqs = ds_stats['total'].iloc[0]
        ax.set_title(f'{DATASET_LABELS.get(ds, ds)}\n(n = {n_seqs})',
                     fontweight='bold', pad=8)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(tools) - 0.5)

    axes[0].set_ylabel('Fraction of sequences')


def make_comparison_figure(compare_dir: str, pt_predictions: pd.DataFrame,
                           pt_threshold: float,
                           out_path: str = 'comparison.png',
                           dpi: int = 200):
    """Figure 5: PhageTransformer vs. iPHoP vs. CHERRY across test datasets."""
    from matplotlib.patches import Patch
    setup_style()

    stats, datasets = load_comparison_data(compare_dir, pt_predictions,
                                           pt_threshold)
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(4.0 * n_ds, 5.0), sharey=True)
    if n_ds == 1:
        axes = [axes]

    plot_tool_comparison(axes, stats, datasets)

    # Legend: rank patches + correct / incorrect / bacterial
    legend_elements = [Patch(facecolor=RANK_COLORS[r], label=r)
                       for r in LEVEL_NAMES]
    legend_elements += [
        Patch(facecolor='gray', alpha=1.0,  label='Correct'),
        Patch(facecolor='gray', alpha=0.35, label='Incorrect'),
        Patch(facecolor='none', edgecolor='#333333', linewidth=0.6,
              hatch='///', label='Bacterial'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=len(legend_elements), frameon=False,
               fontsize=FONT_SIZES['legend'],
               bbox_to_anchor=(0.5, 1.02))

    _suptitle(fig, 'Host prediction accuracy by taxonomic rank', y=1.08)
    fig.tight_layout()

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"Figure saved: {out_path}")

    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    logger.info(f"Figure saved: {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PhageTransformer model. '
                    'Produces figures in <run_dir>/plots/',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to training run directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Override dataset directory')
    parser.add_argument('--testset_dir', type=str, default=None,
                        help='Override testset directory')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold (default: from calibration.json)')
    parser.add_argument('--fdr', type=float, default=None,
                        help='Use FDR threshold from calibration (e.g. 0.1)')
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
    parser.add_argument('--compare_dir', type=str, default=None,
                        help='Directory with tool comparison CSVs '
                             '(target_lineages.csv, iPHoP, CHERRY, PT preds)')
    parser.add_argument('--compare_fdr', type=float, default=0.1,
                        help='FDR level for PT threshold in comparison figure')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- presentation mode -----------------------------------------------
    global PRESENTATION_MODE
    if args.presentation:
        PRESENTATION_MODE = True
        _scale = 1.5
        for key in FONT_SIZES:
            FONT_SIZES[key] = round(FONT_SIZES[key] * _scale, 1)
        logger.info("Presentation mode: font sizes scaled by %.1fx, "
                    "panel letters disabled", _scale)

    # ---- output directory -------------------------------------------------
    plot_dir = os.path.join(args.run_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # ---- load model and calibration --------------------------------------
    model, calib = load_model_and_calibration(args.run_dir, args.checkpoint, device)
    temperature = calib['temperature']
    hosts = np.array(calib['hosts'])
    patch_nt_len = calib['model_config']['patch_nt_len']

    # Determine threshold
    if args.fdr is not None:
        fdr_key = f"fdr_{int(args.fdr * 100):02d}"
        fdr_thresholds = calib.get('fdr_thresholds', {})
        if fdr_key in fdr_thresholds:
            threshold = fdr_thresholds[fdr_key]
            logger.info(f"Using FDR {args.fdr} threshold: {threshold:.4f}")
        else:
            logger.warning(f"FDR key '{fdr_key}' not in calibration, "
                          f"available: {list(fdr_thresholds.keys())}")
            threshold = calib.get('threshold', 0.5)
    elif args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = calib.get('threshold', 0.5)

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
    history = load_training_history(args.run_dir)

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
        n_pcs=50, dpi=args.dpi)

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

    # ---- Figure 5: Tool comparison (optional) ----------------------------
    if args.compare_dir:
        logger.info("Generating tool comparison figure ...")
        # Resolve FDR threshold for PT in comparison
        cmp_fdr_key = f"fdr_{int(args.compare_fdr * 100):02d}"
        cmp_fdr_thresholds = calib.get('fdr_thresholds', {})
        if cmp_fdr_key in cmp_fdr_thresholds:
            pt_compare_threshold = cmp_fdr_thresholds[cmp_fdr_key]
            logger.info(f"  PT comparison threshold (FDR {args.compare_fdr*100:.0f}%): "
                        f"{pt_compare_threshold:.4f}")
        else:
            pt_compare_threshold = threshold
            logger.warning(f"  FDR key '{cmp_fdr_key}' not in calibration, "
                          f"using main threshold {threshold:.4f}")

        # Load test set with IDs and run live inference
        logger.info("  Loading external test set for comparison ...")
        cmp_ids, cmp_seqs, cmp_datasets, cmp_labels = load_test_with_ids(
            ts_path, hosts)
        logger.info(f"  External test set: {len(cmp_seqs)} sequences")

        pt_predictions = predict_test_for_comparison(
            model, cmp_seqs, cmp_ids, hosts, temperature,
            tokenizer, device,
            patch_nt_len=patch_nt_len, max_patches=args.max_patches,
            eval_stride=eval_stride,
            batch_size=args.batch_size, num_workers=args.num_workers)
        logger.info(f"  PT predictions: {len(pt_predictions)} sequences, "
                    f"{(pt_predictions['score'] > pt_compare_threshold).sum()} "
                    f"above FDR threshold")

        make_comparison_figure(
            compare_dir=args.compare_dir,
            pt_predictions=pt_predictions,
            pt_threshold=pt_compare_threshold,
            out_path=_output_path(os.path.join(plot_dir, 'comparison.png')),
            dpi=args.dpi)

    logger.info(f"All outputs saved to {plot_dir}/")
    logger.info("Done.")


if __name__ == '__main__':
    main()
