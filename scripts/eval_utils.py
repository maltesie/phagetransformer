"""Shared evaluation utilities for PhageTransformer.

Contains styling, inference, taxonomic aggregation and metric computation
used by both the evaluation and comparison scripts.
"""

import gzip
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader

from phagetransformer.dataset import PatchSequenceDataset, sequence_collate_fn
from phagetransformer.train import _unpack_sequence_batch

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
    'base':           11,
    'suptitle':       12,
    'title':          11,
    'label':          11,
    'tick':           10,
    'tick_small':      9,
    'tick_tiny':       7.5,
    'legend':         10,
    'legend_small':    8,
    'annotation':      8,
    'bar_value':       6,
    'bar_label':       7,
    'panel_letter':   13,
    'codon':           5.5,
    'colorbar':        9,
    'secondary_axis':  7,
    'fallback_msg':   10,
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


def enable_presentation_mode(scale: float = 1.5):
    """Scale font sizes for presentations and disable panel letters."""
    global PRESENTATION_MODE
    PRESENTATION_MODE = True
    for key in FONT_SIZES:
        FONT_SIZES[key] = round(FONT_SIZES[key] * scale, 1)


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

    seq_ids, seqs = [], []
    with gzip.open(fasta_path, 'rt') as fh:
        for rec in SeqIO.parse(fh, 'fasta'):
            seq_ids.append(rec.id)
            seqs.append(str(rec.seq))

    df = pd.read_csv(csv_path, delimiter=',', dtype=str)

    genus_index = {x: i for i, x in enumerate(hosts)}
    labels = np.zeros((len(df), len(genus_index)), dtype=np.float32)
    for i, hs in enumerate(df['host_genus_lineage']):
        for h in hs.split('|'):
            h_trunc = ';'.join(h.split(';')[1:])
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
        threshold: float = 0.5,
) -> pd.DataFrame:
    """Run model inference on test sequences and return predictions DataFrame.

    Returns DataFrame with columns: sequence_id, genus, score, is_bacterial.
    Multiple rows per sequence — one for each genus prediction above
    ``threshold``.  ``is_bacterial`` is True when the bacterial_fragment
    class probability exceeds the threshold AND all genus probabilities.
    """
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

    n_core = len(hosts)
    has_bact_class = all_probs.shape[1] > n_core

    core_probs = all_probs[:, :n_core]

    rows = []
    for i in range(len(seqs)):
        is_bact = False
        if has_bact_class:
            bact_prob = all_probs[i, n_core].item()
            is_bact = bact_prob >= threshold and bact_prob > core_probs[i].max().item()

        above = (core_probs[i] >= threshold).nonzero(as_tuple=True)[0]
        if len(above) > 0:
            for idx in above:
                rows.append({
                    'sequence_id': seq_ids[i],
                    'genus': hosts[idx.item()],
                    'score': core_probs[i, idx].item(),
                    'is_bacterial': is_bact,
                })
        elif is_bact:
            rows.append({
                'sequence_id': seq_ids[i],
                'genus': '',
                'score': 0.0,
                'is_bacterial': True,
            })

    df = pd.DataFrame(rows)
    n_seqs_with_pred = df['sequence_id'].nunique()
    n_bact = df.drop_duplicates('sequence_id')['is_bacterial'].sum() if len(df) else 0
    logger.info(f"  {len(df)} predictions for {n_seqs_with_pred} sequences"
                + (f" ({n_bact} bacterial)" if n_bact else ''))
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
        'n_classes_with_support': int(has.sum()),
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
    """Count actual per-class support from the full metadata CSV."""
    col = 'host_genus_lineage'
    if full_metadata_df is None or col not in full_metadata_df.columns:
        return {}

    genus_counts = defaultdict(int)
    for lineages_str in full_metadata_df[col].dropna():
        for lin in lineages_str.split('|'):
            genus_counts[lin] += 1

    genus_res = level_results.get('Genus', {})
    if not genus_res:
        return {}

    taxa = genus_res['taxon_names']
    genus_support = np.array([genus_counts.get(t, 0) for t in taxa],
                             dtype=float)

    real_support = {'Genus': genus_support}

    _, lineages = parse_lineages(hosts)
    for level_name, metrics in level_results.items():
        if level_name == 'Genus':
            continue
        level_taxa = metrics['taxon_names']
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
