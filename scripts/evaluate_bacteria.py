#!/usr/bin/env python3
"""Evaluate PhageTransformer bacterial detection and classification.

Standalone script — not part of the installed package.

Samples bacterial subsequences from validation regions of bacterial
genomes and creates phage-bacteria chimeric sequences to evaluate:
  1. Classification: can the model identify the correct host genus?
  2. Detection: can the model detect that the input is bacterial?
  3. Sensitivity: how does the bacterial_fragment score respond to
     the fraction of bacterial DNA in a chimeric sequence?

Usage:
    python evaluate_bacteria.py \
        --model_dir ./models/PT \
        --dataset_dir ./data \
        --host_genome_dir ./genomes
"""

import argparse
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from phagetransformer.model import CodonTokenizer
from phagetransformer.predict import load_model_and_calibration
from phagetransformer.dataset import (
    load_phage_host_merged, BacterialGenomeStore,
    BacterialSequenceDataset, sequence_collate_fn,
)
from eval_utils import (
    COLORS, LEVEL_NAMES, FONT_SIZES,
    setup_style, _add_panel_letters, _output_path, _suptitle,
    enable_presentation_mode,
    collect_logits, parse_lineages, aggregate_to_level,
)

logger = logging.getLogger(__name__)

RANK_COLORS = {
    'Phylum': COLORS['primary'],
    'Class':  COLORS['tertiary'],
    'Order':  COLORS['quaternary'],
    'Family': COLORS['quinary'],
    'Genus':  COLORS['secondary'],
}


# ---------------------------------------------------------------------------
# Sampling & inference
# ---------------------------------------------------------------------------

def sample_bacteria(model, genome_store, hosts, temperature, tokenizer,
                    device, phage_lengths,
                    patch_nt_len, max_patches, eval_stride,
                    n_samples, batch_size, num_workers):
    """Sample bacterial val subsequences and run model.

    Returns (genus_logits, genus_labels, bact_logits, bact_labels,
             genus_to_idx).
    """
    genus_to_idx = {}
    for i, h in enumerate(hosts):
        genus = h.split(';')[-1] if ';' in h else h
        genus_to_idx[genus] = i

    n_host_classes = len(hosts)
    agg_num_classes = n_host_classes + 1
    bact_idx = agg_num_classes - 1

    ds = BacterialSequenceDataset(
        genome_store, tokenizer, phage_lengths,
        n_samples=n_samples,
        agg_num_classes=agg_num_classes,
        genus_to_idx=genus_to_idx,
        patch_nt_len=patch_nt_len,
        max_patches=max_patches,
        is_train=False,
        eval_stride=eval_stride,
    )

    ldr_kw = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=sequence_collate_fn, **ldr_kw)

    logger.info(f"  Running inference on {n_samples} bacterial samples ...")
    logits, labels = collect_logits(model, loader, device)
    logger.info(f"  logits: {logits.shape}  labels: {labels.shape}")

    genus_logits = logits[:, :n_host_classes]
    bact_logits = logits[:, bact_idx]
    genus_labels = labels[:, :n_host_classes]
    bact_labels = labels[:, bact_idx]

    n_with_genus = (genus_labels.sum(1) > 0).sum().item()
    logger.info(f"  {n_with_genus}/{len(logits)} samples have a matching "
                f"host genus")

    return genus_logits, genus_labels, bact_logits, bact_labels, genus_to_idx


# ---------------------------------------------------------------------------
# Chimeric sequences
# ---------------------------------------------------------------------------

class ChimeraDataset(Dataset):
    """Dataset of phage-bacteria chimeric sequences.

    For each sample, takes a phage validation sequence, cuts out a
    contiguous region of length ``ratio * len(phage)`` at a random
    position, and replaces it with bacterial DNA from the correct
    host genus.
    """

    def __init__(self, phage_seqs, phage_labels, hosts, genome_store,
                 tokenizer, ratios, n_per_ratio,
                 patch_nt_len, max_patches, eval_stride):
        self.tokenizer = tokenizer
        self.patch_nt_len = patch_nt_len
        self.max_patches = max_patches
        self.eval_stride = eval_stride

        # Build genus → bacterial species mapping
        genus_to_species = defaultdict(list)
        for sp in genome_store.species_list:
            genus_to_species[sp.split()[0]].append(sp)

        # Build genus_to_idx
        genus_to_idx = {}
        for i, h in enumerate(hosts):
            genus = h.split(';')[-1] if ';' in h else h
            genus_to_idx[genus] = i

        # Select phage sequences that have a host genus with a bacterial genome
        self.samples = []  # (chimera_seq, ratio, true_genus_idx)
        rng = np.random.RandomState(42)

        # Find eligible phages: those with at least one host genus in genome_store
        eligible = []
        for idx in range(len(phage_seqs)):
            label_indices = phage_labels[idx].nonzero()[0]
            for li in label_indices:
                genus = hosts[li].split(';')[-1] if ';' in hosts[li] else hosts[li]
                if genus in genus_to_species:
                    eligible.append((idx, genus, int(li)))
                    break

        if not eligible:
            logger.warning("  No eligible phage sequences with matching "
                           "bacterial genomes")
            return

        logger.info(f"  Chimera dataset: {len(eligible)} eligible phages, "
                    f"{len(ratios)} ratios, {n_per_ratio} per ratio")

        for ratio in ratios:
            for _ in range(n_per_ratio):
                # Pick a random eligible phage
                phage_idx, genus, genus_idx = eligible[rng.randint(len(eligible))]
                phage_seq = phage_seqs[phage_idx]
                seq_len = len(phage_seq)

                bact_len = int(ratio * seq_len)

                if bact_len == 0:
                    # Pure phage
                    self.samples.append((phage_seq, ratio, genus_idx))
                    continue

                if bact_len >= seq_len:
                    # Pure bacterial
                    sp = rng.choice(genus_to_species[genus])
                    bact_seq, _ = genome_store.sample_subseq('val', seq_len)
                    self.samples.append((bact_seq[:seq_len], ratio, genus_idx))
                    continue

                # Create chimera: cut a piece from phage, replace with bacterial
                cut_start = rng.randint(0, max(1, seq_len - bact_len))
                cut_end = cut_start + bact_len

                # Get bacterial insert
                sp = rng.choice(genus_to_species[genus])
                bact_chunk, _ = genome_store.sample_subseq('val', bact_len)
                if len(bact_chunk) < bact_len:
                    bact_chunk = bact_chunk + bact_chunk  # repeat if short
                bact_chunk = bact_chunk[:bact_len]

                chimera = phage_seq[:cut_start] + bact_chunk + phage_seq[cut_end:]
                assert len(chimera) == seq_len, \
                    f"Chimera length {len(chimera)} != {seq_len}"
                self.samples.append((chimera, ratio, genus_idx))

    def __len__(self):
        return len(self.samples)

    def _tile(self, seq):
        plen = self.patch_nt_len
        stride = self.eval_stride
        patches = []
        for start in range(0, max(1, len(seq) - plen + 1), stride):
            patches.append(seq[start:start + plen])
            if len(patches) >= self.max_patches:
                break
        if len(seq) > plen:
            last = seq[len(seq) - plen:]
            if not patches or last != patches[-1]:
                patches.append(last)
        if not patches:
            patches.append(seq)
        return patches[:self.max_patches]

    def __getitem__(self, idx):
        seq, ratio, genus_idx = self.samples[idx]
        patches = self._tile(seq)
        toks = [self.tokenizer.tokenize(p) for p in patches]
        max_cl = max(t.size(1) for t in toks)
        padded = torch.zeros(len(toks), 6, max_cl, dtype=torch.long)
        for i, t in enumerate(toks):
            padded[i, :, :t.size(1)] = t
        # Encode ratio and true genus index in the label
        meta = torch.tensor([ratio, float(genus_idx)], dtype=torch.float32)
        return padded, len(toks), meta


def run_chimera_experiment(model, phage_seqs, phage_labels, hosts,
                           genome_store, temperature, tokenizer, device,
                           patch_nt_len, max_patches, eval_stride,
                           n_per_ratio, batch_size, num_workers):
    """Create chimeric sequences and measure bacterial_fragment score.

    Returns DataFrame with columns: ratio, bact_score.
    """
    ratios = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    n_host_classes = len(hosts)

    ds = ChimeraDataset(
        phage_seqs, phage_labels, hosts, genome_store, tokenizer,
        ratios, n_per_ratio, patch_nt_len, max_patches, eval_stride)

    if len(ds) == 0:
        logger.warning("  No chimeric samples created")
        return pd.DataFrame(columns=['ratio', 'bact_score'])

    ldr_kw = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=sequence_collate_fn, **ldr_kw)

    logger.info(f"  Running inference on {len(ds)} chimeric sequences ...")
    logits, meta_labels = collect_logits(model, loader, device)

    # Extract bacterial_fragment logit (last column beyond host classes)
    if logits.shape[1] > n_host_classes:
        bact_logits = logits[:, n_host_classes]
    else:
        bact_logits = torch.full((len(logits),), -10.0)

    bact_scores = torch.sigmoid(bact_logits / temperature).numpy()
    ratios_out = meta_labels[:, 0].numpy()
    true_genus_indices = meta_labels[:, 1].long()

    # Check genus prediction correctness
    genus_logits = logits[:, :n_host_classes]
    genus_probs = torch.sigmoid(genus_logits / temperature)

    genus_correct = []
    for i in range(len(logits)):
        gi = true_genus_indices[i].item()
        true_prob = genus_probs[i, int(gi)].item()
        # Correct if the true genus has the highest prob among all genera
        best_prob = genus_probs[i].max().item()
        genus_correct.append(true_prob >= best_prob - 1e-6)

    df = pd.DataFrame({
        'ratio': ratios_out,
        'bact_score': bact_scores,
        'genus_correct': genus_correct,
    })
    logger.info(f"  Chimera results: {len(df)} samples across "
                f"{df['ratio'].nunique()} ratios, "
                f"genus correct: {df['genus_correct'].mean():.1%}")
    return df


# ---------------------------------------------------------------------------
# PR curves
# ---------------------------------------------------------------------------

def _compute_pr_curve(probs, labels, n_thresholds=200):
    """Compute micro- and macro-averaged PR curves."""
    thresholds = np.linspace(0, 1, n_thresholds)
    micro_p = np.zeros(n_thresholds)
    micro_r = np.zeros(n_thresholds)
    macro_p = np.zeros(n_thresholds)
    macro_r = np.zeros(n_thresholds)

    support = labels.sum(0)
    has_support = support > 0

    for ti, t in enumerate(thresholds):
        preds = (probs >= t).float()
        tp = (preds * labels).sum(0)
        fp = (preds * (1 - labels)).sum(0)
        fn = ((1 - preds) * labels).sum(0)

        tp_sum = tp.sum().item()
        fp_sum = fp.sum().item()
        fn_sum = fn.sum().item()
        micro_p[ti] = tp_sum / max(tp_sum + fp_sum, 1e-12)
        micro_r[ti] = tp_sum / max(tp_sum + fn_sum, 1e-12)

        per_p = tp / (tp + fp).clamp(min=1e-12)
        per_r = tp / (tp + fn).clamp(min=1e-12)
        no_pred = (tp + fp) == 0
        per_p[no_pred & has_support] = 1.0

        if has_support.any():
            macro_p[ti] = per_p[has_support].mean().item()
            macro_r[ti] = per_r[has_support].mean().item()

    return thresholds, micro_p, micro_r, macro_p, macro_r


def _compute_auprc(precision, recall):
    """Area under the PR curve (trapezoidal)."""
    order = np.argsort(recall)
    return np.trapz(precision[order], recall[order])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _violin_box(ax, data, labels_list, colors, ylabel, title):
    """Shared violin + box plot logic."""
    positions = list(range(len(data)))

    vp = ax.violinplot(data, positions=positions, showextrema=False,
                       showmedians=False)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.4)

    bp = ax.boxplot(data, positions=positions, widths=0.3,
                    patch_artist=True, showfliers=False, zorder=3)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color=COLORS['text_light'], linewidth=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_list, fontsize=FONT_SIZES['tick'])
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for i, d in enumerate(data):
        if len(d) > 0:
            med = np.median(d)
            ax.text(i, med + 0.03, f'{med:.2f}', ha='center',
                    fontsize=FONT_SIZES['bar_value'], color=COLORS['text'])


def plot_classification_violins(ax, genus_probs, genus_labels, hosts,
                                threshold):
    """Panel A: per-rank violin/box plots of per-class F1."""
    max_depth, lineages = parse_lineages(list(hosts))

    level_names = []
    data = []
    colors = []
    for level in range(max_depth):
        name = LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f'Level_{level}'
        if level == max_depth - 1:
            probs_l, labels_l = genus_probs, genus_labels
        else:
            probs_l, labels_l, _ = aggregate_to_level(
                genus_probs, genus_labels, lineages, level)

        preds = (probs_l >= threshold).float()
        tp = (preds * labels_l).sum(0)
        fp = (preds * (1 - labels_l)).sum(0)
        fn = ((1 - preds) * labels_l).sum(0)
        per_p = tp / (tp + fp).clamp(min=1e-12)
        per_r = tp / (tp + fn).clamp(min=1e-12)
        per_f1 = 2 * per_p * per_r / (per_p + per_r).clamp(min=1e-12)

        has_support = (tp + fn) > 0
        f1_vals = per_f1[has_support].numpy()
        if len(f1_vals) == 0:
            f1_vals = np.array([0.0])

        level_names.append(name)
        data.append(f1_vals)
        colors.append(RANK_COLORS.get(name, '#999'))

    _violin_box(ax, data, level_names, colors,
                'Per-class F1',
                'Classification F1 per rank\n(bacterial samples)')


def plot_classification_pr(ax, genus_probs, genus_labels, hosts):
    """Panel B: macro-averaged classification PR curves per rank."""
    max_depth, lineages = parse_lineages(list(hosts))

    for level in range(max_depth):
        name = LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f'Level_{level}'
        if level == max_depth - 1:
            probs_l, labels_l = genus_probs, genus_labels
        else:
            probs_l, labels_l, _ = aggregate_to_level(
                genus_probs, genus_labels, lineages, level)

        _, _, _, macro_p, macro_r = _compute_pr_curve(probs_l, labels_l)
        auprc = _compute_auprc(macro_p, macro_r)
        color = RANK_COLORS.get(name, '#999')
        ax.plot(macro_r, macro_p, color=color, linewidth=1.5, alpha=0.85,
                label=f'{name} (AUPRC={auprc:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title('Classification macro PR\n(bacterial samples)',
                 fontweight='bold', pad=8)
    ax.legend(fontsize=FONT_SIZES['legend_small'], loc='lower left',
              frameon=True, framealpha=0.9, edgecolor=COLORS['grid'])
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')


def plot_detection_violins(ax, bact_probs_bact, genus_labels_bact,
                           hosts, threshold):
    """Panel C: detection rate per taxonomic group as violin plots."""
    max_depth, lineages = parse_lineages(list(hosts))
    bact_detected = (bact_probs_bact >= threshold).float()

    level_names = []
    data = []
    colors = []
    for level in range(max_depth):
        name = LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else f'Level_{level}'
        if level == max_depth - 1:
            labels_l = genus_labels_bact
        else:
            _, labels_l, _ = aggregate_to_level(
                bact_probs_bact.unsqueeze(1).expand_as(genus_labels_bact),
                genus_labels_bact, lineages, level)

        n_classes = labels_l.shape[1]
        rates = []
        for c in range(n_classes):
            mask = labels_l[:, c] > 0
            if mask.sum() < 2:
                continue
            rate = bact_detected[mask].mean().item()
            rates.append(rate)

        if not rates:
            rates = [0.0]

        level_names.append(name)
        data.append(np.array(rates))
        colors.append(RANK_COLORS.get(name, '#999'))

    _violin_box(ax, data, level_names, colors,
                'Detection rate per group',
                'Detection rate by taxonomic group\n(bacterial samples)')


def plot_chimera_sensitivity(ax, chimera_df, threshold):
    """Panel D: bacterial_fragment score vs bacterial DNA ratio.

    Scatter of individual chimeras colored by whether the correct host
    genus was predicted, plus median/IQR trend line.
    """
    if chimera_df.empty:
        ax.text(0.5, 0.5, 'No chimera data', transform=ax.transAxes,
                ha='center', va='center', fontsize=FONT_SIZES['fallback_msg'])
        ax.set_axis_off()
        return

    correct = chimera_df['genus_correct']
    ax.scatter(chimera_df.loc[correct, 'ratio'],
               chimera_df.loc[correct, 'bact_score'],
               s=12, alpha=0.3, color=COLORS['tertiary'],
               edgecolors='none', zorder=2, label='Genus correct')
    ax.scatter(chimera_df.loc[~correct, 'ratio'],
               chimera_df.loc[~correct, 'bact_score'],
               s=12, alpha=0.3, color=COLORS['secondary'],
               edgecolors='none', zorder=2, label='Genus incorrect')

    # Median + IQR per ratio bin
    grouped = chimera_df.groupby('ratio')['bact_score']
    medians = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)

    ax.plot(medians.index, medians.values, color=COLORS['text'],
            linewidth=2, zorder=4, label='Median')
    ax.fill_between(medians.index, q25.values, q75.values,
                    color=COLORS['text'], alpha=0.1, zorder=3,
                    label='IQR')

    # Threshold line
    ax.axhline(threshold, color=COLORS['text_light'], linestyle='--',
               linewidth=1, alpha=0.7, zorder=1,
               label=f'Threshold ({threshold:.2f})')

    ax.set_xlabel('Bacterial DNA fraction')
    ax.set_ylabel('bacterial_fragment score')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('Detection sensitivity\n(phage–bacteria chimeras)',
                 fontweight='bold', pad=8)
    ax.legend(fontsize=FONT_SIZES['legend_small'], loc='upper left',
              frameon=True, framealpha=0.9, edgecolor=COLORS['grid'])
    ax.grid(alpha=0.3, linestyle='--')


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_figure(fig, out_path: str, dpi: int = 200):
    fig.savefig(out_path, dpi=dpi)
    logger.info(f"Figure saved: {out_path}")
    root = os.path.splitext(out_path)[0]
    for ext in ('.pdf', '.svg'):
        path = root + ext
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_bacteria_figure(genus_probs_cls, genus_labels_cls,
                         bact_probs_bact, genus_labels_bact,
                         hosts, threshold, chimera_df,
                         out_path='bacteria_evaluation.png', dpi=200):
    """Four-panel bacterial evaluation figure.

    A: Classification F1 violin plots per rank
    B: Classification macro-averaged PR curves per rank
    C: Detection rate violin plots per taxonomic group
    D: Bacterial score vs bacterial DNA fraction (chimeras)
    """
    setup_style()

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.08, right=0.96, top=0.93, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_classification_violins(ax_a, genus_probs_cls, genus_labels_cls,
                                hosts, threshold)

    ax_b = fig.add_subplot(gs[0, 1])
    plot_classification_pr(ax_b, genus_probs_cls, genus_labels_cls, hosts)

    ax_c = fig.add_subplot(gs[1, 0])
    plot_detection_violins(ax_c, bact_probs_bact, genus_labels_bact,
                           hosts, threshold)

    ax_d = fig.add_subplot(gs[1, 1])
    plot_chimera_sensitivity(ax_d, chimera_df, threshold)

    _add_panel_letters([ax_a, ax_b, ax_c, ax_d], 'ABCD', x=-0.08, y=1.06)
    _suptitle(fig, 'Bacterial detection and classification evaluation',
              y=0.98)

    _save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate bacterial detection and classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Phage dataset directory (for val set and '
                             'sequence length distribution)')
    parser.add_argument('--host_genome_dir', type=str, required=True,
                        help='Directory with host_genome_manifest.tsv and '
                             'genome FASTA files')
    parser.add_argument('--output', '-o', type=str,
                        default='bacteria_evaluation.png',
                        help='Output filename')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold (default: from calibration)')
    parser.add_argument('--fdr', type=float, default=None,
                        help='FDR level (default: 0.2 if --threshold not set)')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of bacterial subsequences to sample')
    parser.add_argument('--n_chimeras', type=int, default=100,
                        help='Number of chimeric sequences per ratio')
    parser.add_argument('--max_patches', type=int, default=512)
    parser.add_argument('--eval_stride', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--presentation', action='store_true',
                        help='Increase font sizes for presentations')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.presentation:
        enable_presentation_mode()
        logger.info("Presentation mode enabled")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- load model and calibration --------------------------------------
    model, calib = load_model_and_calibration(
        args.model_dir, args.checkpoint, device)
    temperature = calib['temperature']
    hosts = np.array(calib['hosts'])
    patch_nt_len = calib['model_config']['patch_nt_len']

    # ---- resolve threshold -----------------------------------------------
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Threshold (fixed): {threshold:.4f}")
    else:
        fdr = args.fdr if args.fdr is not None else 0.2
        fdr_key = f"fdr_{int(fdr * 100):02d}"
        fdr_thresholds = calib.get('fdr_thresholds', {})
        if fdr_key in fdr_thresholds:
            threshold = fdr_thresholds[fdr_key]
            logger.info(f"Threshold (FDR {fdr*100:.0f}%): {threshold:.4f}")
        else:
            threshold = calib.get('threshold', 0.5)
            logger.warning(f"FDR key '{fdr_key}' not in calibration, "
                          f"using default threshold {threshold:.4f}")

    # ---- load phage data -------------------------------------------------
    logger.info("Loading phage dataset ...")
    train_seqs, _, val_seqs, val_labels, hosts_from_data = \
        load_phage_host_merged(args.dataset_dir)
    phage_lengths = [len(s) for s in train_seqs]
    logger.info(f"  {len(phage_lengths)} phage train sequences "
                f"(median len={np.median(phage_lengths):.0f})")
    logger.info(f"  {len(val_seqs)} phage val sequences")
    del train_seqs

    # ---- load bacterial genomes ------------------------------------------
    logger.info("Loading bacterial genomes ...")
    genome_store = BacterialGenomeStore(args.host_genome_dir)

    # ---- bacterial inference ---------------------------------------------
    tokenizer = CodonTokenizer()
    eval_stride = args.eval_stride or patch_nt_len // 2

    genus_logits_b, genus_labels_b, bact_logits_b, bact_labels_b, _ = \
        sample_bacteria(
            model, genome_store, hosts, temperature, tokenizer, device,
            phage_lengths,
            patch_nt_len=patch_nt_len,
            max_patches=args.max_patches,
            eval_stride=eval_stride,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # ---- chimera experiment ----------------------------------------------
    logger.info("Running chimera experiment ...")
    chimera_df = run_chimera_experiment(
        model, val_seqs, val_labels, hosts, genome_store,
        temperature, tokenizer, device,
        patch_nt_len=patch_nt_len,
        max_patches=args.max_patches,
        eval_stride=eval_stride,
        n_per_ratio=args.n_chimeras,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---- convert to probs ------------------------------------------------
    genus_probs_b = torch.sigmoid(genus_logits_b / temperature)
    bact_probs_b = torch.sigmoid(bact_logits_b / temperature)

    # ---- log summary stats -----------------------------------------------
    n_bact = len(bact_probs_b)
    bact_detected = (bact_probs_b >= threshold).float()
    logger.info(f"  Detection rate (bacterial): "
                f"{bact_detected.mean():.1%} "
                f"({int(bact_detected.sum())}/{n_bact})")

    # Classification: only samples with matching host genus
    has_genus = genus_labels_b.sum(1) > 0
    genus_probs_cls = genus_probs_b[has_genus]
    genus_labels_cls = genus_labels_b[has_genus]
    logger.info(f"  Classification: {has_genus.sum()} samples with matching "
                f"host genus")

    # ---- generate figure -------------------------------------------------
    logger.info("Generating bacterial evaluation figure ...")
    make_bacteria_figure(
        genus_probs_cls, genus_labels_cls,
        bact_probs_b, genus_labels_b,
        hosts, threshold, chimera_df,
        out_path=args.output, dpi=args.dpi)

    logger.info("Done.")


if __name__ == '__main__':
    main()
