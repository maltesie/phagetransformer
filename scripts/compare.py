#!/usr/bin/env python3
"""Tool comparison figure: PhageTransformer vs iPHoP vs CHERRY.

Standalone script — not part of the installed package.

Usage:
    python compare.py \
        --model_dir ./models/my_run \
        --compare_dir ./eval_combined \
        --testset_dir ./testset
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

from phagetransformer.model import CodonTokenizer
from phagetransformer.predict import load_model_and_calibration
from eval_utils import (
    COLORS, LEVEL_NAMES, FONT_SIZES,
    setup_style, _suptitle, enable_presentation_mode,
    load_test_with_ids, predict_test_for_comparison,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANK_COLORS = {
    'Phylum': COLORS['primary'],
    'Class':  COLORS['tertiary'],
    'Order':  COLORS['quaternary'],
    'Family': COLORS['quinary'],
    'Genus':  COLORS['secondary'],
}

DATASET_LABELS = {
    'cow_fecal':    'Cow fecal',
    'human_gut':    'Human gut',
    'waste_water':  'Waste water',
    'refseq':       'RefSeq',
}

HIC_DATASETS = ['cow_fecal', 'human_gut', 'waste_water']


# ---------------------------------------------------------------------------
# Lineage parsers
# ---------------------------------------------------------------------------

def _parse_lineage_target(lin: str) -> List[Optional[str]]:
    if pd.isna(lin) or lin == '':
        return [None] * 5
    parts = lin.split(';')
    return [parts[i] if i < len(parts) and parts[i] else None
            for i in range(1, 6)]


def _parse_lineage_gtdb(lin: str) -> List[Optional[str]]:
    if pd.isna(lin) or lin == '':
        return [None] * 5
    parts = lin.split(';')
    cleaned = [p.split('__', 1)[1] if '__' in p else p for p in parts]
    return [cleaned[i] if i < len(cleaned) and cleaned[i] else None
            for i in range(1, 6)]


def _parse_lineage_pt(lin: str) -> List[Optional[str]]:
    if pd.isna(lin) or lin == '':
        return [None] * 5
    parts = lin.split(';')
    return [parts[i] if i < len(parts) and parts[i] else None
            for i in range(0, 5)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_comparison_data(compare_dir: str, pt_predictions: pd.DataFrame,
                         pt_threshold: float):
    """Load all prediction data and return merged DataFrames per tool.

    Returns (target, iphop_m, pt_m, cherry_m, datasets) where *_m are
    left-joined onto target.
    """
    target = pd.read_csv(os.path.join(compare_dir, 'target_lineages.csv'))
    iphop  = pd.read_csv(os.path.join(compare_dir,
                                       'iphop_predictions_gtdb226.csv'))
    cherry = pd.read_csv(os.path.join(compare_dir,
                                       'cherry_final_prediction_gtdb.csv'))

    # Exclusion: remove sequences where iPHoP predicts to genus level
    has_genus = iphop['host_genus_lineage'].str.contains('g__', na=False)
    excluded_ids = set(iphop.loc[has_genus, 'Virus'])
    logger.info(f"  Excluding {len(excluded_ids)} sequences "
                f"(iPHoP genus-level)")

    iphop = iphop[~iphop['Virus'].isin(excluded_ids)].copy()
    cherry = cherry[~cherry['contig_name'].isin(excluded_ids)].copy()
    target = target[~target['Phage'].isin(excluded_ids)].copy()

    pt = pt_predictions[~pt_predictions['sequence_id'].isin(excluded_ids)].copy()
    has_bact_col = 'is_bacterial' in pt.columns
    if has_bact_col:
        pt = pt[(pt['score'] > pt_threshold) | pt['is_bacterial']]
    else:
        pt = pt[pt['score'] > pt_threshold]
    logger.info(f"  PT predictions after threshold: {len(pt)} rows for "
                f"{pt['sequence_id'].nunique()} sequences")

    # Parse lineages
    for i, lvl in enumerate(LEVEL_NAMES):
        target[f'true_{lvl}'] = target['host_genus_lineage'].apply(
            lambda x, _i=i: _parse_lineage_target(x)[_i])
        iphop[f'pred_{lvl}'] = iphop['host_genus_lineage'].apply(
            lambda x, _i=i: _parse_lineage_gtdb(x)[_i])
        pt[f'pred_{lvl}'] = pt['genus'].apply(
            lambda x, _i=i: _parse_lineage_pt(x)[_i])
        cherry[f'pred_{lvl}'] = cherry['Top_1_label_lineage'].apply(
            lambda x, _i=i: _parse_lineage_gtdb(x)[_i])

    # Merge onto target
    true_cols = ['Phage', 'dataset'] + [f'true_{l}' for l in LEVEL_NAMES]
    pred_cols = lambda col: [col] + [f'pred_{l}' for l in LEVEL_NAMES]

    iphop_m  = target[true_cols].merge(
        iphop[pred_cols('Virus')],
        left_on='Phage', right_on='Virus', how='left')

    pt_merge_cols = pred_cols('sequence_id')
    if has_bact_col:
        pt_merge_cols = pt_merge_cols + ['is_bacterial']
    pt_m = target[true_cols].merge(
        pt[pt_merge_cols],
        left_on='Phage', right_on='sequence_id', how='left')
    if has_bact_col:
        pt_m['is_bacterial'] = pt_m['is_bacterial'].fillna(False)

    cherry_m = target[true_cols].merge(
        cherry[pred_cols('contig_name')],
        left_on='Phage', right_on='contig_name', how='left')

    datasets = [ds for ds in ['cow_fecal', 'human_gut', 'waste_water', 'refseq']
                if ds in target['dataset'].values]

    return target, iphop_m, pt_m, cherry_m, datasets


def load_quality_data(compare_dir: str):
    """Load geNomad, CheckV, and plasmid results.

    Returns (genomad_viral_or_plasmid_ids, checkv_nonviral_ids) where:
    - genomad_viral_or_plasmid_ids includes both viral and plasmid contigs
      (so that plasmids are excluded from the non-viral set)
    - checkv_nonviral_ids are contigs with viral_genes/host_genes < 0.1
      (only when host_genes > 0)
    """
    genomad_path = os.path.join(compare_dir, 'hic_combined_virus_summary.tsv')
    checkv_path = os.path.join(compare_dir, 'quality_summary.tsv')
    plasmid_path = os.path.join(compare_dir, 'hic_combined_plasmid_summary.tsv')

    # geNomad: sequences in virus file are viral
    genomad = pd.read_csv(genomad_path, sep='\t')
    genomad_viral = set(genomad['seq_name'].str.replace(
        r'\|provirus_.*', '', regex=True))
    logger.info(f"  geNomad: {len(genomad_viral)} viral contigs")

    # Plasmids: exclude from non-viral classification
    plasmid_ids = set()
    if os.path.exists(plasmid_path):
        plasmids = pd.read_csv(plasmid_path, sep='\t')
        plasmid_ids = set(plasmids['seq_name'])
        logger.info(f"  geNomad: {len(plasmid_ids)} plasmid contigs (excluded "
                    f"from non-viral)")
    else:
        logger.info(f"  No plasmid file found, skipping plasmid exclusion")

    # Combine viral + plasmid → not-non-viral
    genomad_not_nonviral = genomad_viral | plasmid_ids

    # CheckV: viral_genes / host_genes < 0.1 (with host_genes > 0) → non-viral
    checkv = pd.read_csv(checkv_path, sep='\t')
    has_host = checkv['host_genes'] > 0
    low_ratio = (checkv['viral_genes'] / checkv['host_genes']) < 0.1
    checkv_nonviral = set(checkv.loc[has_host & low_ratio, 'contig_id'])
    logger.info(f"  CheckV: {len(checkv_nonviral)} contigs with "
                f"viral/host gene ratio < 0.1 (of {len(checkv)} total)")

    return genomad_not_nonviral, checkv_nonviral


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats(iphop_m, pt_m, cherry_m, datasets):
    """Compute per-dataset, per-rank any-match accuracy for all tools."""
    tools_data = [('iPHoP', iphop_m), ('PT', pt_m), ('CHERRY', cherry_m)]
    rows = []
    for tool_name, merged_df in tools_data:
        for ds in datasets:
            sub = merged_df[merged_df['dataset'] == ds]
            total = sub['Phage'].nunique()
            for lvl in LEVEL_NAMES:
                has_pred_per_seq = sub.groupby('Phage')[f'pred_{lvl}'].apply(
                    lambda s: s.notna().any())
                has_pred = int(has_pred_per_seq.sum())

                match = sub[f'pred_{lvl}'] == sub[f'true_{lvl}']
                correct_per_seq = match.groupby(sub['Phage']).any()
                correct = int(correct_per_seq.sum())

                rows.append({
                    'dataset': ds, 'rank': lvl, 'tool': tool_name,
                    'total': total, 'has_pred': has_pred, 'correct': correct,
                    'ratio_correct': correct / total if total else 0,
                    'ratio_wrong': (has_pred - correct) / total if total else 0,
                })
    return pd.DataFrame(rows)


def log_stats(stats: pd.DataFrame, label: str):
    """Log prediction ratio and precision for each dataset × tool × rank."""
    logger.info(f"  --- {label} ---")
    for ds in stats['dataset'].unique():
        ds_stats = stats[stats['dataset'] == ds]
        total = ds_stats['total'].iloc[0]
        logger.info(f"  {DATASET_LABELS.get(ds, ds)} (n={total}):")
        for tool in ['iPHoP', 'PT', 'CHERRY']:
            ts = ds_stats[ds_stats['tool'] == tool]
            parts = []
            for _, row in ts.iterrows():
                pred_ratio = row['has_pred'] / row['total'] if row['total'] else 0
                precision = row['correct'] / row['has_pred'] if row['has_pred'] else 0
                parts.append(f"{row['rank']}: "
                             f"pred={pred_ratio:.1%} "
                             f"prec={precision:.1%} "
                             f"({row['correct']}/{row['has_pred']}/{row['total']})")
            logger.info(f"    {tool:8s}  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_tool_comparison(axes, stats: pd.DataFrame, datasets: List[str]):
    """Grouped bar chart comparing tool accuracy."""
    tools = ['iPHoP', 'PT', 'CHERRY']
    n_levels = len(LEVEL_NAMES)
    bar_width = 0.08
    rank_cols = [RANK_COLORS[r] for r in LEVEL_NAMES]

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
                       edgecolor='white', linewidth=0.5)
                ax.bar(x_pos[j], w, bottom=c, width=bar_width,
                       color=rank_cols[j], edgecolor='white', linewidth=0.5,
                       alpha=0.35)

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


def plot_nonviral_overlap(ax, hic_ids: set,
                          genomad_viral_ids: set,
                          checkv_nonviral_ids: set,
                          pt_bacterial_ids: set):
    """Bar chart of non-viral classifications.

    Seven bars: 3 individual tool counts, 3 pairwise agreements,
    and the union (all three agree).
    """
    genomad_nv = hic_ids - genomad_viral_ids
    checkv_nv = hic_ids & checkv_nonviral_ids
    pt_nv = hic_ids & pt_bacterial_ids

    bars = [
        ('geNomad',           genomad_nv,              COLORS['tertiary']),
        ('CheckV',            checkv_nv,               COLORS['quaternary']),
        ('PT',                pt_nv,                   COLORS['primary']),
        ('geNomad\n∩ CheckV', genomad_nv & checkv_nv,  COLORS['quinary']),
        ('geNomad\n∩ PT',     genomad_nv & pt_nv,      COLORS['quinary']),
        ('CheckV\n∩ PT',      checkv_nv & pt_nv,       COLORS['quinary']),
        ('All three',         genomad_nv & checkv_nv & pt_nv, COLORS['text']),
    ]

    labels = [b[0] for b in bars]
    counts = [len(b[1]) for b in bars]
    colors = [b[2] for b in bars]

    x = np.arange(len(bars))
    bar_objs = ax.bar(x, counts, color=colors, edgecolor='white',
                      linewidth=0.8, alpha=0.85)

    for bar, count in zip(bar_objs, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom',
                    fontsize=FONT_SIZES['bar_value'], fontweight='bold',
                    color=COLORS['text'])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZES['tick_tiny'])
    ax.set_ylabel('Sequences')
    ax.set_title(f'Non-viral classifications\n'
                 f'(HiC datasets, n = {len(hic_ids)})',
                 fontweight='bold', pad=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_comparison_figure(compare_dir: str, pt_predictions: pd.DataFrame,
                           pt_threshold: float,
                           out_path: str = 'comparison.png',
                           dpi: int = 200):
    """Four-panel comparison figure.

    A (top-left):     RefSeq unfiltered
    B (top-right):    HiC datasets unfiltered
    C (bottom-left):  Non-viral counts (geNomad, CheckV, PT — individual,
                      pairwise, and union)
    D (bottom-right): HiC datasets filtered (consensus non-viral removed —
                      sequences all three tools agree are non-viral)
    """
    from matplotlib.patches import Patch
    setup_style()

    # Load prediction data
    target, iphop_m, pt_m, cherry_m, datasets = load_comparison_data(
        compare_dir, pt_predictions, pt_threshold)

    hic_ds = [ds for ds in HIC_DATASETS if ds in datasets]
    has_refseq = 'refseq' in datasets
    n_hic = len(hic_ds)

    # Compute stats for all datasets (unfiltered)
    stats_all = compute_stats(iphop_m, pt_m, cherry_m, datasets)
    log_stats(stats_all, 'Unfiltered')

    # Load quality data
    genomad_viral_ids, checkv_nonviral_ids = load_quality_data(compare_dir)

    # HiC sequence IDs
    hic_target = target[target['dataset'].isin(hic_ds)]
    hic_ids = set(hic_target['Phage'].unique())

    # PT bacterial IDs (HiC only)
    has_bact_col = 'is_bacterial' in pt_m.columns
    pt_bacterial_ids = set()
    if has_bact_col:
        hic_pt = pt_m[pt_m['dataset'].isin(hic_ds)]
        pt_bacterial_ids = set(
            hic_pt.loc[hic_pt['is_bacterial'], 'Phage'].unique())

    # Non-viral sets for each tool
    genomad_nonviral = hic_ids - genomad_viral_ids
    checkv_nv = hic_ids & checkv_nonviral_ids

    # Consensus non-viral: all three tools agree
    consensus_nonviral = genomad_nonviral & checkv_nv & pt_bacterial_ids
    consensus_keep = hic_ids - consensus_nonviral
    logger.info(f"  Consensus non-viral: {len(consensus_nonviral)} sequences "
                f"(all 3 tools agree) → {len(consensus_keep)}/{len(hic_ids)} "
                f"kept for panel D")

    iphop_filt  = iphop_m[iphop_m['Phage'].isin(consensus_keep) |
                          ~iphop_m['dataset'].isin(hic_ds)]
    pt_filt     = pt_m[pt_m['Phage'].isin(consensus_keep) |
                       ~pt_m['dataset'].isin(hic_ds)]
    cherry_filt = cherry_m[cherry_m['Phage'].isin(consensus_keep) |
                           ~cherry_m['dataset'].isin(hic_ds)]

    stats_filt = compute_stats(iphop_filt, pt_filt, cherry_filt, hic_ds)
    log_stats(stats_filt, 'Filtered (excl. consensus non-viral)')

    # ---- Figure layout ----
    # Top row: [A: RefSeq] [B: 3× HiC]
    # Bottom row: [C: overlap] [D: 3× HiC filtered]
    fig = plt.figure(figsize=(4.0 + 4.0 * n_hic, 10.0))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[1, n_hic],
                           hspace=0.35, wspace=0.25,
                           left=0.06, right=0.97, top=0.90, bottom=0.06)

    # Panel A: RefSeq
    ax_a = fig.add_subplot(gs[0, 0])
    if has_refseq:
        stats_refseq = stats_all[stats_all['dataset'] == 'refseq']
        plot_tool_comparison([ax_a], stats_refseq, ['refseq'])
    else:
        ax_a.text(0.5, 0.5, 'No RefSeq data', transform=ax_a.transAxes,
                  ha='center', va='center', fontsize=FONT_SIZES['fallback_msg'])
        ax_a.set_axis_off()
    ax_a.set_ylabel('Fraction of sequences')
    ax_a.text(-0.15, 1.06, 'A', transform=ax_a.transAxes,
              fontsize=FONT_SIZES['panel_letter'], fontweight='bold',
              color=COLORS['text'], va='top')

    # Panel B: HiC unfiltered
    gs_b = gs[0, 1].subgridspec(1, n_hic, wspace=0.08)
    axes_b = [fig.add_subplot(gs_b[0, i]) for i in range(n_hic)]
    stats_hic = stats_all[stats_all['dataset'].isin(hic_ds)]
    plot_tool_comparison(axes_b, stats_hic, hic_ds)
    for i, ax in enumerate(axes_b):
        if i > 0:
            ax.set_yticklabels([])
    axes_b[0].set_ylabel('Fraction of sequences')
    axes_b[0].text(-0.25, 1.06, 'B', transform=axes_b[0].transAxes,
                   fontsize=FONT_SIZES['panel_letter'], fontweight='bold',
                   color=COLORS['text'], va='top')

    # Panel C: Non-viral overlap
    ax_c = fig.add_subplot(gs[1, 0])
    plot_nonviral_overlap(ax_c, hic_ids, genomad_viral_ids,
                          checkv_nonviral_ids, pt_bacterial_ids)
    ax_c.text(-0.15, 1.06, 'C', transform=ax_c.transAxes,
              fontsize=FONT_SIZES['panel_letter'], fontweight='bold',
              color=COLORS['text'], va='top')

    # Panel D: HiC filtered (consensus non-viral removed)
    gs_d = gs[1, 1].subgridspec(1, n_hic, wspace=0.08)
    axes_d = [fig.add_subplot(gs_d[0, i]) for i in range(n_hic)]
    plot_tool_comparison(axes_d, stats_filt, hic_ds)
    for i, ax in enumerate(axes_d):
        if i > 0:
            ax.set_yticklabels([])
    axes_d[0].set_ylabel('Fraction of sequences\n(excl. consensus non-viral)')
    axes_d[0].text(-0.25, 1.06, 'D', transform=axes_d[0].transAxes,
                   fontsize=FONT_SIZES['panel_letter'], fontweight='bold',
                   color=COLORS['text'], va='top')

    # Legend
    legend_elements = [Patch(facecolor=RANK_COLORS[r], label=r)
                       for r in LEVEL_NAMES]
    legend_elements += [
        Patch(facecolor='gray', alpha=1.0,  label='Correct'),
        Patch(facecolor='gray', alpha=0.35, label='Incorrect'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=len(legend_elements), frameon=False,
               fontsize=FONT_SIZES['legend'],
               bbox_to_anchor=(0.5, 0.98))

    _suptitle(fig, 'Host prediction accuracy by taxonomic rank', y=1.02)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"Figure saved: {out_path}")
    root = os.path.splitext(out_path)[0]
    for ext in ('.pdf', '.svg'):
        path = root + ext
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate tool comparison figure '
                    '(PhageTransformer vs iPHoP vs CHERRY)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory (contains calibration.json '
                             'and checkpoints/)')
    parser.add_argument('--compare_dir', type=str, required=True,
                        help='Directory with tool comparison CSVs '
                             '(target_lineages.csv, iPHoP, CHERRY, '
                             'geNomad, CheckV results)')
    parser.add_argument('--testset_dir', type=str, required=True,
                        help='Directory with combined.fna.gz and '
                             'combined_lineage.csv')
    parser.add_argument('--output', '-o', type=str, default='comparison.png',
                        help='Output filename')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed score threshold for PT predictions '
                             '(overrides --fdr)')
    parser.add_argument('--fdr', type=float, default=None,
                        help='FDR level for PT threshold from calibration '
                             '(default: 0.2 if --threshold not set)')
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

    # ---- load model ------------------------------------------------------
    model, calib = load_model_and_calibration(
        args.model_dir, args.checkpoint, device)
    temperature = calib['temperature']
    hosts = np.array(calib['hosts'])
    patch_nt_len = calib['model_config']['patch_nt_len']

    # ---- resolve threshold -------------------------------------------------
    if args.threshold is not None:
        pt_threshold = args.threshold
        logger.info(f"PT threshold (fixed): {pt_threshold:.4f}")
    else:
        fdr = args.fdr if args.fdr is not None else 0.2
        fdr_key = f"fdr_{int(fdr * 100):02d}"
        fdr_thresholds = calib.get('fdr_thresholds', {})
        if fdr_key in fdr_thresholds:
            pt_threshold = fdr_thresholds[fdr_key]
            logger.info(f"PT threshold (FDR {fdr*100:.0f}%): "
                        f"{pt_threshold:.4f}")
        else:
            pt_threshold = calib.get('threshold', 0.5)
            logger.warning(f"FDR key '{fdr_key}' not in calibration, "
                          f"using default threshold {pt_threshold:.4f}")

    # ---- load test data and run inference --------------------------------
    tokenizer = CodonTokenizer()
    eval_stride = args.eval_stride or patch_nt_len // 2

    logger.info("Loading external test set ...")
    cmp_ids, cmp_seqs, cmp_datasets, cmp_labels = load_test_with_ids(
        args.testset_dir, hosts)
    logger.info(f"  {len(cmp_seqs)} sequences")

    pt_predictions = predict_test_for_comparison(
        model, cmp_seqs, cmp_ids, hosts, temperature,
        tokenizer, device,
        patch_nt_len=patch_nt_len, max_patches=args.max_patches,
        eval_stride=eval_stride,
        batch_size=args.batch_size, num_workers=args.num_workers,
        threshold=pt_threshold)
    logger.info(f"PT predictions: {len(pt_predictions)} rows for "
                f"{pt_predictions['sequence_id'].nunique()} sequences")

    # ---- generate figure -------------------------------------------------
    make_comparison_figure(
        compare_dir=args.compare_dir,
        pt_predictions=pt_predictions,
        pt_threshold=pt_threshold,
        out_path=args.output,
        dpi=args.dpi)

    logger.info("Done.")


if __name__ == '__main__':
    main()
