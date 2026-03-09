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
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phagetransformer.model import CodonTokenizer
from phagetransformer.predict import load_model_and_calibration
from phagetransformer.dataset import PatchSequenceDataset, sequence_collate_fn
from eval_utils import (
    COLORS, LEVEL_NAMES, FONT_SIZES,
    setup_style, _suptitle, enable_presentation_mode,
    collect_logits, load_test_with_ids, predict_test_for_comparison,
)

logger = logging.getLogger(__name__)

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

    # ---- filter each tool (keep all predictions, no dedup) -----------------
    iphop = iphop[~iphop['Virus'].isin(excluded_ids)].copy()

    pt = pt_predictions[~pt_predictions['sequence_id'].isin(excluded_ids)].copy()
    # Keep sequences that pass the genus score threshold OR are flagged as
    # bacterial (the model made a confident decision — just not a genus one).
    has_bact_col = 'is_bacterial' in pt.columns
    if has_bact_col:
        pt = pt[(pt['score'] > pt_threshold) | pt['is_bacterial']]
    else:
        pt = pt[pt['score'] > pt_threshold]
    logger.info(f"  PT predictions after FDR threshold ({pt_threshold:.4f}): "
                f"{len(pt)} rows for {pt['sequence_id'].nunique()} sequences"
                + (f" ({pt.drop_duplicates('sequence_id')['is_bacterial'].sum()} bacterial)"
                   if has_bact_col else ''))

    cherry = cherry[~cherry['contig_name'].isin(excluded_ids)].copy()

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

    # ---- merge predictions onto target (one-to-many for all tools) ---------
    true_cols = ['Phage', 'dataset'] + [f'true_{l}' for l in LEVEL_NAMES]
    pred_cols = lambda col: [col] + [f'pred_{l}' for l in LEVEL_NAMES]

    iphop_m  = target[true_cols].merge(
        iphop[pred_cols('Virus')],
        left_on='Phage', right_on='Virus', how='left')
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

    # ---- compute per-dataset, per-rank accuracy (any-match) ----------------
    datasets = [ds for ds in ['cow_fecal', 'human_gut', 'waste_water', 'refseq']
                if ds in target['dataset'].values]

    def _stats(merged_df, tool_name):
        """Compute accuracy treating any matching prediction as correct.

        After the one-to-many merge, each target sequence may appear
        multiple times (one row per prediction).  A sequence counts as
        'has_pred' if any row has a non-null prediction, and 'correct'
        if any row's prediction matches the true label.
        """
        rows = []
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

    # Panel A: all sequences, no bacterial sentinel — pure genus predictions
    stats_all = pd.concat([
        _stats(iphop_m,  'iPHoP'),
        _stats(pt_m,     'PT'),
        _stats(cherry_m, 'CHERRY'),
    ], ignore_index=True)

    # Panel B: exclude sequences flagged as bacterial by PT from ALL tools
    if has_bact_col:
        bact_ids = set(
            pt_m.loc[pt_m['is_bacterial'], 'Phage'].unique())
        logger.info(f"  Panel B: excluding {len(bact_ids)} bacterial-flagged "
                    f"sequences from all tools")
        iphop_nb  = iphop_m[~iphop_m['Phage'].isin(bact_ids)]
        pt_nb     = pt_m[~pt_m['Phage'].isin(bact_ids)]
        cherry_nb = cherry_m[~cherry_m['Phage'].isin(bact_ids)]

        stats_nonbact = pd.concat([
            _stats(iphop_nb,  'iPHoP'),
            _stats(pt_nb,     'PT'),
            _stats(cherry_nb, 'CHERRY'),
        ], ignore_index=True)



    return stats_all, stats_nonbact, datasets


def plot_tool_comparison(axes, stats: pd.DataFrame, datasets: List[str]):
    """Grouped bar chart comparing tool accuracy across datasets.

    One subplot per dataset, with grouped bars for each tool × taxonomic rank.
    Solid bars = correct, translucent extensions = incorrect predictions.
    """
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

    axes[0].set_ylabel('Fraction of sequences')



def make_comparison_figure(compare_dir: str, pt_predictions: pd.DataFrame,
                           pt_threshold: float,
                           out_path: str = 'comparison.png',
                           dpi: int = 200):
    """Figure 5: PhageTransformer vs. iPHoP vs. CHERRY across test datasets.

    Two rows of panels:
      Row A: All sequences — PT treated like other tools (genus predictions only).
      Row B: Non-bacterial sequences — sequences flagged as bacterial by PT
             are excluded from all tools.
    """
    from matplotlib.patches import Patch
    setup_style()

    stats_all, stats_nonbact, datasets = load_comparison_data(
        compare_dir, pt_predictions, pt_threshold)
    n_ds = len(datasets)
    has_two_rows = stats_nonbact is not None

    n_rows = 2 if has_two_rows else 1
    fig, axes = plt.subplots(n_rows, n_ds,
                             figsize=(4.0 * n_ds, 5.0 * n_rows),
                             sharey=True, squeeze=False)

    # Row A: all sequences
    plot_tool_comparison(axes[0], stats_all, datasets)
    axes[0][0].set_ylabel('All sequences\nFraction of sequences')

    # Row B: non-bacterial
    if has_two_rows:
        plot_tool_comparison(axes[1], stats_nonbact, datasets)
        axes[1][0].set_ylabel('Excl. bacterial\nFraction of sequences')

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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate tool comparison figure '
                    '(PhageTransformer vs iPHoP vs CHERRY)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory (contains calibration.json and checkpoints/)')
    parser.add_argument('--compare_dir', type=str, required=True,
                        help='Directory with tool comparison CSVs '
                             '(target_lineages.csv, iPHoP, CHERRY predictions)')
    parser.add_argument('--testset_dir', type=str, required=True,
                        help='Directory with combined.fna.gz and combined_lineage.csv')
    parser.add_argument('--output', '-o', type=str, default='comparison.png',
                        help='Output filename')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--fdr', type=float, default=0.2,
                        help='FDR level for PT threshold')
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

    # ---- resolve FDR threshold -------------------------------------------
    fdr_key = f"fdr_{int(args.fdr * 100):02d}"
    fdr_thresholds = calib.get('fdr_thresholds', {})
    if fdr_key in fdr_thresholds:
        pt_threshold = fdr_thresholds[fdr_key]
        logger.info(f"PT threshold (FDR {args.fdr*100:.0f}%): {pt_threshold:.4f}")
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
