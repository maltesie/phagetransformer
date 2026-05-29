#!/usr/bin/env python3
"""Per-layer attention projection for the hierarchical DNA classifier.

Extracts individual attention layers from a trained model and plots
each as a separate heatmap projected onto a single phage genome, with
predicted or externally-provided coding regions overlaid.

The base layer (always plotted) is the main cross-frame attention from
the first transformer.  Additional layers are shown as separate
subplots via flags or ``--all``.

Usage:
    python attention.py -i phage.fasta --model_dir ./models/HierDNA
    python attention.py -i phage.gb --model_dir ./models/HierDNA --all
    python attention.py -i phage.fasta --model_dir ./models/HierDNA \
        --branch_cross_attention --encoder_pooling_attention
    python attention.py -i phage.gb --model_dir ./models/HierDNA \
        --protein_annotations proteins.tsv --top_n 5 --all

Supported input formats:
    FASTA  (.fasta, .fa, .fna, .fasta.gz) — CDS predicted via pyrodigal
    GenBank (.gb, .gbk, .genbank, .gbff, + .gz) — CDS extracted from file

Expected model_dir contents:
    calibration.json   — temperature, thresholds, model config, host list
    checkpoints/       — model checkpoint(s)
"""

import argparse
import copy
import csv as csv_mod
import logging
import os
from typing import Dict, List

import numpy as np
import torch

from phagetransformer.model import CodonTokenizer
from eval_utils import (
    COLORS, FONT_SIZES, FIG_WIDTH, FIG_HEIGHT_ROW,
    setup_style, _save_figure, _output_path,
    enable_presentation_mode, _add_panel_letters,
)
from phagetransformer.predict import (
    read_fasta,
    load_model_and_calibration,
    tile_sequence,
    tokenize_patches,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_LABELS = ['+1', '+2', '+3', '-1', '-2', '-3']


# ---------------------------------------------------------------------------
# Gene prediction / external annotations
# ---------------------------------------------------------------------------

def predict_genes(seq: str) -> list:
    import pyrodigal
    finder = pyrodigal.GeneFinder(meta=True)
    genes_obj = finder.find_genes(seq.encode())

    genes = []
    for g in genes_obj:
        genes.append({
            'begin': g.begin,
            'end': g.end,
            'strand': g.strand,
        })
    return genes


def read_genbank(path: str) -> List[dict]:
    from Bio import SeqIO
    import gzip

    opener = gzip.open if path.endswith('.gz') else open
    records = []
    with opener(path, 'rt') as fh:
        for rec in SeqIO.parse(fh, 'genbank'):
            seq = str(rec.seq).upper()
            genes = []

            for feat in rec.features:
                if feat.type != 'CDS':
                    continue

                begin = int(feat.location.start) + 1   # BioPython is 0-based
                end = int(feat.location.end)
                strand = 1 if feat.location.strand == 1 else -1

                # Extract annotation and category from qualifiers
                annot = feat.qualifiers.get('product', [''])[0]
                gene_name = feat.qualifiers.get('gene', [''])[0]
                category = annot if annot else gene_name

                genes.append({
                    'begin': begin,
                    'end': end,
                    'strand': strand,
                    'strand_str': '+' if strand == 1 else '-',
                    'annot': annot,
                    'category': category,
                })

            records.append({
                'id': rec.id if rec.id != '.' else rec.name,
                'seq': seq,
                'genes': genes,
            })

    return records


_GENBANK_EXTENSIONS = {'.gb', '.gbk', '.genbank', '.gbff'}


def load_protein_annotations(path: str) -> Dict[str, list]:
    annotations = {}
    with open(path) as f:
        reader = csv_mod.DictReader(f, delimiter='\t')
        for row in reader:
            contig = row['contig']
            start = int(row['start'])
            stop = int(row['stop'])
            begin = min(start, stop)
            end = max(start, stop)
            strand_str = row['strand']
            strand = 1 if strand_str == '+' else -1

            annot = row.get('annot', '')
            category = row.get('category', '')

            annotations.setdefault(contig, []).append({
                'begin': begin,
                'end': end,
                'strand': strand,
                'strand_str': strand_str,
                'annot': annot,
                'category': category,
            })

    # frame_idx deferred to when we know seq_len — store raw for now
    return annotations


def assign_frame_indices(genes: list) -> list:
    for g in genes:
        if g['strand'] == 1:
            g['frame_idx'] = (g['begin'] - 1) % 3
        else:
            g['frame_idx'] = 3 + (g['end'] % 3)
    return genes


def _frame_permutation(start_nt: int, patch_nt_len: int) -> list:
    perm = [0] * 6
    for f in range(3):
        bio = (start_nt + f) % 3
        perm[bio] = f
    for bio_fwd in range(3):
        f = perm[bio_fwd]
        o = (patch_nt_len - f) % 3
        perm[3 + bio_fwd] = 3 + o
    return perm


def squeeze_frame_weights(w_patches, patch_starts, patch_nt_len, seq_len,
                          compression_factor):
    weights_per_patch = (patch_nt_len // 3) // compression_factor
    weights_in_seq = (seq_len // 3) // compression_factor
    w_out = np.zeros((weights_in_seq, 6))

    for start_nt, patch_w in zip(patch_starts, w_patches):
        perm = _frame_permutation(start_nt, patch_nt_len)
        aligned = patch_w[:, perm]

        pos = (start_nt // 3) // compression_factor
        end = min(pos + weights_per_patch, weights_in_seq)
        n = end - pos
        w_out[pos:end] = np.maximum(w_out[pos:end], aligned[:n])

    return w_out


# Keep backward-compatible alias used by other scripts
squeeze_weights = squeeze_frame_weights


def squeeze_pool_weights(w_patches, patch_starts, patch_nt_len, seq_len,
                         compression_factor):
    weights_per_patch = (patch_nt_len // 3) // compression_factor
    weights_in_seq = (seq_len // 3) // compression_factor
    w_out = np.zeros(weights_in_seq)

    for start_nt, patch_w in zip(patch_starts, w_patches):
        pos = (start_nt // 3) // compression_factor
        end = min(pos + weights_per_patch, weights_in_seq)
        n = end - pos
        w_out[pos:end] = np.maximum(w_out[pos:end], patch_w[:n])

    return w_out


def squeeze_agg_weights(agg_w, patch_starts, patch_nt_len, seq_len,
                        compression_factor):
    weights_per_patch = (patch_nt_len // 3) // compression_factor
    weights_in_seq = (seq_len // 3) // compression_factor
    w_out = np.zeros(weights_in_seq)

    for start_nt, scalar in zip(patch_starts, agg_w):
        pos = (start_nt // 3) // compression_factor
        end = min(pos + weights_per_patch, weights_in_seq)
        w_out[pos:end] = np.maximum(w_out[pos:end], scalar)

    return w_out


# ---------------------------------------------------------------------------
# Weight extraction — per-layer (no combining)
# ---------------------------------------------------------------------------

def _normalize_patches(w):
    w = w.copy()
    for i in range(w.shape[0]):
        pmax = w[i].max()
        if pmax > 0:
            w[i] /= pmax
    return w


@torch.no_grad()
def extract_layer_weights(model, tokenizer, seq: str, patch_nt_len: int,
                          stride: int, device: torch.device,
                          max_patches: int = 4096) -> dict:
    assert stride % 3 == 0, (
        f"stride must be divisible by 3 for consistent frame alignment "
        f"(got {stride})")

    patches, starts = tile_sequence(seq, patch_nt_len, stride, max_patches)

    # Snap the last patch start to a multiple of 3 so its forward
    # frame permutation matches the stride-aligned patches.
    if len(starts) > 1 and starts[-1] % 3 != 0:
        last_start = (starts[-1] // 3) * 3
        starts[-1] = last_start
        patches[-1] = seq[last_start:last_start + patch_nt_len]
    tokens, counts = tokenize_patches(patches, tokenizer)
    tokens = tokens.to(device, non_blocking=True)
    counts = counts.to(device, non_blocking=True)
    compression_factor = model.patch_encoder.frame_cnn.compression_factor
    seq_len = len(seq)

    all_weights = model.annotate(tokens, counts)
    w = all_weights[0]   # dict for first (only) sequence in the batch

    layers = {}

    # 1. Main cross-frame attention — (n_patches, L', 6) → (positions, 6)
    layers['cross_frame'] = squeeze_frame_weights(
        _normalize_patches(w['frame_w']),
        starts, patch_nt_len, seq_len, compression_factor)

    # 2. Branch cross-frame attention — (n_patches, L', 6) → (positions, 6)
    layers['branch_cross_frame'] = squeeze_frame_weights(
        _normalize_patches(w['branch_frame_w']),
        starts, patch_nt_len, seq_len, compression_factor)

    # 3. Main encoder pooling — (n_patches, L') → (positions,)
    layers['encoder_pooling'] = squeeze_pool_weights(
        w['pool_w'], starts, patch_nt_len, seq_len, compression_factor)

    # 4. Branch pooling — (n_patches, L') → (positions,)
    layers['branch_pooling'] = squeeze_pool_weights(
        w['branch_pool_w'], starts, patch_nt_len, seq_len, compression_factor)

    # 5. Aggregator — (n_patches,) → (positions,)
    layers['aggregator'] = squeeze_agg_weights(
        w['agg_w'], starts, patch_nt_len, seq_len, compression_factor)

    return layers


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plotting — per-layer panels
# ---------------------------------------------------------------------------

# Ordered layer definitions: key, human label, type, gridspec height ratio
LAYER_DEFS = [
    ('cross_frame',        'Main cross-frame attention',    'frame', 3),
    ('branch_cross_frame', 'Branch cross-frame attention',  'frame', 3),
    ('encoder_pooling',    'Main position attention',       'pool',  1),
    ('branch_pooling',     'Branch position attention',     'pool',  1),
    ('aggregator',         'Aggregator patch attention',    'pool',  1),
]

# Map from CLI flag name to layer key
FLAG_TO_LAYER = {
    'branch_cross_attention':   'branch_cross_frame',
    'encoder_pooling_attention': 'encoder_pooling',
    'branch_pooling_attention':  'branch_pooling',
    'aggregator_attention':      'aggregator',
}


def _overlay_genes_frame(ax, genes, nt_to_pos, gene_colors):
    import matplotlib.patches as mpatches
    for gene in genes:
        x_start = nt_to_pos(gene['begin'])
        x_end = nt_to_pos(gene['end'])
        fi = gene['frame_idx']
        bar_height = 0.4
        rect = mpatches.FancyBboxPatch(
            (x_start, fi - bar_height / 2),
            max(x_end - x_start, 0.5), bar_height,
            boxstyle='square',
            facecolor='none', edgecolor=gene_colors[fi],
            linewidth=1.0, alpha=0.9, zorder=3)
        ax.add_patch(rect)


def _overlay_genes_pool(ax, genes, nt_to_pos):
    import matplotlib.patches as mpatches
    for gene in genes:
        x_start = nt_to_pos(gene['begin'])
        x_end = nt_to_pos(gene['end'])
        color = COLORS['gene_fwd'] if gene['strand'] == 1 else COLORS['gene_rev']
        y = -0.15 if gene['strand'] == 1 else 0.15
        rect = mpatches.FancyBboxPatch(
            (x_start, y), max(x_end - x_start, 0.5), 0.3,
            boxstyle='square', facecolor='none',
            edgecolor=color, linewidth=0.8, alpha=0.8,
            zorder=3, clip_on=False)
        ax.add_patch(rect)


def _draw_gene_strip(ax, genes, nt_to_pos, n_pos):
    import matplotlib.patches as mpatches
    ax.set_xlim(0, n_pos)
    ax.set_ylim(0, 1)
    for gene in genes:
        x_start = nt_to_pos(gene['begin'])
        x_end = nt_to_pos(gene['end'])
        color = COLORS['gene_fwd'] if gene['strand'] == 1 else COLORS['gene_rev']
        rect = mpatches.Rectangle(
            (x_start, 0), max(x_end - x_start, 0.5), 1,
            facecolor=color, edgecolor='none', alpha=0.7, zorder=2)
        ax.add_patch(rect)
        ax.axvline(x_start, color='black', lw=0.5, alpha=0.7, zorder=3)
        ax.axvline(x_end, color='black', lw=0.5, alpha=0.7, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _label_top_genes(ax, genes, w, nt_to_pos, top_n):
    n_pos = w.shape[0]
    scored_genes = []
    for gene in genes:
        p0 = max(0, int(nt_to_pos(gene['begin'])))
        p1 = min(n_pos, int(np.ceil(nt_to_pos(gene['end']))))
        if p1 > p0:
            mean_attn = w[p0:p1, :].max()
        else:
            mean_attn = 0.0
        cat = gene.get('category', '')
        scored_genes.append((mean_attn, gene, cat))
    scored_genes.sort(key=lambda x: x[0], reverse=True)
    top_genes = [(g, cat) for _, g, cat in scored_genes[:top_n] if cat]

    top_genes.sort(key=lambda x: (x[0]['frame_idx'], x[0]['begin']))
    frame_counters = {}

    for gene, cat in top_genes:
        x_mid = (nt_to_pos(gene['begin']) + nt_to_pos(gene['end'])) / 2
        fi = gene['frame_idx']
        count = frame_counters.get(fi, 0)
        place_above = (count % 2 == 0)
        frame_counters[fi] = count + 1
        y_label = fi - 0.45 if place_above else fi + 0.45
        va = 'bottom' if place_above else 'top'
        label_text = cat if len(cat) <= 25 else cat[:22] + '…'
        ax.annotate(
            label_text,
            xy=(x_mid, fi), xytext=(x_mid, y_label),
            fontsize=FONT_SIZES['overlay_small'], ha='center', va=va,
            color='white', fontweight='bold', zorder=5,
            bbox=dict(boxstyle='round,pad=0.15',
                      facecolor=COLORS['dark_overlay'],
                      alpha=0.8, edgecolor='none'),
            arrowprops=dict(arrowstyle='-', color='white',
                            lw=0.5, alpha=0.6),
        )


def _label_top_genes_split(row_axes, genes, w, nt_to_pos, top_n):
    n_pos = w.shape[0]
    scored_genes = []
    for gene in genes:
        p0 = max(0, int(nt_to_pos(gene['begin'])))
        p1 = min(n_pos, int(np.ceil(nt_to_pos(gene['end']))))
        if p1 > p0:
            mean_attn = w[p0:p1, :].max()
        else:
            mean_attn = 0.0
        cat = gene.get('category', '')
        scored_genes.append((mean_attn, gene, cat))
    scored_genes.sort(key=lambda x: x[0], reverse=True)
    top_genes = [(g, cat) for _, g, cat in scored_genes[:top_n] if cat]

    top_genes.sort(key=lambda x: (x[0]['frame_idx'], x[0]['begin']))
    frame_counters = {}

    for gene, cat in top_genes:
        fi = gene['frame_idx']
        ax = row_axes[fi]
        x_mid = (nt_to_pos(gene['begin']) + nt_to_pos(gene['end'])) / 2
        count = frame_counters.get(fi, 0)
        place_above = (count % 2 == 0)
        frame_counters[fi] = count + 1
        y_label = -0.45 if place_above else 0.45
        va = 'bottom' if place_above else 'top'
        label_text = cat if len(cat) <= 25 else cat[:22] + '…'
        ax.annotate(
            label_text,
            xy=(x_mid, 0), xytext=(x_mid, y_label),
            fontsize=FONT_SIZES['overlay_small'], ha='center', va=va,
            color='white', fontweight='bold', zorder=5,
            clip_on=False,
            bbox=dict(boxstyle='round,pad=0.15',
                      facecolor=COLORS['dark_overlay'],
                      alpha=0.8, edgecolor='none'),
            arrowprops=dict(arrowstyle='-', color='white',
                            lw=0.5, alpha=0.6),
        )


def plot_layer_panels(layers: dict, active_keys: list,
                      seq_id: str, seq_len: int, genes: list,
                      out_path: str, compression_factor: int,
                      normalize: bool = False,
                      dpi: int = 150, top_n: int = 0):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    setup_style()

    # Build ordered list of (key, label, type, height_ratio, array)
    panels = []
    for key, label, ltype, hratio in LAYER_DEFS:
        if key in active_keys:
            panels.append((key, label, ltype, hratio, layers[key]))

    if not panels:
        logger.warning("No layers to plot.")
        return

    n_panels = len(panels)
    height_ratios = [p[3] for p in panels]
    total_units = sum(height_ratios)
    fig_height = max(FIG_HEIGHT_ROW, 1.0 * total_units + 1.5)

    fig = plt.figure(figsize=(FIG_WIDTH, fig_height))
    gs = GridSpec(n_panels, 1, figure=fig, height_ratios=height_ratios,
                  hspace=0.45)

    gene_colors = {
        0: COLORS['gene_fwd'], 1: COLORS['gene_fwd'], 2: COLORS['gene_fwd'],
        3: COLORS['gene_rev'], 4: COLORS['gene_rev'], 5: COLORS['gene_rev'],
    }

    def nt_to_pos(nt_coord):
        return (nt_coord - 1) // 3 // compression_factor

    last_im = {}  # track last imshow per colorbar sharing group

    for pi, (key, label, ltype, _hratio, w_raw) in enumerate(panels):

        if normalize:
            if ltype == 'frame':
                wmax = w_raw.max(axis=1)
                w = w_raw / wmax[:, None] if (wmax > 0).all() else w_raw.copy()
            else:
                wmax = w_raw.max()
                w = w_raw / wmax if wmax > 0 else w_raw.copy()
            
        else:
            w = w_raw

        if ltype == 'frame':
            # 6 separate row axes with gene strips, gaps between frames
            sub_gs = gs[pi].subgridspec(6, 1, hspace=0.35)
            row_axes = []
            strip_axes = []
            n_pos = w.shape[0]
            for fi in range(6):
                inner_gs = sub_gs[fi].subgridspec(
                    2, 1, height_ratios=[0.15, 1], hspace=0.05)
                ax_strip = fig.add_subplot(inner_gs[0])
                ax_row = fig.add_subplot(inner_gs[1])
                strip_axes.append(ax_strip)
                row_axes.append(ax_row)

                # Gene strip filtered to this frame
                frame_genes = [g for g in genes if g['frame_idx'] == fi]
                _draw_gene_strip(ax_strip, frame_genes, nt_to_pos, n_pos)

                im = ax_row.imshow(w[:, fi].reshape(1, -1), aspect='auto',
                                   cmap='viridis', interpolation='none',
                                   extent=[0, n_pos, 0.5, -0.5])

                ax_row.set_ylim(0.5, -0.5)
                ax_row.set_yticks([0])
                ax_row.set_yticklabels([FRAME_LABELS[fi]],
                                       fontsize=FONT_SIZES['label'])
                if fi < 5:
                    ax_row.set_xticks([])

            title_ax = strip_axes[0]
            ax = row_axes[-1]
            last_im['frame'] = im

            if top_n > 0 and genes:
                _label_top_genes_split(row_axes, genes, w, nt_to_pos, top_n)

        else:
            # 1-D pooling heatmap with gene strip above
            sub_gs = gs[pi].subgridspec(2, 1, height_ratios=[0.15, 1],
                                         hspace=0.05)
            title_ax = fig.add_subplot(sub_gs[0])
            ax = fig.add_subplot(sub_gs[1])

            n_pos = w.shape[0]
            _draw_gene_strip(title_ax, genes, nt_to_pos, n_pos)

            im = ax.imshow(w.reshape(1, -1), aspect='auto', cmap='magma',
                           interpolation='none',
                           extent=[0, n_pos, 0.5, -0.5])
            ax.set_yticks([])
            ax.set_ylim(0.5, -0.5)
            last_im['pool'] = im

        title_ax.set_title(label, loc='left', fontsize=FONT_SIZES['legend'],
                           pad=4)

        # x-axis in kilobases
        n_ticks = min(8, max(2, n_pos // 20))
        tick_positions = np.linspace(0, n_pos, n_ticks)
        tick_labels_kb = [f'{nt / 1000:.1f}' for nt in
                          np.linspace(0, seq_len, n_ticks)]
        ax.set_xticks(tick_positions)
        if pi == n_panels - 1:
            ax.set_xticklabels(tick_labels_kb,
                               fontsize=FONT_SIZES['label'])
            ax.set_xlabel('Genome position (kb)',
                          fontsize=FONT_SIZES['legend'])
        else:
            ax.set_xticklabels([])

    # Colorbars — one per visual type present
    fig.subplots_adjust(left=0.06, right=0.84)
    cbar_width = 0.012
    cbar_kw = dict(labelsize=FONT_SIZES['label'])
    if 'frame' in last_im and 'pool' in last_im:
        ax_cb1 = fig.add_axes([0.86, 0.52, cbar_width, 0.38])
        cb1 = fig.colorbar(last_im['frame'], cax=ax_cb1,
                           label='Frame attention')
        cb1.set_label('Frame attention', fontsize=FONT_SIZES['legend'])
        cb1.ax.tick_params(**cbar_kw)
        ax_cb2 = fig.add_axes([0.86, 0.08, cbar_width, 0.38])
        cb2 = fig.colorbar(last_im['pool'], cax=ax_cb2,
                           label='Pooling weight')
        cb2.set_label('Pooling weight', fontsize=FONT_SIZES['legend'])
        cb2.ax.tick_params(**cbar_kw)
    elif 'frame' in last_im:
        ax_cb = fig.add_axes([0.86, 0.15, cbar_width, 0.7])
        cb = fig.colorbar(last_im['frame'], cax=ax_cb,
                          label='Frame attention')
        cb.set_label('Frame attention', fontsize=FONT_SIZES['legend'])
        cb.ax.tick_params(**cbar_kw)
    elif 'pool' in last_im:
        ax_cb = fig.add_axes([0.86, 0.15, cbar_width, 0.7])
        cb = fig.colorbar(last_im['pool'], cax=ax_cb,
                          label='Pooling weight')
        cb.set_label('Pooling weight', fontsize=FONT_SIZES['legend'])
        cb.ax.tick_params(**cbar_kw)

    # Gene legend
    legend_patches = [
        mpatches.Patch(facecolor='none', edgecolor=COLORS['gene_fwd'],
                       linewidth=1.5, label='Forward genes'),
        mpatches.Patch(facecolor='none', edgecolor=COLORS['gene_rev'],
                       linewidth=1.5, label='Reverse genes'),
    ]
    fig.legend(handles=legend_patches, loc='lower right',
               bbox_to_anchor=(0.96, 0.92),
               fontsize=FONT_SIZES['label'],
               frameon=True,
               framealpha=0.9, edgecolor=COLORS['grid'])

    # Title
    sid_label = seq_id if len(seq_id) <= 50 else seq_id[:47] + '...'
    n_fwd = sum(1 for g in genes if g['strand'] == 1)
    n_rev = len(genes) - n_fwd
    title = (f'{sid_label} — {n_panels} attention layer(s), '
             f'{len(genes)} genes ({n_fwd}→ {n_rev}←)')
    fig.suptitle(title, fontsize=FONT_SIZES['title'],
                 fontweight='bold', color=COLORS['text'], y=0.99)

    out_path = _output_path(out_path)
    _save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Quantification figure
# ---------------------------------------------------------------------------

def _gene_position_ranges(genes, compression_factor, n_pos):
    nt_per_pos = 3 * compression_factor
    ranges = []
    for g in genes:
        p0 = max(0, int((g['begin'] - 1) / nt_per_pos))
        p1 = min(n_pos, int(np.ceil(g['end'] / nt_per_pos)))
        if p1 > p0:
            ranges.append((g, p0, p1))
    return ranges


def _metagene_profiles(w_pool, genes, compression_factor, window=30):
    n_pos = len(w_pool)
    nt_per_pos = 3 * compression_factor
    start_profiles, end_profiles = [], []
    for g in genes:
        if g['strand'] == 1:
            p_start = int((g['begin'] - 1) / nt_per_pos)
            p_end = int(g['end'] / nt_per_pos)
        else:
            p_start = int(g['end'] / nt_per_pos)
            p_end = int((g['begin'] - 1) / nt_per_pos)

        lo_s, hi_s = p_start - window, p_start + window
        lo_e, hi_e = p_end - window, p_end + window

        if lo_s >= 0 and hi_s < n_pos:
            prof = w_pool[lo_s:hi_s]
            start_profiles.append(prof if g['strand'] == 1 else prof[::-1])
        if lo_e >= 0 and hi_e < n_pos:
            prof = w_pool[lo_e:hi_e]
            end_profiles.append(prof if g['strand'] == 1 else prof[::-1])

    return start_profiles, end_profiles


def _plot_metagene_panel(ax, w_pool, genes, compression_factor, window,
                         ylabel, title):
    start_p, end_p = _metagene_profiles(
        w_pool, genes, compression_factor, window)
    x_axis = np.arange(-window, window)
    if start_p:
        ax.plot(x_axis, np.mean(start_p, axis=0),
                color=COLORS['gene_fwd'], lw=1.5, label='Gene start')
    if end_p:
        ax.plot(x_axis, np.mean(end_p, axis=0),
                color=COLORS['gene_rev'], lw=1.5, label='Gene end')
    if start_p or end_p:
        ax.axvline(0, color=COLORS['grid'], ls='--', lw=0.8)
        ax.legend(fontsize=FONT_SIZES['legend'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.set_title(title, loc='left', fontsize=FONT_SIZES['label'])
    ax.tick_params(labelsize=FONT_SIZES['label'])


def plot_quantification(layers, genes, seq_len, compression_factor,
                        out_path, dpi=150):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    if not genes:
        logger.warning("No genes \u2014 skipping quantification figure")
        return

    has_branch_pool = 'branch_pooling' in layers
    has_main_pool = 'encoder_pooling' in layers
    window = 30

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT_ROW))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # --- Panel A: Sense vs antisense frame enrichment --------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    # Paired frames: 0↔3, 1↔4, 2↔5
    def _paired_frame(fi):
        return fi + 3 if fi < 3 else fi - 3

    layer_configs = [('Main', 'cross_frame', COLORS['primary']),
                     ('Branch', 'branch_cross_frame', COLORS['secondary'])]
    width = 0.35
    x_base = np.arange(2)  # sense, antisense
    legend_handles = []

    for li, (tag, key, color) in enumerate(layer_configs):
        if key not in layers:
            continue
        w = layers[key]
        n_pos = w.shape[0]
        gene_ranges = _gene_position_ranges(genes, compression_factor, n_pos)

        sense_vals, anti_vals = [], []
        for g, p0, p1 in gene_ranges:
            fi = g['frame_idx']
            region = w[p0:p1, :]
            all_mean = region.mean()
            if all_mean <= 0:
                continue
            sense_mean = region[:, fi].mean()
            anti_mean = region[:, _paired_frame(fi)].mean()
            if sense_mean > 0:
                sense_vals.append(np.log2(sense_mean / all_mean))
            if anti_mean > 0:
                anti_vals.append(np.log2(anti_mean / all_mean))

        offset = (li - 0.5) * width
        box_data = [sense_vals or [0.0], anti_vals or [0.0]]
        bp = ax_a.boxplot(box_data,
                          positions=x_base + offset,
                          widths=width * 0.85,
                          patch_artist=True, showfliers=False,
                          manage_ticks=False)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for element in ('whiskers', 'caps', 'medians'):
            for line in bp[element]:
                line.set_color(COLORS['text'])
                line.set_linewidth(0.8)
        legend_handles.append(mpatches.Patch(facecolor=color, alpha=0.5,
                                             label=tag))

    ax_a.set_xticks(x_base)
    ax_a.set_xticklabels(['Sense', 'Antisense'], fontsize=FONT_SIZES['label'])
    ax_a.axhline(0, color=COLORS['grid'], ls='--', lw=0.8)
    if legend_handles:
        ax_a.legend(handles=legend_handles, fontsize=FONT_SIZES['legend'])

    ax_a.set_ylabel('log\u2082 enrichment', fontsize=FONT_SIZES['label'])
    ax_a.set_title('Coding frame enrichment', loc='left',
                   fontsize=FONT_SIZES['label'])
    ax_a.tick_params(labelsize=FONT_SIZES['label'])

    # --- Panel B: Metagene boundary (main + branch pooling) ------------------
    ax_b = fig.add_subplot(gs[0, 1])
    x_label = (f'Compressed codon position\nrelative to boundary '
               f'({compression_factor}\u00d7)')

    pool_layers = [('Main', 'encoder_pooling', COLORS['primary']),
                   ('Branch', 'branch_pooling', COLORS['secondary'])]

    x_axis = np.arange(-window, window)
    legend_lines = []

    for tag, key, color in pool_layers:
        if key not in layers:
            continue
        start_p, end_p = _metagene_profiles(
            layers[key], genes, compression_factor, window)
        if start_p:
            line, = ax_b.plot(x_axis, np.mean(start_p, axis=0),
                              color=color, ls='-', lw=1.5,
                              label=f'{tag} — gene start')
            legend_lines.append(line)
        if end_p:
            line, = ax_b.plot(x_axis, np.mean(end_p, axis=0),
                              color=color, ls='--', lw=1.5,
                              label=f'{tag} — gene end')
            legend_lines.append(line)

    if legend_lines:
        ax_b.axvline(0, color=COLORS['grid'], ls='--', lw=0.8)
        ax_b.legend(fontsize=FONT_SIZES['legend'])

    ax_b.set_xlabel(x_label, fontsize=FONT_SIZES['label'])
    ax_b.set_ylabel('Pooling weight', fontsize=FONT_SIZES['label'])
    ax_b.set_title('Metagene boundary profile', loc='left',
                   fontsize=FONT_SIZES['label'])
    ax_b.tick_params(labelsize=FONT_SIZES['label'])

    # --- Panel letters -------------------------------------------------------
    _add_panel_letters([ax_a, ax_b])

    # --- Suptitle and save ---------------------------------------------------
    fig.suptitle('Attention quantification',
                 fontsize=FONT_SIZES['title'],
                 fontweight='bold', color=COLORS['text'], y=1.02)

    base, ext = os.path.splitext(out_path)
    quant_path = _output_path(f'{base}_quantification{ext}')
    _save_figure(fig, quant_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot per-layer attention weights from the hierarchical '
                    'DNA model projected onto a single genome',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input file: FASTA (.fasta/.fa/.fna) or '
                             'GenBank (.gb/.gbk/.genbank/.gbff). '
                             'Gzipped files supported. GenBank input '
                             'provides CDS annotations directly. '
                             'First sequence is used (see --seq_index).')
    parser.add_argument('--seq_index', type=int, default=0,
                        help='0-based index of the sequence to use from '
                             'a multi-record file')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory (contains calibration.json '
                             'and checkpoints/)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--output', '-o', type=str,
                        default='attention_layers.png',
                        help='Output filename for the layer panel plot')
    parser.add_argument('--protein_annotations', type=str, default=None,
                        help='External protein annotation TSV (columns: '
                             'gene, start, stop, strand, contig, annot, '
                             'category). Replaces pyrodigal gene prediction.')
    parser.add_argument('--top_n', type=int, default=3,
                        help='Label the top-n genes by max attention on '
                             'frame heatmaps (requires annotations)')
    parser.add_argument('--normalize_importance', action='store_true',
                        help='Normalize each layer to [0, 1]')

    # Layer selection flags
    layer_grp = parser.add_argument_group('attention layers',
        'Select which layers to plot (base cross-frame attention is '
        'always included). Use --all to enable all layers.')
    layer_grp.add_argument('--all', action='store_true',
                           help='Plot all available attention layers')
    layer_grp.add_argument('--branch_cross_attention', action='store_true',
                           help='Include the frame-stats branch\'s '
                                'cross-frame attention')
    layer_grp.add_argument('--encoder_pooling_attention', action='store_true',
                           help='Include the main encoder path\'s '
                                'query-attention pooling')
    layer_grp.add_argument('--branch_pooling_attention', action='store_true',
                           help='Include the frame-stats branch\'s '
                                'query-attention pooling')
    layer_grp.add_argument('--aggregator_attention', action='store_true',
                           help='Include the aggregator\'s '
                                'patch-level pooling')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--presentation', action='store_true',
                        help='Increase font sizes for presentations')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.presentation:
        enable_presentation_mode()

    # ---- Determine which layers to plot -------------------------------------
    active_keys = ['cross_frame']   # always present
    if args.all:
        active_keys = [key for key, _, _, _ in LAYER_DEFS]
    else:
        for flag_name, layer_key in FLAG_TO_LAYER.items():
            if getattr(args, flag_name, False):
                active_keys.append(layer_key)

    layer_labels = []
    for key, label, _, _ in LAYER_DEFS:
        if key in active_keys:
            layer_labels.append(label)
    logger.info(f"Layers to plot: {', '.join(layer_labels)}")

    # ---- Load model ---------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model, calib = load_model_and_calibration(
        args.model_dir, args.checkpoint, device)

    patch_nt_len = calib['model_config']['patch_nt_len']

    # ---- Read input (single sequence) ---------------------------------------
    logger.info(f"Reading {args.input} …")

    input_base = args.input[:-3] if args.input.endswith('.gz') else args.input
    is_genbank = os.path.splitext(input_base)[1].lower() in _GENBANK_EXTENSIONS

    if is_genbank:
        gb_records = read_genbank(args.input)
        records = [{'id': r['id'], 'seq': r['seq']} for r in gb_records]
        gb_annot = {r['id']: r['genes'] for r in gb_records}
    else:
        records = read_fasta(args.input)
        gb_annot = None

    if args.seq_index >= len(records):
        logger.error(f"--seq_index {args.seq_index} out of range "
                     f"({len(records)} sequences available)")
        return

    rec = records[args.seq_index]
    seq = rec['seq']
    seq_id = rec['id']
    logger.info(f"  Using sequence {args.seq_index}: {seq_id} "
                f"({len(seq):,} nt)")

    # ---- Resolve gene annotations -------------------------------------------
    ext_annot = None
    if args.protein_annotations:
        ext_annot = load_protein_annotations(args.protein_annotations)
        logger.info(f"  Loaded external annotations for "
                    f"{len(ext_annot)} contigs")

    if gb_annot and seq_id in gb_annot:
        genes = copy.deepcopy(gb_annot[seq_id])
    elif ext_annot and seq_id in ext_annot:
        genes = copy.deepcopy(ext_annot[seq_id])
    else:
        if gb_annot:
            logger.warning(f"  {seq_id}: not in GenBank records, "
                           f"falling back to pyrodigal")
        elif ext_annot:
            logger.warning(f"  {seq_id}: not in external annotations, "
                           f"falling back to pyrodigal")
        genes = predict_genes(seq)
    genes = assign_frame_indices(genes)

    top_n = args.top_n if (ext_annot or gb_annot) else 0
    n_fwd = sum(1 for g in genes if g['strand'] == 1)
    n_rev = len(genes) - n_fwd
    logger.info(f"  {len(genes)} genes ({n_fwd}→ {n_rev}←)")

    # ---- Extract all layers -------------------------------------------------
    compression_factor = model.patch_encoder.frame_cnn.compression_factor
    tokenizer = CodonTokenizer()
    stride = int(patch_nt_len * (2 / 3) // 3 * 3)

    logger.info(f"Extracting attention layers "
                f"(stride {stride} nt, {compression_factor}× compression) …")

    layers = extract_layer_weights(
        model, tokenizer, seq, patch_nt_len, stride, device)

    for key in active_keys:
        w = layers[key]
        shape_str = ' × '.join(str(d) for d in w.shape)
        logger.info(f"  {key}: {shape_str}")

    # ---- Plot ---------------------------------------------------------------
    plot_layer_panels(
        layers, active_keys, seq_id, len(seq), genes,
        args.output, compression_factor,
        normalize=args.normalize_importance,
        dpi=150, top_n=top_n,
    )

    plot_quantification(
        layers, genes, len(seq), compression_factor,
        args.output, dpi=150,
    )

    logger.info("Done.")


if __name__ == '__main__':
    main()
