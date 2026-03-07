#!/usr/bin/env python3
"""Genome annotation script for the hierarchical DNA classifier.

Extracts hierarchical importance weights (cross-frame attention ×
position pooling × aggregator pooling) from a trained model and plots
heatmaps with predicted or externally-provided coding regions overlaid.

Usage:
    python annotate.py --input phages.fasta --run_dir ./models/HierDNA
    python annotate.py --input phage.gb --run_dir ./models/HierDNA
    python annotate.py --input phages.fasta --run_dir ./models/HierDNA \
        --first_n 20 --output importance.png
    python annotate.py --input phages.fasta --run_dir ./models/HierDNA \
        --protein_annotations proteins.tsv --top_n 5

Supported input formats:
    FASTA  (.fasta, .fa, .fna, .fasta.gz) — CDS predicted via pyrodigal
    GenBank (.gb, .gbk, .genbank, .gbff, + .gz) — CDS extracted from file

Expected run_dir contents:
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

from .model import CodonTokenizer
from .predict import (
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
    """Run pyrodigal (metagenomic mode) on a nucleotide sequence.

    Returns list of dicts with keys:
        begin, end  : 1-based genomic coordinates
        strand      : +1 or -1
        frame_idx   : 0–5 matching the model's frame convention
                      (0=+1, 1=+2, 2=+3, 3=-1, 4=-2, 5=-3)
    """
    import pyrodigal
    finder = pyrodigal.GeneFinder(meta=True)
    genes_obj = finder.find_genes(seq.encode())
    seq_len = len(seq)

    genes = []
    for g in genes_obj:
        if g.strand == 1:
            frame_idx = (g.begin - 1) % 3       # 0, 1, 2
        else:
            frame_idx = 3 + (seq_len - g.end) % 3   # 3, 4, 5
        genes.append({
            'begin': g.begin,
            'end': g.end,
            'strand': g.strand,
            'frame_idx': frame_idx,
        })
    return genes


def read_genbank(path: str) -> List[dict]:
    """Read a GenBank file and extract sequences with their CDS annotations.

    Returns list of dicts with keys:
        id   : record identifier (LOCUS name or accession)
        seq  : nucleotide sequence as uppercase string
        genes: list of gene dicts with keys:
               begin, end, strand, frame_idx, annot, category
    """
    from Bio import SeqIO
    import gzip

    opener = gzip.open if path.endswith('.gz') else open
    records = []
    with opener(path, 'rt') as fh:
        for rec in SeqIO.parse(fh, 'genbank'):
            seq = str(rec.seq).upper()
            seq_len = len(seq)
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

                if strand == 1:
                    frame_idx = (begin - 1) % 3
                else:
                    frame_idx = 3 + (seq_len - end) % 3

                genes.append({
                    'begin': begin,
                    'end': end,
                    'strand': strand,
                    'strand_str': '+' if strand == 1 else '-',
                    'frame_idx': frame_idx,
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
    """Load external protein annotations from a TSV file.

    Expected columns: gene, start, stop, strand, contig, annot, category.

    Returns dict mapping contig ID -> list of gene dicts with keys:
        begin, end, strand, frame_idx, annot, category
    """
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


def assign_frame_indices(genes: list, seq_len: int) -> list:
    """Assign frame_idx (0–5) to gene dicts based on coordinates and seq length."""
    for g in genes:
        if g['strand'] == 1:
            g['frame_idx'] = (g['begin'] - 1) % 3
        else:
            g['frame_idx'] = 3 + (seq_len - g['end']) % 3
    return genes


def squeeze_weights(w, patch_nt_len:int, stride: int, seq_len: int, compression_factor: int):
    weights_per_patch = (patch_nt_len // 3) // compression_factor
    weights_in_seq = (seq_len // 3) // compression_factor
    compressed_stride = (stride // 3) // compression_factor
    w_out = np.zeros((weights_in_seq, 6))
    for i, j in zip(range(0, len(w)-weights_per_patch, compressed_stride), 
                    range(0, len(w)-weights_per_patch, weights_per_patch)):
        patch = w[j:j+weights_per_patch]
        merged_patch = w_out[i:i+weights_per_patch]
        w_out[i:i+weights_per_patch] = np.maximum(merged_patch, patch)
    last_patch = w[len(w)-weights_per_patch:]
    last_merged = w_out[weights_in_seq-weights_per_patch:]
    w_out[weights_in_seq-weights_per_patch:] = np.maximum(last_merged, last_patch)
    return w_out
        

# ---------------------------------------------------------------------------
# Weight extraction (single tiling for all levels)
# ---------------------------------------------------------------------------

def combine_weights(weights: dict,
                    branch_cross_attention: bool = False,
                    branch_pooling_attention: bool = False,
                    encoder_pooling_attention: bool = False,
                    aggregator_attention: bool = False) -> np.ndarray:
    """Combine per-patch weight layers into a single (total_positions, 6) array.

    Starts with the main cross-frame attention (n_patches, L', 6) and
    optionally multiplies in additional weight layers.

    Parameters
    ----------
    weights : dict from ``model.annotate()`` for one sequence
    branch_cross_attention : multiply by branch's cross-frame attention
    branch_pooling_attention : multiply by branch's query-attention pooling
    encoder_pooling_attention : multiply by main path's query-attention pooling
    aggregator_attention : multiply by aggregator's patch-level pooling
    """
    w = weights['frame_w'].copy()                          # (n_patches, L', 6)

    if branch_cross_attention:
        w *= weights['branch_frame_w']

    if branch_pooling_attention:
        w *= weights['branch_pool_w'][..., None]           # (n, L') → (n, L', 1)

    if encoder_pooling_attention:
        w *= weights['pool_w'][..., None]                  # (n, L') → (n, L', 1)

    if aggregator_attention:
        w *= weights['agg_w'][:, None, None]               # (n,) → (n, 1, 1)

    return w.reshape(-1, 6)


@torch.no_grad()
def annotate_sequence(model, tokenizer, seq: str, patch_nt_len: int,
                      stride: int, device: torch.device,
                      max_patches: int = 4096,
                      branch_cross_attention: bool = False,
                      branch_pooling_attention: bool = False,
                      encoder_pooling_attention: bool = False,
                      aggregator_attention: bool = False) -> np.ndarray:
    """Extract per-position, per-frame weights from a single tiling pass.

    By default returns the main cross-frame attention.  Additional
    weight layers can be multiplied in via boolean flags.

    Returns (total_compressed_positions, 6) array.
    """
    patches = tile_sequence(seq, patch_nt_len, stride, max_patches)
    tokens, counts = tokenize_patches(patches, tokenizer)
    tokens = tokens.to(device, non_blocking=True)
    counts = counts.to(device, non_blocking=True)
    compression_factor = model.patch_encoder.frame_cnn.compression_factor
    seq_len = len(seq)

    all_weights = model.annotate(tokens, counts)
    w = combine_weights(
        all_weights[0],
        branch_cross_attention=branch_cross_attention,
        branch_pooling_attention=branch_pooling_attention,
        encoder_pooling_attention=encoder_pooling_attention,
        aggregator_attention=aggregator_attention,
    )
    return squeeze_weights(w, patch_nt_len, stride, seq_len,
                           compression_factor)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_annotations(weight_matrices: list, seq_ids: list,
                     seq_lengths: list, gene_lists: list,
                     out_path: str, patch_nt_len: int,
                     compression_factor: int, dpi: int = 150,
                     title: str = 'Cross-frame attention weights with predicted coding regions',
                     cbar_label: str = 'Frame attention weight',
                     top_n: int = 0):
    """Plot stacked per-frame attention heatmaps with gene overlay.

    Parameters
    ----------
    weight_matrices : list of (positions, 6) arrays
    seq_ids         : list of sequence identifiers
    seq_lengths     : list of sequence lengths in nt
    gene_lists      : list of lists from predict_genes() or external annotations
    out_path        : output filename (png/pdf/svg)
    patch_nt_len    : nucleotide length per patch (for coordinate mapping)
    top_n           : label the n genes with highest mean attention (0 = off)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = len(weight_matrices)
    if n == 0:
        logger.warning("No sequences to annotate.")
        return

    # Each genome gets 2 rows: gene track (thin) + heatmap
    fig_height = max(3.0, 1.6 * n + 1.5)
    fig, axes = plt.subplots(n, 1, figsize=(14, fig_height), squeeze=False,
                             gridspec_kw={'hspace': 0.55})

    # Shared color range across all genomes
    vmin = min(w.min() for w in weight_matrices if w.size > 0)
    vmax = max(w.max() for w in weight_matrices if w.size > 0)

    gene_colors = {
        0: '#B22222', 1: '#B22222', 2: '#B22222',   # forward
        3: '#FF8C00', 4: '#FF8C00', 5: '#FF8C00',   # reverse
    }
    
    def nt_to_pos(nt_coord):
            return (nt_coord - 1) // 3 // compression_factor
            
    for i, (w, sid, seq_len, genes) in enumerate(
            zip(weight_matrices, seq_ids, seq_lengths, gene_lists)):
        ax = axes[i, 0]
        n_pos = w.shape[0]

        # Heatmap
        im = ax.imshow(w.T, aspect='auto', cmap='viridis',
                       vmin=vmin, vmax=vmax, interpolation='none',
                       extent=[0, n_pos, 5.5, -0.5])

        # Overlay genes as semi-transparent bars
        for gene in genes:
            x_start = nt_to_pos(gene['begin'])
            x_end = nt_to_pos(gene['end'])
            fi = gene['frame_idx']
            y_center = fi
            bar_height = 0.4
            rect = mpatches.FancyBboxPatch(
                (x_start, y_center - bar_height / 2),
                max(x_end - x_start, 0.5), bar_height,
                boxstyle='square',
                facecolor='none', edgecolor=gene_colors[fi],
                linewidth=1.0, alpha=0.9, zorder=3)
            ax.add_patch(rect)

        # Label top-n genes by mean attention if they have category info
        if top_n > 0 and genes:
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

            # Sort by frame then position so alternation applies within each frame
            top_genes.sort(key=lambda x: (x[0]['frame_idx'], x[0]['begin']))
            frame_counters = {}

            for gene, cat in top_genes:
                x_mid = (nt_to_pos(gene['begin']) + nt_to_pos(gene['end'])) / 2
                fi = gene['frame_idx']
                # Alternate above / below within each frame
                count = frame_counters.get(fi, 0)
                place_above = (count % 2 == 0)
                frame_counters[fi] = count + 1
                if place_above:
                    y_label = fi - 0.45
                    va = 'bottom'
                else:
                    y_label = fi + 0.45
                    va = 'top'
                # Truncate long categories
                label_text = cat if len(cat) <= 25 else cat[:22] + '…'
                ax.annotate(
                    label_text,
                    xy=(x_mid, fi), xytext=(x_mid, y_label),
                    fontsize=6, ha='center', va=va, color='white',
                    fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#222',
                              alpha=0.8, edgecolor='none'),
                    arrowprops=dict(arrowstyle='-', color='white',
                                    lw=0.5, alpha=0.6),
                )

        ax.set_yticks(range(6))
        ax.set_yticklabels(FRAME_LABELS, fontsize=8)
        ax.set_ylim(5.5, -0.5)

        # Truncate long IDs
        label = sid if len(sid) <= 40 else sid[:37] + '...'
        ax.set_ylabel(label, fontsize=11, rotation=0, ha='right', va='center',
                      labelpad=25)

        # x-axis in kilobases
        n_ticks = min(8, max(2, n_pos // 20))
        tick_positions = np.linspace(0, n_pos, n_ticks)
        tick_labels_kb = [f'{nt / 1000:.1f}' for nt in
                          np.linspace(0, seq_len, n_ticks)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_kb, fontsize=9)
        if i == n - 1:
            ax.set_xlabel('Genome position (kb)', fontsize=11)

        # Gene count annotation
        n_fwd = sum(1 for g in genes if g['strand'] == 1)
        n_rev = sum(1 for g in genes if g['strand'] == -1)
        ax.text(1.0, 1.0, f'{len(genes)} genes ({n_fwd}→ {n_rev}←)',
                transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
                color='white', backgroundcolor='#333333',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#333333',
                          alpha=0.7, edgecolor='none'))

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=cbar_label)

    # Legend for gene colors
    legend_patches = [
        mpatches.Patch(facecolor='none', edgecolor=gene_colors[0],
                       linewidth=1.5, label='Forward genes'),
        mpatches.Patch(facecolor='none', edgecolor=gene_colors[3],
                       linewidth=1.5, label='Reverse genes'),
    ]
    fig.legend(handles=legend_patches, loc='lower right',
               bbox_to_anchor=(0.96, 0.92), fontsize=11, frameon=True,
               framealpha=0.9)

    fig.suptitle(title,
                 fontsize=11, fontweight='bold', y=0.99)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Annotation plot saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Annotate genomes with attention weights from the '
                    'hierarchical DNA model and plot heatmaps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input file: FASTA (.fasta/.fa/.fna) or '
                             'GenBank (.gb/.gbk/.genbank/.gbff). '
                             'Gzipped files supported. GenBank input '
                             'provides CDS annotations directly.')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Training run directory (contains calibration.json)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--output', '-o', type=str, default='frame_attention.png',
                        help='Output filename for annotation plot')
    parser.add_argument('--first_n', type=int, default=10,
                        help='Number of sequences to annotate and plot')
    parser.add_argument('--protein_annotations', type=str, default=None,
                        help='External protein annotation TSV (columns: gene, start, '
                             'stop, strand, contig, annot, category). '
                             'Replaces pyrodigal gene prediction.')
    parser.add_argument('--top_n', type=int, default=3,
                        help='Label the top-n genes by mean attention with their '
                             'category (requires --protein_annotations)')
    parser.add_argument('--normalize_importance', action='store_true',
                        help='Normalize each genome\'s weight matrix to [0, 1] '
                             'by dividing by its maximum value')
    parser.add_argument('--branch_cross_attention', action='store_true',
                        help='Multiply in the frame-stats branch\'s cross-frame '
                             'attention weights')
    parser.add_argument('--branch_pooling_attention', action='store_true',
                        help='Multiply in the frame-stats branch\'s '
                             'query-attention pooling weights')
    parser.add_argument('--encoder_pooling_attention', action='store_true',
                        help='Multiply in the main encoder path\'s '
                             'query-attention pooling weights')
    parser.add_argument('--aggregator_attention', action='store_true',
                        help='Multiply in the aggregator\'s '
                             'patch-level pooling weights')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # ---- load model ------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model, calib = load_model_and_calibration(
        args.run_dir, args.checkpoint, device)

    patch_nt_len = calib['model_config']['patch_nt_len']

    # ---- read input ------------------------------------------------------
    logger.info(f"Reading {args.input} …")

    # Detect GenBank by extension (strip .gz first if present)
    input_base = args.input[:-3] if args.input.endswith('.gz') else args.input
    is_genbank = os.path.splitext(input_base)[1].lower() in _GENBANK_EXTENSIONS

    if is_genbank:
        gb_records = read_genbank(args.input)
        logger.info(f"  {len(gb_records)} GenBank records "
                    f"(CDS annotations will be used directly)")
        # Convert to the same format as FASTA records
        records = [{'id': r['id'], 'seq': r['seq']} for r in gb_records]
        # Build annotation dict keyed by record id
        gb_annot = {r['id']: r['genes'] for r in gb_records}
    else:
        records = read_fasta(args.input)
        gb_annot = None

    logger.info(f"  {len(records)} sequences")

    # ---- annotate --------------------------------------------------------
    tokenizer = CodonTokenizer()
    n_annot = min(args.first_n, len(records))
    stride = int(patch_nt_len * 0.5)

    # Build description of active weight layers
    active_layers = ['cross-frame']
    if args.branch_cross_attention:
        active_layers.append('branch cross-frame')
    if args.branch_pooling_attention:
        active_layers.append('branch pooling')
    if args.encoder_pooling_attention:
        active_layers.append('encoder pooling')
    if args.aggregator_attention:
        active_layers.append('aggregator')
    mode_str = ' × '.join(active_layers)

    logger.info(f"Annotating first {n_annot} sequences "
                f"({mode_str}, stride {stride} nt) …")

    # Load external protein annotations if provided
    ext_annot = None
    if args.protein_annotations:
        ext_annot = load_protein_annotations(args.protein_annotations)
        logger.info(f"Loaded external annotations for "
                    f"{len(ext_annot)} contigs from {args.protein_annotations}")

    top_n = args.top_n if (ext_annot or gb_annot) else 0
    compression_factor = model.patch_encoder.frame_cnn.compression_factor

    weight_matrices = []
    annot_ids = []
    annot_lengths = []
    annot_genes = []
    for rec in records[:n_annot]:
        seq = rec['seq'][1:]

        w = annotate_sequence(
            model, tokenizer, seq, patch_nt_len, stride, device,
            branch_cross_attention=args.branch_cross_attention,
            branch_pooling_attention=args.branch_pooling_attention,
            encoder_pooling_attention=args.encoder_pooling_attention,
            aggregator_attention=args.aggregator_attention,
        )

        weight_matrices.append(w)
        annot_ids.append(rec['id'])
        annot_lengths.append(len(seq))

        # Get genes: GenBank annotations > external TSV > pyrodigal
        if gb_annot and rec['id'] in gb_annot:
            genes = copy.deepcopy(gb_annot[rec['id']])
        elif ext_annot and rec['id'] in ext_annot:
            genes = copy.deepcopy(ext_annot[rec['id']])
            genes = assign_frame_indices(genes, len(seq))
        else:
            if gb_annot:
                logger.warning(f"  {rec['id']}: not found in GenBank records, "
                               f"falling back to pyrodigal")
            elif ext_annot:
                logger.warning(f"  {rec['id']}: not found in external "
                               f"annotations, falling back to pyrodigal")
            genes = predict_genes(seq)

        annot_genes.append(genes)
        n_fwd = sum(1 for g in genes if g['strand'] == 1)
        n_rev = len(genes) - n_fwd
        logger.info(f"  {rec['id']}: {len(seq)} nt → "
                    f"{w.shape[0]} positions × {w.shape[1]} frames, "
                    f"{len(genes)} genes ({n_fwd}→ {n_rev}←)")

    # ---- optional per-genome normalization ---------------------------------
    if args.normalize_importance:
        for i, w in enumerate(weight_matrices):
            wmax = w.max()
            if wmax > 0:
                weight_matrices[i] = w / wmax
        logger.info("Normalized weight matrices to per-genome maximum")

    plot_title = f'Attention weights ({mode_str}) with predicted coding regions'
    plot_cbar = mode_str

    if args.normalize_importance:
        plot_cbar += ' (normalized)'

    plot_annotations(weight_matrices, annot_ids, annot_lengths,
                     annot_genes, args.output, patch_nt_len,
                     compression_factor, title=plot_title, 
                     cbar_label=plot_cbar, top_n=top_n)

    logger.info("Done.")


if __name__ == '__main__':
    main()
