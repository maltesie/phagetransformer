#!/usr/bin/env python3
"""Prophage region detection in bacterial genomes.

Scans a bacterial genome with a sliding window, runs the phage host-
prediction model on each window, and plots prediction confidence and
importance along the genome.  Windows where the model confidently
predicts a host are candidate prophage regions.

Usage:
    python extract.py --input bacterium.fasta --model_dir ./models/HierDNA
    python extract.py --input bacterium.gb --model_dir ./models/HierDNA \
        --window_size 120000 --stride 60000 --threshold 0.5
    python extract.py --input bacterium.fasta --model_dir ./models/HierDNA \
        --output prophage_scan.png --importance

Supported input formats:
    FASTA  (.fasta, .fa, .fna, + .gz)
    GenBank (.gb, .gbk, .genbank, .gbff, + .gz) — CDS overlaid on plot
"""

import argparse
import logging
import os
from typing import Optional

import numpy as np
import torch

from phagetransformer.model import CodonTokenizer
from phagetransformer.predict import (
    read_fasta,
    load_model_and_calibration,
    tile_sequence,
    tokenize_patches,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GenBank support (shared with annotate.py)
# ---------------------------------------------------------------------------

_GENBANK_EXTENSIONS = {'.gb', '.gbk', '.genbank', '.gbff'}


def read_genbank(path: str) -> list:
    """Read a GenBank file and extract sequences with CDS annotations."""
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
                begin = int(feat.location.start) + 1
                end = int(feat.location.end)
                strand = 1 if feat.location.strand == 1 else -1
                annot = feat.qualifiers.get('product', [''])[0]
                gene_name = feat.qualifiers.get('gene', [''])[0]
                category = annot if annot else gene_name
                genes.append({
                    'begin': begin, 'end': end,
                    'strand': strand,
                    'annot': annot, 'category': category,
                })
            records.append({'id': rec.id if rec.id != '.' else rec.name,
                            'seq': seq, 'genes': genes})
    return records


# ---------------------------------------------------------------------------
# Sliding-window scanning
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_window(model, tokenizer, seq: str, patch_nt_len: int,
                   device: torch.device,
                   max_patches: int = 4096) -> np.ndarray:
    """Run model on a single sequence window.

    Returns
    -------
    probs : (num_classes,) sigmoid probabilities (multi-label)
    """
    stride = int(patch_nt_len / 3) * 3
    patches, _ = tile_sequence(seq, patch_nt_len, stride, max_patches)
    tokens, counts = tokenize_patches(patches, tokenizer)
    tokens = tokens.to(device, non_blocking=True)
    counts = counts.to(device, non_blocking=True)

    logits = model(tokens, counts)               # (1, num_classes)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    return probs


@torch.no_grad()
def scan_genome(model, tokenizer, seq: str, patch_nt_len: int,
                device: torch.device, window_size: int, stride: int,
                host_list: list, max_patches: int = 4096,
                ignore_idx: Optional[int] = None) -> dict:
    """Slide a window along a genome, collecting predictions per window.

    If *ignore_idx* is set, that class is excluded when determining
    the top prediction per window (but kept in all_probs).

    Returns dict with:
        windows     : list of (start_nt, end_nt) tuples
        confidences : (n_windows,) top-1 confidence per window
        predictions : list of predicted host names
        pred_indices: (n_windows,) predicted class indices
        all_probs   : (n_windows, num_classes) full probability matrix
        importance  : (n_positions,) genome-wide importance signal or None
    """
    seq_len = len(seq)
    windows = []
    start = 0
    while start < seq_len:
        end = min(start + window_size, seq_len)
        # Skip very short trailing windows
        if end - start < patch_nt_len:
            break
        windows.append((start, end))
        if end == seq_len:
            break
        start += stride

    n_windows = len(windows)
    logger.info(f"  Scanning {seq_len:,} nt with {n_windows} windows "
                f"({window_size:,} nt, stride {stride:,})")

    all_probs = []
    confidences = []
    predictions = []
    pred_indices = []

    for wi, (ws, we) in enumerate(windows):
        chunk = seq[ws:we]

        probs = predict_window(model, tokenizer, chunk, patch_nt_len,
                               device, max_patches)
        # Determine top prediction, optionally ignoring a class
        if ignore_idx is not None:
            probs_masked = probs.copy()
            probs_masked[ignore_idx] = -1.0
            top_idx = int(np.argmax(probs_masked))
        else:
            top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])

        all_probs.append(probs)
        confidences.append(top_conf)
        pred_indices.append(top_idx)
        predictions.append(host_list[top_idx] if top_idx < len(host_list)
                           else f'class_{top_idx}')

        if (wi + 1) % 10 == 0 or wi == n_windows - 1:
            logger.info(f"    Window {wi+1}/{n_windows}: "
                        f"{ws:,}–{we:,} nt → {predictions[-1]} "
                        f"({top_conf:.3f})")

    # Build genome-wide confidence signal at 1-nt resolution, using max
    # across overlapping windows
    confidence_track = np.zeros(seq_len, dtype=np.float32)
    pred_track = np.full(seq_len, -1, dtype=np.int32)
    for (ws, we), conf, pidx in zip(windows, confidences, pred_indices):
        mask = confidence_track[ws:we] < conf
        confidence_track[ws:we] = np.where(mask, conf, confidence_track[ws:we])
        pred_track[ws:we] = np.where(mask, pidx, pred_track[ws:we])

    return {
        'windows': windows,
        'confidences': np.array(confidences),
        'predictions': predictions,
        'pred_indices': np.array(pred_indices),
        'all_probs': np.array(all_probs),
        'confidence_track': confidence_track,
        'pred_track': pred_track,
    }


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def extract_regions(scan: dict, threshold: float,
                    host_list: list,
                    min_region_nt: int = 5000,
                    merge_gap_nt: int = 10000,
                    ignore_idx: Optional[int] = None) -> list:
    """Extract contiguous regions above confidence threshold.

    Returns list of dicts:
        start, end    : 0-based nt coordinates
        peak_conf     : max confidence in region
        mean_conf     : mean confidence in region
        predicted_host: host with highest mean probability across region
    """
    track = scan['confidence_track']
    above = track >= threshold

    # Find contiguous runs
    regions = []
    in_region = False
    for i in range(len(above)):
        if above[i] and not in_region:
            region_start = i
            in_region = True
        elif not above[i] and in_region:
            regions.append((region_start, i))
            in_region = False
    if in_region:
        regions.append((region_start, len(above)))

    # Merge regions separated by small gaps
    merged = []
    for start, end in regions:
        if merged and start - merged[-1][1] < merge_gap_nt:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Filter by minimum size and annotate
    result = []
    for start, end in merged:
        if end - start < min_region_nt:
            continue
        region_track = track[start:end]
        # Find which windows overlap this region
        region_probs = []
        for (ws, we), probs in zip(scan['windows'], scan['all_probs']):
            overlap = max(0, min(we, end) - max(ws, start))
            if overlap > 0:
                region_probs.append(probs)

        if region_probs:
            mean_probs = np.mean(region_probs, axis=0)
            if ignore_idx is not None:
                mean_probs_masked = mean_probs.copy()
                mean_probs_masked[ignore_idx] = -1.0
                best_idx = int(np.argmax(mean_probs_masked))
            else:
                best_idx = int(np.argmax(mean_probs))
            predicted_host = (host_list[best_idx] if best_idx < len(host_list)
                              else f'class_{best_idx}')
        else:
            predicted_host = 'unknown'

        result.append({
            'start': start,
            'end': end,
            'length': end - start,
            'peak_conf': float(region_track.max()),
            'mean_conf': float(region_track.mean()),
            'predicted_host': predicted_host,
        })

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scan(scan: dict, seq_id: str, seq_len: int,
              threshold: float, out_path: str,
              host_list: list,
              genes: Optional[list] = None,
              ignore_idx: Optional[int] = None,
              dpi: int = 150):
    """Plot genome-wide prophage scan as a coloured confidence line.

    Top panel:  Continuous line of max-class confidence per window,
                coloured by the predicted genus at each segment.
    Bottom panel (if genes): CDS annotation track.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    from matplotlib.colors import hsv_to_rgb

    windows = scan['windows']
    all_probs = scan['all_probs']           # (n_windows, num_classes)
    n_windows, n_classes = all_probs.shape

    # Colour palette — one per class
    colors = [hsv_to_rgb((i / max(n_classes, 1), 0.65, 0.85))
              for i in range(n_classes)]
    if n_classes > 0 and host_list and host_list[-1] == 'bacterial_fragment':
        colors[-1] = (0.5, 0.5, 0.5)

    # Per-window: midpoint (kb), max confidence, argmax class
    mid_kb = np.array([(ws + we) / 2000.0 for ws, we in windows])
    if ignore_idx is not None:
        probs_masked = all_probs.copy()
        probs_masked[:, ignore_idx] = -1.0
        max_conf = probs_masked.max(axis=1)
        max_class = probs_masked.argmax(axis=1)
    else:
        max_conf = all_probs.max(axis=1)
        max_class = all_probs.argmax(axis=1)

    has_genes = genes is not None and len(genes) > 0
    n_panels = 2 if has_genes else 1
    height_ratios = [3, 1] if has_genes else [1]

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 + 1.5 * n_panels),
                             gridspec_kw={'height_ratios': height_ratios,
                                          'hspace': 0.3},
                             squeeze=False)

    ax = axes[0, 0]

    # Draw coloured line segments between consecutive windows
    if n_windows > 1:
        for i in range(n_windows - 1):
            ax.plot(mid_kb[i:i+2], max_conf[i:i+2],
                    color=colors[max_class[i] % len(colors)],
                    linewidth=1.5, solid_capstyle='round')
        # Fill under the line with light colour per segment
        for i in range(n_windows - 1):
            ax.fill_between(mid_kb[i:i+2], 0, max_conf[i:i+2],
                            color=colors[max_class[i] % len(colors)],
                            alpha=0.25, linewidth=0)
    elif n_windows == 1:
        ax.bar(mid_kb[0], max_conf[0],
               width=(windows[0][1] - windows[0][0]) / 1000.0,
               color=colors[max_class[0] % len(colors)], alpha=0.6)

    # Threshold line
    ax.axhline(threshold, color='#B2182B', linewidth=1.0, linestyle='--',
               alpha=0.8, label=f'Threshold ({threshold:.2f})')

    ax.set_xlim(0, seq_len / 1000)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Max prediction\nprobability', fontsize=10)

    # Legend for genera that appear as top prediction above threshold
    active_classes = set()
    for i in range(n_windows):
        if max_conf[i] >= threshold:
            active_classes.add(max_class[i])
    legend_handles = [mpatches.Patch(facecolor='#B2182B', alpha=0.3,
                                     label=f'Threshold ({threshold:.2f})')]
    for ci in sorted(active_classes):
        label = (host_list[ci] if ci < len(host_list)
                 else f'class_{ci}')
        legend_handles.append(
            mpatches.Patch(facecolor=colors[ci % len(colors)],
                           alpha=0.85, label=label))
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=7, framealpha=0.9, ncol=min(4, len(legend_handles)))

    if not has_genes:
        ax.set_xlabel('Genome position (kb)', fontsize=10)

    # Title
    n_above = sum(1 for c in max_conf if c >= threshold)
    title = (f'Prophage scan: {seq_id}  ({seq_len:,} nt) — '
             f'{n_above}/{n_windows} windows above threshold')
    ax.set_title(title, fontsize=11, fontweight='bold')

    # --- Gene track ---
    if has_genes:
        ax_g = axes[1, 0]
        gene_colors_fwd = '#4393C3'
        gene_colors_rev = '#F4A582'

        for gene in genes:
            start_kb = (gene['begin'] - 1) / 1000
            width_kb = (gene['end'] - gene['begin'] + 1) / 1000
            color = gene_colors_fwd if gene['strand'] == 1 else gene_colors_rev
            y = 0.55 if gene['strand'] == 1 else 0.15
            rect = mpatches.FancyBboxPatch(
                (start_kb, y), width_kb, 0.3,
                boxstyle='square', facecolor=color,
                edgecolor='none', alpha=0.7, zorder=2)
            ax_g.add_patch(rect)

        ax_g.set_xlim(0, seq_len / 1000)
        ax_g.set_ylim(0, 1)
        ax_g.set_yticks([0.3, 0.7])
        ax_g.set_yticklabels(['Rev', 'Fwd'], fontsize=8)
        ax_g.set_xlabel('Genome position (kb)', fontsize=10)
        ax_g.set_ylabel('CDS', fontsize=10)

        legend_patches = [
            mpatches.Patch(facecolor=gene_colors_fwd, alpha=0.7,
                           label='Forward CDS'),
            mpatches.Patch(facecolor=gene_colors_rev, alpha=0.7,
                           label='Reverse CDS'),
        ]
        ax_g.legend(handles=legend_patches, loc='upper right',
                    fontsize=7, framealpha=0.9)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Scan plot saved: {out_path}")
    root = os.path.splitext(out_path)[0]
    for ext in ('.pdf', '.svg'):
        path = root + ext
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Figure saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# TSV output
# ---------------------------------------------------------------------------

def write_regions_tsv(regions: list, seq_id: str, out_path: str):
    """Write detected regions to a TSV file."""
    import csv
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['contig', 'start', 'end', 'length',
                           'peak_conf', 'mean_conf', 'predicted_host'],
            delimiter='\t')
        writer.writeheader()
        for reg in regions:
            writer.writerow({
                'contig': seq_id,
                'start': reg['start'] + 1,   # 1-based
                'end': reg['end'],
                'length': reg['length'],
                'peak_conf': f"{reg['peak_conf']:.4f}",
                'mean_conf': f"{reg['mean_conf']:.4f}",
                'predicted_host': reg['predicted_host'],
            })
    logger.info(f"Regions TSV saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Scan bacterial genomes for candidate prophage regions '
                    'using the phage host-prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input FASTA or GenBank file')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory (contains calibration.json and checkpoints/)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--output', '-o', type=str, default='prophage_scan.png',
                        help='Output plot filename')
    parser.add_argument('--output_tsv', type=str, default=None,
                        help='Output TSV of detected regions '
                             '(default: <output_stem>_regions.tsv)')
    parser.add_argument('--window_size', type=int, default=100000,
                        help='Sliding window size in nucleotides')
    parser.add_argument('--stride', type=int, default=None,
                        help='Window stride in nt (default: window_size // 2)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for prophage candidate regions')
    parser.add_argument('--min_region', type=int, default=5000,
                        help='Minimum region size in nt to report')
    parser.add_argument('--merge_gap', type=int, default=5000,
                        help='Merge candidate regions separated by less than '
                             'this many nt')
    parser.add_argument('--first_n', type=int, default=1,
                        help='Number of sequences to scan')
    parser.add_argument('--ignore_bacterial', action='store_true',
                        help='Ignore the bacterial_fragment class when '
                             'determining top predictions and labels')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.stride is None:
        args.stride = args.window_size // 2

    # Default TSV path
    if args.output_tsv is None:
        stem = os.path.splitext(args.output)[0]
        args.output_tsv = f'{stem}_regions.tsv'

    # ---- load model ---------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model, calib = load_model_and_calibration(
        args.model_dir, args.checkpoint, device)

    patch_nt_len = calib['model_config']['patch_nt_len']
    host_list = [h.split(';')[-1] for h in calib.get('host_list', calib.get('hosts', []))]

    # Detect bacterial_fragment index
    ignore_idx = None
    if args.ignore_bacterial:
        try:
            ignore_idx = host_list.index('bacterial_fragment')
            logger.info(f"  Ignoring bacterial_fragment class (index {ignore_idx})")
        except ValueError:
            logger.info("  No bacterial_fragment class found, nothing to ignore")

    # ---- read input ---------------------------------------------------------
    logger.info(f"Reading {args.input} …")

    input_base = args.input[:-3] if args.input.endswith('.gz') else args.input
    is_genbank = os.path.splitext(input_base)[1].lower() in _GENBANK_EXTENSIONS

    if is_genbank:
        gb_records = read_genbank(args.input)
        records = [{'id': r['id'], 'seq': r['seq']} for r in gb_records]
        gb_genes = {r['id']: r['genes'] for r in gb_records}
        logger.info(f"  {len(records)} GenBank records")
    else:
        records = read_fasta(args.input)
        gb_genes = None
        logger.info(f"  {len(records)} FASTA sequences")

    # ---- scan ---------------------------------------------------------------
    tokenizer = CodonTokenizer()
    n_scan = min(args.first_n, len(records))

    all_regions = []
    for ri, rec in enumerate(records[:n_scan]):
        seq = rec['seq']
        sid = rec['id']
        logger.info(f"Scanning {sid} ({len(seq):,} nt) …")

        # Get genes for gene track
        genes = None
        if gb_genes and sid in gb_genes:
            genes = gb_genes[sid]

        # Per-sequence output filenames
        if n_scan > 1:
            stem, ext = os.path.splitext(args.output)
            plot_path = f'{stem}_{ri}{ext}'
            tsv_path = f'{stem}_{ri}_regions.tsv'
        else:
            plot_path = args.output
            tsv_path = args.output_tsv

        scan = scan_genome(
            model, tokenizer, seq, patch_nt_len, device,
            window_size=args.window_size, stride=args.stride,
            host_list=host_list, ignore_idx=ignore_idx,
        )

        regions = extract_regions(
            scan, threshold=args.threshold, host_list=host_list,
            min_region_nt=args.min_region, merge_gap_nt=args.merge_gap,
            ignore_idx=ignore_idx,
        )

        logger.info(f"  {len(regions)} candidate prophage region(s) "
                    f"above {args.threshold:.2f} threshold")
        for reg in regions:
            logger.info(f"    {reg['start']+1:>10,}–{reg['end']:>10,}  "
                        f"({reg['length']:>7,} nt)  "
                        f"peak={reg['peak_conf']:.3f}  "
                        f"host={reg['predicted_host']}")

        plot_scan(scan, sid, len(seq), args.threshold,
                  plot_path, host_list, genes=genes,
                  ignore_idx=ignore_idx)

        if regions:
            write_regions_tsv(regions, sid, tsv_path)
            all_regions.extend(regions)

    total = len(all_regions)
    logger.info(f"Done. {total} total candidate region(s) across "
                f"{n_scan} sequence(s).")


if __name__ == '__main__':
    main()
