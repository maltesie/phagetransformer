#!/usr/bin/env python3
"""Prophage region detection in bacterial genomes.

Scans a bacterial genome with a sliding window, runs the phage host-
prediction model on each window, and plots prediction confidence and
importance along the genome.  Windows where the model confidently
predicts a host are candidate prophage regions.

Single-genome usage:
    python scan.py -i bacterium.fasta --model_dir ./models/HierDNA
    python scan.py -i bacterium.gb --model_dir ./models/HierDNA \
        -o ./my_results --window_size 120000 --stride 60000

Batch mode (folder of genomes):
    python scan.py --input_dir ./genomes --model_dir ./models/HierDNA \
        -o ./scan_results --threshold 0.4
    python scan.py --input_dir ./genomes --model_dir ./models/HierDNA \
        --manifest host_genome_manifest.tsv --one_per_genus \
        --per_genome_output

Output (--output, always a directory):
    Single mode:  <output>/<input_stem>_scan.png, *_regions.tsv
    Batch mode:   <output>/batch_summary.tsv (one row per genome file,
                  aggregated across all sequences), batch_scatter.png
                  + per-genome files only with --per_genome_output

Replot from existing results (no model needed):
    python scan.py --replot -o ./scan_results

Supported input formats:
    FASTA  (.fasta, .fa, .fna, + .gz)
    GenBank (.gb, .gbk, .genbank, .gbff, + .gz) — CDS overlaid on plot
"""

import argparse
import csv
import glob
import logging
import os
from typing import Optional

import numpy as np
import torch

from phagetransformer.model import CodonTokenizer
from eval_utils import (
    COLORS, FONT_SIZES, FIG_WIDTH, FIG_HEIGHT_ROW,
    setup_style, _save_figure, _output_path,
    enable_presentation_mode,
)
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

    setup_style()

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

    fig, axes = plt.subplots(n_panels, 1, figsize=(FIG_WIDTH, FIG_HEIGHT_ROW),
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
    ax.axhline(threshold, color=COLORS['threshold'], linewidth=1.0,
               linestyle='--', alpha=0.8,
               label=f'Threshold ({threshold:.2f})')

    ax.set_xlim(0, seq_len / 1000)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Max prediction\nprobability')

    # Legend for genera that appear as top prediction above threshold
    active_classes = set()
    for i in range(n_windows):
        if max_conf[i] >= threshold:
            active_classes.add(max_class[i])
    legend_handles = [mpatches.Patch(facecolor=COLORS['threshold'], alpha=0.3,
                                     label=f'Threshold ({threshold:.2f})')]
    for ci in sorted(active_classes):
        label = (host_list[ci] if ci < len(host_list)
                 else f'class_{ci}')
        legend_handles.append(
            mpatches.Patch(facecolor=colors[ci % len(colors)],
                           alpha=0.85, label=label))
    ax.legend(handles=legend_handles, loc='lower right',
              framealpha=0.9, edgecolor=COLORS['grid'],
              ncol=min(4, len(legend_handles)))

    if not has_genes:
        ax.set_xlabel('Genome position (kb)')

    # Title
    n_above = sum(1 for c in max_conf if c >= threshold)
    title = (f'Prophage scan: {seq_id}  ({seq_len:,} nt) — '
             f'{n_above}/{n_windows} windows above threshold')
    ax.set_title(title,
                 color=COLORS['text'])

    # --- Gene track ---
    if has_genes:
        ax_g = axes[1, 0]

        for gene in genes:
            start_kb = (gene['begin'] - 1) / 1000
            width_kb = (gene['end'] - gene['begin'] + 1) / 1000
            color = (COLORS['gene_fwd'] if gene['strand'] == 1
                     else COLORS['gene_rev'])
            y = 0.55 if gene['strand'] == 1 else 0.15
            rect = mpatches.FancyBboxPatch(
                (start_kb, y), width_kb, 0.3,
                boxstyle='square', facecolor=color,
                edgecolor='none', alpha=0.7, zorder=2)
            ax_g.add_patch(rect)

        ax_g.set_xlim(0, seq_len / 1000)
        ax_g.set_ylim(0, 1)
        ax_g.set_yticks([0.3, 0.7])
        ax_g.set_yticklabels(['Rev', 'Fwd'])
        ax_g.set_xlabel('Genome position (kb)')
        ax_g.set_ylabel('CDS')

        legend_patches = [
            mpatches.Patch(facecolor=COLORS['gene_fwd'], alpha=0.7,
                           label='Forward CDS'),
            mpatches.Patch(facecolor=COLORS['gene_rev'], alpha=0.7,
                           label='Reverse CDS'),
        ]
        ax_g.legend(handles=legend_patches, loc='upper right',
                    framealpha=0.9, edgecolor=COLORS['grid'])

    out_path = _output_path(out_path)
    _save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# TSV output
# ---------------------------------------------------------------------------

def write_regions_tsv(regions: list, seq_id: str, out_path: str):
    """Write detected regions to a TSV file."""
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
# Batch mode helpers
# ---------------------------------------------------------------------------

_FASTA_EXTENSIONS = {'.fasta', '.fa', '.fna'}
_ALL_GENOME_EXTENSIONS = _FASTA_EXTENSIONS | _GENBANK_EXTENSIONS


def collect_genome_files(input_dir: str,
                         manifest: Optional[str] = None,
                         one_per_genus: bool = False) -> list:
    """Collect genome file paths from a directory.

    If *manifest* is given (filename relative to *input_dir* or absolute
    path), reads a TSV with at least a ``genome_path`` column (the same
    format used by ``codon_stats.py`` / ``host_genome_manifest.tsv``).
    Otherwise globs for FASTA and GenBank files in *input_dir*.

    When *one_per_genus* is True and a manifest with a ``species`` column
    is available, only the first genome per genus (first word of species)
    is kept.

    Returns list of dicts ``{'path': str, 'label': str, 'genus': str}``
    where *label* is a short identifier derived from the filename or
    manifest metadata, and *genus* is the host genus (empty string when
    no manifest species info is available).
    """
    entries = []

    if manifest:
        manifest_path = (manifest if os.path.isabs(manifest)
                         else os.path.join(input_dir, manifest))
        with open(manifest_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                genome_path = row['genome_path']
                if not os.path.isabs(genome_path):
                    genome_path = os.path.join(input_dir, genome_path)
                if not os.path.exists(genome_path):
                    logger.warning(f"  Skipping missing file: {genome_path}")
                    continue
                species = row.get('species', '')
                genus = species.split()[0] if species else ''
                label = species or row.get('id', '')
                if not label:
                    label = os.path.splitext(
                        os.path.basename(genome_path))[0]
                entries.append({'path': genome_path, 'label': label,
                                'genus': genus})
    else:
        # Glob for genome files (including .gz variants)
        for ext in sorted(_ALL_GENOME_EXTENSIONS):
            for path in sorted(glob.glob(os.path.join(input_dir,
                                                       f'*{ext}'))):
                entries.append({
                    'path': path,
                    'label': os.path.splitext(
                        os.path.basename(path))[0],
                    'genus': '',
                })
            for path in sorted(glob.glob(os.path.join(input_dir,
                                                       f'*{ext}.gz'))):
                entries.append({
                    'path': path,
                    'label': os.path.splitext(
                        os.path.splitext(
                            os.path.basename(path))[0])[0],
                    'genus': '',
                })

    # Deduplicate by resolved path while preserving order
    seen = set()
    deduped = []
    for entry in entries:
        rp = os.path.realpath(entry['path'])
        if rp not in seen:
            seen.add(rp)
            deduped.append(entry)

    # One-per-genus filtering
    if one_per_genus:
        seen_genera = set()
        filtered = []
        n_before = len(deduped)
        for entry in deduped:
            g = entry['genus']
            if not g:
                # No genus info — keep everything
                filtered.append(entry)
            elif g not in seen_genera:
                seen_genera.add(g)
                filtered.append(entry)
        logger.info(f"  --one_per_genus: {n_before} → {len(filtered)} "
                    f"genomes ({len(seen_genera)} genera)")
        return filtered

    return deduped


def aggregate_file_summary(summaries: list) -> dict:
    """Aggregate per-sequence summaries into one row per genome file.

    Parameters
    ----------
    summaries : list of dicts from ``scan_single_file`` for one file.

    Returns a single dict with pooled stats.
    """
    n_sequences = len(summaries)
    total_len = sum(s['seq_len'] for s in summaries)
    total_win = sum(s['n_windows'] for s in summaries)
    total_reg = sum(s['n_regions'] for s in summaries)
    total_reg_nt = sum(s['total_region_nt'] for s in summaries)
    total_correct = sum(s['n_correct'] for s in summaries)
    total_correct_conf = sum(s['correct_conf_sum'] for s in summaries)
    total_correct_strict = sum(s['n_correct_strict'] for s in summaries)
    total_correct_conf_strict = sum(s['correct_conf_sum_strict']
                                    for s in summaries)
    total_conf = sum(s['conf_sum'] for s in summaries)
    total_bacterial = sum(s['n_bacterial'] for s in summaries)

    peak = max((s['peak_conf'] for s in summaries), default=0.0)
    top_host = ''
    if any(s['top_host'] for s in summaries):
        best = max((s for s in summaries if s['top_host']),
                   key=lambda s: s['peak_conf'])
        top_host = best['top_host']

    correct_ratio = (total_correct / total_win) if total_win else None
    mean_correct_conf = ((total_correct_conf / total_correct)
                         if total_correct else None)
    correct_ratio_strict = ((total_correct_strict / total_win)
                            if total_win else None)
    mean_correct_conf_strict = ((total_correct_conf_strict
                                 / total_correct_strict)
                                if total_correct_strict else None)
    mean_all_conf = (total_conf / total_win) if total_win else None
    bacterial_ratio = (total_bacterial / total_win) if total_win else 0.0

    return {
        'genome_file': summaries[0]['genome_file'],
        'label': summaries[0]['label'],
        'genus': summaries[0]['genus'],
        'n_sequences': n_sequences,
        'seq_len': total_len,
        'n_windows': total_win,
        'n_regions': total_reg,
        'total_region_nt': total_reg_nt,
        'peak_conf': peak,
        'top_host': top_host,
        'correct_ratio': correct_ratio,
        'mean_correct_conf': mean_correct_conf,
        'correct_ratio_strict': correct_ratio_strict,
        'mean_correct_conf_strict': mean_correct_conf_strict,
        'mean_all_conf': mean_all_conf,
        'bacterial_ratio': bacterial_ratio,
    }


def write_batch_summary_tsv(summary_rows: list, out_path: str):
    """Write a summary TSV with one row per genome file in batch mode."""
    fieldnames = ['genome_file', 'label', 'genus', 'n_sequences',
                  'seq_len', 'n_windows', 'n_regions', 'total_region_nt',
                  'peak_conf', 'top_host',
                  'correct_ratio', 'mean_correct_conf',
                  'correct_ratio_strict', 'mean_correct_conf_strict',
                  'mean_all_conf', 'bacterial_ratio']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in summary_rows:
            out = dict(row)
            out['peak_conf'] = f"{row['peak_conf']:.4f}"
            out['bacterial_ratio'] = f"{row['bacterial_ratio']:.4f}"
            for key in ('correct_ratio', 'mean_correct_conf',
                        'correct_ratio_strict', 'mean_correct_conf_strict',
                        'mean_all_conf'):
                if row[key] is not None:
                    out[key] = f"{row[key]:.4f}"
                else:
                    out[key] = ''
            writer.writerow(out)
    logger.info(f"Batch summary saved: {out_path} ({len(summary_rows)} rows)")


def read_batch_summary_tsv(path: str) -> list:
    """Read a batch_summary.tsv back into the summary-row format.

    Numeric fields are parsed back to floats (or None for empty cells)
    so the result can be passed directly to :func:`plot_batch_scatter`.
    """
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            parsed = dict(row)
            # Parse optional float fields (empty string → None)
            for key in ('correct_ratio', 'mean_correct_conf',
                        'correct_ratio_strict', 'mean_correct_conf_strict',
                        'mean_all_conf'):
                val = row.get(key, '')
                parsed[key] = float(val) if val else None
            # Parse always-present float fields
            for key in ('peak_conf', 'bacterial_ratio'):
                parsed[key] = float(row[key]) if row.get(key) else 0.0
            # Parse int fields
            for key in ('n_sequences', 'seq_len', 'n_windows',
                        'n_regions', 'total_region_nt'):
                parsed[key] = int(row[key]) if row.get(key) else 0
            rows.append(parsed)
    logger.info(f"Read {len(rows)} rows from {path}")
    return rows


def plot_batch_scatter(summary_rows: list, out_path: str,
                       strict: bool = False, dpi: int = 150):
    """Two-panel scatter with marginal density strips.

    Panel A: correct_ratio vs mean confidence of correct windows.
    Panel B: correct_ratio vs mean confidence of all windows.

    When *strict* is True, the x-axis uses ``correct_ratio_strict``
    and Panel A uses ``mean_correct_conf_strict`` (windows must predict
    the correct genus AND have bacterial_fragment below threshold).

    Only genomes with a known genus are plotted.  Panel A additionally
    requires at least one correct window.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    setup_style()

    # ---- column selection ----------------------------------------------------
    cr_key = 'correct_ratio_strict' if strict else 'correct_ratio'
    mcc_key = 'mean_correct_conf_strict' if strict else 'mean_correct_conf'
    mode_label = '(strict — excl. bacterial)' if strict else ''

    # ---- data prep ----------------------------------------------------------
    rows_all = [r for r in summary_rows
                if r.get('genus')
                and r.get(cr_key) is not None
                and r.get('mean_all_conf') is not None]
    rows_correct = [r for r in rows_all
                    if r.get(mcc_key) is not None]

    if not rows_all:
        logger.warning("  No genomes with genus info — skipping "
                       "batch scatter plot.")
        return

    x_a = np.array([r[cr_key] for r in rows_correct])
    y_a = np.array([r[mcc_key] for r in rows_correct])
    x_b = np.array([r[cr_key] for r in rows_all])
    y_b = np.array([r['mean_all_conf'] for r in rows_all])

    # ---- figure layout ------------------------------------------------------
    # 2 panels, each with scatter + x-margin + y-margin
    #   col0: scatter_A   col1: ystrip_A   col2: scatter_B   col3: ystrip_B
    #   row0: scatter                       scatter
    #   row1: xstrip                        xstrip
    margin_ratio = 0.15
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT_ROW + 0.8))
    gs = GridSpec(2, 5, figure=fig,
                  width_ratios=[1, margin_ratio, 0.12, 1, margin_ratio],
                  height_ratios=[1, margin_ratio],
                  hspace=0.05, wspace=0.05)

    scatter_kw = dict(s=16, alpha=0.5, color=COLORS['primary'],
                      edgecolors='none', rasterized=True)
    hist_kw = dict(bins=30, color=COLORS['primary'], alpha=0.5,
                   edgecolor='none')

    x_label = ('Fraction of windows predicting correct genus'
               + (' (excl. bacterial)' if strict else ''))

    panels = [
        (gs[0, 0], gs[1, 0], gs[0, 1], x_a, y_a,
         'Mean confidence\n(correct windows)',
         f'A — correct windows only (n={len(rows_correct)})'),
        (gs[0, 3], gs[1, 3], gs[0, 4], x_b, y_b,
         'Mean confidence\n(all windows)',
         f'B — all windows (n={len(rows_all)})'),
    ]

    for gs_main, gs_xm, gs_ym, x, y, ylabel, title in panels:
        ax = fig.add_subplot(gs_main)
        ax_xm = fig.add_subplot(gs_xm, sharex=ax)
        ax_ym = fig.add_subplot(gs_ym, sharey=ax)

        # Main scatter
        ax.scatter(x, y, **scatter_kw)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=FONT_SIZES['label'])

        # Median cross-hairs
        if len(x) > 0:
            med_x, med_y = float(np.median(x)), float(np.median(y))
            ax.axvline(med_x, color=COLORS['grid'],
                       linewidth=0.8, linestyle='--', alpha=0.7)
            ax.axhline(med_y, color=COLORS['grid'],
                       linewidth=0.8, linestyle='--', alpha=0.7)

        # X margin (bottom)
        ax_xm.hist(x, range=(0, 1), **hist_kw)
        ax_xm.set_xlabel(x_label)
        ax_xm.set_yticks([])
        ax_xm.spines['left'].set_visible(False)
        plt.setp(ax.get_xticklabels(), visible=False)

        # Y margin (right)
        ax_ym.hist(y, range=(0, 1), orientation='horizontal', **hist_kw)
        ax_ym.set_xticks([])
        ax_ym.spines['bottom'].set_visible(False)
        plt.setp(ax_ym.get_yticklabels(), visible=False)

    # Suptitle
    n_perfect = int(np.sum(x_b == 1.0))
    n_zero = int(np.sum(x_b == 0.0))
    fig.suptitle(
        f'Batch scan: {len(rows_all)} genomes — '
        f'{n_perfect} all-correct, {n_zero} none-correct'
        + (f' {mode_label}' if mode_label else ''),
        fontsize=FONT_SIZES['title'], fontweight='bold',
        color=COLORS['text'], y=1.02)

    out_path = _output_path(out_path)
    _save_figure(fig, out_path, dpi=dpi)

# ---------------------------------------------------------------------------
# Single-file scanning (shared by single and batch modes)
# ---------------------------------------------------------------------------

def scan_single_file(input_path: str, model, tokenizer, calib: dict,
                     device: torch.device, host_list: list,
                     patch_nt_len: int, args,
                     ignore_idx: Optional[int] = None,
                     bact_idx: Optional[int] = None,
                     bact_threshold: float = 0.5,
                     output_plot: Optional[str] = None,
                     output_tsv: Optional[str] = None,
                     true_genus: str = '') -> list:
    """Scan one genome file and write plot + region TSV.

    If *true_genus* is provided, each summary dict will include
    ``n_correct`` (number of windows whose top-1 prediction, excluding
    bacterial_fragment when *bact_idx* is set, matches the true genus)
    and ``correct_conf_sum`` (sum of their confidences) for downstream
    aggregation.  ``n_correct_strict`` additionally requires the
    bacterial_fragment probability to be below *bact_threshold*.

    Returns list of summary dicts (one per scanned sequence).
    """
    input_base = input_path[:-3] if input_path.endswith('.gz') else input_path
    is_genbank = os.path.splitext(input_base)[1].lower() in _GENBANK_EXTENSIONS

    if is_genbank:
        gb_records = read_genbank(input_path)
        records = [{'id': r['id'], 'seq': r['seq']} for r in gb_records]
        gb_genes = {r['id']: r['genes'] for r in gb_records}
    else:
        records = read_fasta(input_path)
        gb_genes = None

    n_scan = len(records) if args.first_n == 0 else min(args.first_n, len(records))
    records_to_scan = records[:n_scan]

    # Filter out sequences shorter than one patch
    short = [r for r in records_to_scan if len(r['seq']) < patch_nt_len]
    if short:
        for r in short:
            logger.info(f"  Skipping {r['id']} ({len(r['seq']):,} nt) — "
                        f"shorter than patch length ({patch_nt_len:,} nt)")
        records_to_scan = [r for r in records_to_scan
                           if len(r['seq']) >= patch_nt_len]

    summaries = []
    all_regions = []

    # Pre-compute genus for each host_list entry for matching
    host_genera = [h.split()[0].lower() for h in host_list]
    true_genus_lower = true_genus.lower().strip()

    for ri, rec in enumerate(records_to_scan):
        seq = rec['seq']
        sid = rec['id']
        logger.info(f"  Scanning {sid} ({len(seq):,} nt) …")

        genes = None
        if gb_genes and sid in gb_genes:
            genes = gb_genes[sid]

        # Per-sequence output filenames
        if output_plot:
            if len(records_to_scan) > 1:
                stem, ext = os.path.splitext(output_plot)
                plot_path = f'{stem}_{ri}{ext}'
                tsv_path = f'{stem}_{ri}_regions.tsv'
            else:
                plot_path = output_plot
                tsv_path = (output_tsv if output_tsv
                            else os.path.splitext(output_plot)[0]
                            + '_regions.tsv')
        else:
            plot_path = None
            tsv_path = None

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

        logger.info(f"    {len(regions)} candidate prophage region(s) "
                    f"above {args.threshold:.2f} threshold")
        for reg in regions:
            logger.info(f"      {reg['start']+1:>10,}–{reg['end']:>10,}  "
                        f"({reg['length']:>7,} nt)  "
                        f"peak={reg['peak_conf']:.3f}  "
                        f"host={reg['predicted_host']}")

        if plot_path:
            plot_scan(scan, sid, len(seq), args.threshold,
                      plot_path, host_list, genes=genes,
                      ignore_idx=ignore_idx)

        if regions and tsv_path:
            write_regions_tsv(regions, sid, tsv_path)

        all_regions.extend(regions)

        # --- Per-window accuracy and confidence stats ---
        n_win = len(scan['windows'])
        all_probs = scan['all_probs']                    # (n_windows, n_classes)
        conf_sum = float(all_probs.max(axis=1).sum()) if n_win else 0.0

        n_correct = 0
        correct_conf_sum = 0.0
        n_correct_strict = 0
        correct_conf_sum_strict = 0.0
        n_bacterial = 0

        if n_win:
            # Pre-compute bacterial mask and genus-check argmax
            bact_mask = np.zeros(n_win, dtype=bool)
            if bact_idx is not None:
                bact_mask = all_probs[:, bact_idx] >= bact_threshold
                n_bacterial = int(bact_mask.sum())
                # Argmax excluding bacterial_fragment class
                probs_excl = all_probs.copy()
                probs_excl[:, bact_idx] = -1.0
                genus_pred_idx = probs_excl.argmax(axis=1)
                genus_conf = np.array([all_probs[i, gi]
                                       for i, gi in enumerate(genus_pred_idx)])
            else:
                genus_pred_idx = all_probs.argmax(axis=1)
                genus_conf = all_probs.max(axis=1)

            if true_genus_lower:
                for wi, (pi, conf) in enumerate(
                        zip(genus_pred_idx, genus_conf)):
                    if (pi < len(host_genera)
                            and host_genera[pi] == true_genus_lower):
                        n_correct += 1
                        correct_conf_sum += float(conf)
                        # Strict: also require non-bacterial
                        if not bact_mask[wi]:
                            n_correct_strict += 1
                            correct_conf_sum_strict += float(conf)

        # Summary row — raw numerics, formatted at write time
        total_region_nt = sum(r['length'] for r in regions)
        peak = max((r['peak_conf'] for r in regions), default=0.0)
        top_host = (max(regions, key=lambda r: r['peak_conf'])
                    ['predicted_host'] if regions else '')
        summaries.append({
            'genome_file': os.path.basename(input_path),
            'label': '',          # filled by caller in batch mode
            'genus': true_genus,
            'seq_id': sid,
            'seq_len': len(seq),
            'n_windows': n_win,
            'n_regions': len(regions),
            'total_region_nt': total_region_nt,
            'peak_conf': peak,
            'top_host': top_host,
            # Raw counts for aggregation
            'n_correct': n_correct,
            'correct_conf_sum': correct_conf_sum,
            'n_correct_strict': n_correct_strict,
            'correct_conf_sum_strict': correct_conf_sum_strict,
            'conf_sum': conf_sum,
            'n_bacterial': n_bacterial,
        })

    return summaries


def _safe_name(name: str, max_len: int = 80) -> str:
    """Sanitise a label or seq_id into a filesystem-safe stem."""
    return (name.replace(' ', '_').replace('/', '_')
                .replace('\\', '_'))[:max_len]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Scan bacterial genomes for candidate prophage regions '
                    'using the phage host-prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input: single file, directory, or replot from existing TSV
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument('--input', '-i', type=str, default=None,
                           help='Input FASTA or GenBank file (single mode)')
    input_grp.add_argument('--input_dir', type=str, default=None,
                           help='Directory of genome files (batch mode)')
    input_grp.add_argument('--replot', action='store_true', default=False,
                           help='Skip scanning; read an existing '
                                'batch_summary.tsv from --output and '
                                'regenerate the scatter plot only')

    parser.add_argument('--manifest', type=str, default=None,
                        help='TSV manifest inside --input_dir with a '
                             'genome_path column (like host_genome_manifest.tsv). '
                             'If omitted, globs for FASTA/GenBank files.')
    parser.add_argument('--one_per_genus', action='store_true',
                        help='In batch mode with a manifest, keep only the '
                             'first genome per genus (genus = first word of '
                             'species column)')

    parser.add_argument('--model_dir', type=str, default=None,
                        help='Model directory (contains calibration.json '
                             'and checkpoints/). Required unless --replot.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')

    # Output — always a directory
    parser.add_argument('--output', '-o', type=str, default='./scan_output',
                        help='Output directory (created if needed). '
                             'Per-genome plots and TSVs, plus batch '
                             'summaries in batch mode, are written here.')
    parser.add_argument('--per_genome_output', action='store_true',
                        help='In batch mode, also write per-genome scan '
                             'plots and region TSVs (off by default to '
                             'save time and disk). Always on in single '
                             'mode.')

    # Scan parameters
    parser.add_argument('--window_size', type=int, default=100000,
                        help='Sliding window size in nucleotides')
    parser.add_argument('--stride', type=int, default=None,
                        help='Window stride in nt (default: window_size // 2)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for prophage candidate '
                             'regions')
    parser.add_argument('--min_region', type=int, default=5000,
                        help='Minimum region size in nt to report')
    parser.add_argument('--merge_gap', type=int, default=5000,
                        help='Merge candidate regions separated by less '
                             'than this many nt')
    parser.add_argument('--first_n', type=int, default=0,
                        help='Max sequences to scan per genome file '
                             '(0 = all sequences, default)')
    parser.add_argument('--ignore_bacterial', action='store_true',
                        help='Ignore the bacterial_fragment class when '
                             'determining top predictions and labels')
    parser.add_argument('--bacterial_threshold', type=float, default=0.5,
                        help='Confidence threshold for counting a window '
                             'as bacterial_fragment in the summary')
    parser.add_argument('--strict_bacterial', action='store_true',
                        help='Use strict correct_ratio in the scatter plot: '
                             'a window counts as correct only if it predicts '
                             'the right genus AND has bacterial_fragment '
                             'below --bacterial_threshold')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--presentation', action='store_true',
                        help='Increase font sizes for presentations')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.presentation:
        enable_presentation_mode()

    # ---- Replot mode: read existing TSV and regenerate scatter only ----------
    if args.replot:
        summary_path = os.path.join(args.output, 'batch_summary.tsv')
        if not os.path.exists(summary_path):
            logger.error(f"No batch_summary.tsv found in {args.output}")
            return
        all_summaries = read_batch_summary_tsv(summary_path)
        scatter_path = os.path.join(args.output, 'batch_scatter.png')
        plot_batch_scatter(all_summaries, scatter_path,
                           strict=args.strict_bacterial)
        logger.info("Replot done.")
        return

    # ---- Validate model_dir for scanning mode --------------------------------
    if not args.model_dir:
        parser.error("--model_dir is required unless --replot is set.")

    if args.stride is None:
        args.stride = args.window_size // 2

    # ---- Create output directory --------------------------------------------
    os.makedirs(args.output, exist_ok=True)

    # ---- Load model ---------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model, calib = load_model_and_calibration(
        args.model_dir, args.checkpoint, device)

    patch_nt_len = calib['model_config']['patch_nt_len']
    host_list = [h.split(';')[-1]
                 for h in calib.get('host_list', calib.get('hosts', []))]

    # Detect bacterial_fragment index (for counting bacterial windows)
    bact_idx = None
    try:
        bact_idx = host_list.index('bacterial_fragment')
        logger.info(f"  bacterial_fragment class detected (index {bact_idx}, "
                    f"threshold {args.bacterial_threshold:.2f})")
    except ValueError:
        logger.info("  No bacterial_fragment class found — "
                    "bacterial_ratio will be 0")

    # Optionally ignore bacterial_fragment for top-prediction labels
    ignore_idx = None
    if args.ignore_bacterial:
        ignore_idx = bact_idx
        if ignore_idx is not None:
            logger.info(f"  Ignoring bacterial_fragment class for "
                        f"top predictions")
        else:
            logger.info("  --ignore_bacterial set but class not found, "
                        "nothing to ignore")

    tokenizer = CodonTokenizer()

    # ==================================================================
    # Batch mode
    # ==================================================================
    if args.input_dir:
        genome_entries = collect_genome_files(
            args.input_dir, args.manifest,
            one_per_genus=args.one_per_genus)
        n_genomes = len(genome_entries)
        logger.info(f"Batch mode: {n_genomes} genome files "
                    f"from {args.input_dir}")

        if n_genomes == 0:
            logger.warning("No genome files found — nothing to do.")
            return

        logger.info(f"  Output directory: {args.output}")
        if not args.per_genome_output:
            logger.info("  Per-genome plots/TSVs disabled "
                        "(use --per_genome_output to enable)")

        all_summaries = []
        total_regions = 0

        for gi, entry in enumerate(genome_entries):
            gpath = entry['path']
            glabel = entry['label']
            ggenus = entry.get('genus', '')
            safe_label = _safe_name(glabel)

            logger.info(f"[{gi+1}/{n_genomes}] {glabel} — {gpath}")

            if args.per_genome_output:
                plot_path = os.path.join(args.output,
                                         f'{safe_label}_scan.png')
                tsv_path = os.path.join(args.output,
                                        f'{safe_label}_regions.tsv')
            else:
                plot_path = None
                tsv_path = None

            try:
                seq_summaries = scan_single_file(
                    gpath, model, tokenizer, calib, device,
                    host_list, patch_nt_len, args,
                    ignore_idx=ignore_idx,
                    bact_idx=bact_idx,
                    bact_threshold=args.bacterial_threshold,
                    output_plot=plot_path,
                    output_tsv=tsv_path,
                    true_genus=ggenus,
                )
            except Exception as exc:
                logger.error(f"  Failed on {gpath}: {exc}")
                continue

            if not seq_summaries:
                continue

            for s in seq_summaries:
                s['label'] = glabel

            # Aggregate all sequences in this file into one row
            agg = aggregate_file_summary(seq_summaries)
            all_summaries.append(agg)
            total_regions += agg['n_regions']

            # Log aggregated scatter values
            cr_str = (f"{agg['correct_ratio']:.4f}"
                      if agg['correct_ratio'] is not None else 'n/a')
            crs_str = (f"{agg['correct_ratio_strict']:.4f}"
                       if agg['correct_ratio_strict'] is not None else 'n/a')
            mcc_str = (f"{agg['mean_correct_conf']:.4f}"
                       if agg['mean_correct_conf'] is not None else 'n/a')
            mac_str = (f"{agg['mean_all_conf']:.4f}"
                       if agg['mean_all_conf'] is not None else 'n/a')
            logger.info(
                f"    {agg['n_sequences']} seq, "
                f"{agg['n_windows']} windows, "
                f"{agg['n_regions']} regions — "
                f"correct_ratio={cr_str} (strict={crs_str}), "
                f"mean_correct_conf={mcc_str}, "
                f"mean_all_conf={mac_str}, "
                f"bacterial_ratio={agg['bacterial_ratio']:.4f}")

            # Progress logging every 100 genomes
            n_done = gi + 1
            if n_done % 100 == 0:
                logger.info(f"  ---- Progress: {n_done}/{n_genomes} "
                            f"genomes processed, {total_regions} total "
                            f"candidate regions so far ----")

        # Final summary TSV
        summary_path = os.path.join(args.output, 'batch_summary.tsv')
        write_batch_summary_tsv(all_summaries, summary_path)

        # Scatter plot: correct-window ratio vs mean top confidence
        scatter_path = os.path.join(args.output, 'batch_scatter.png')
        plot_batch_scatter(all_summaries, scatter_path,
                           strict=args.strict_bacterial)

        logger.info(f"Done. {n_genomes} genomes processed, "
                    f"{total_regions} total candidate region(s).")
        return

    # ==================================================================
    # Single-file mode
    # ==================================================================
    logger.info(f"Reading {args.input} …")

    input_stem = _safe_name(
        os.path.splitext(os.path.basename(args.input))[0])
    plot_path = os.path.join(args.output, f'{input_stem}_scan.png')
    tsv_path = os.path.join(args.output, f'{input_stem}_regions.tsv')

    summaries = scan_single_file(
        args.input, model, tokenizer, calib, device,
        host_list, patch_nt_len, args,
        ignore_idx=ignore_idx,
        bact_idx=bact_idx,
        bact_threshold=args.bacterial_threshold,
        output_plot=plot_path,
        output_tsv=tsv_path,
    )

    total = sum(s['n_regions'] for s in summaries)
    n_scan = len(summaries)
    logger.info(f"Done. {total} total candidate region(s) across "
                f"{n_scan} sequence(s).")


if __name__ == '__main__':
    main()
