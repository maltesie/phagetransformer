#!/usr/bin/env python3
"""Protein importance via scrambling — single-genome / multi-contig version.

Reads a FASTA and a phold per-CDS annotation table, then measures the effect
of scrambling each annotated protein on the PhageTransformer host prediction.
Two scramble types per protein:

  nucleotide — shuffle individual nucleotides within the CDS
  codon      — shuffle whole codons (preserving reading frame)

Both modes preserve start and stop codons (first/last 3 nt).

Usage:
    python importance_single.py \
        --fasta phage.fna \
        --phold phold_per_cds_predictions.tsv \
        --model_dir ./models/PT \
        --output importance_deltas.tsv
"""

import argparse
import csv
import gzip
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

# ---------------------------------------------------------------------------
# Imports from the package
# ---------------------------------------------------------------------------
try:
    from phagetransformer.model import CodonTokenizer
    from phagetransformer.predict import (
        load_model_and_calibration, predict_sequence,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model import CodonTokenizer
    from predict import load_model_and_calibration, predict_sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phold annotation loader
# ---------------------------------------------------------------------------

def load_phold_annotations(tsv_path: str) -> Dict[str, List[dict]]:
    """Parse a phold per-CDS predictions TSV.

    Returns {contig_id: [protein dicts]} where each protein dict has:
        cds_id, begin, end, strand, length_nt, category, product
    """
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)

    # --- Resolve column names (phold versions may vary) --------------------
    def _find_col(candidates, label):
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"Cannot find {label} column in {tsv_path}. "
                         f"Columns: {list(df.columns)}")

    def _find_col_optional(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    contig_col = _find_col(['contig', 'Contig', 'contig_id'], 'contig')
    cds_col    = _find_col(['cds_id', 'CDS_ID', 'gene', 'Gene', 'protein_id'],
                           'CDS ID')
    start_col  = _find_col(['start', 'Start'], 'start')
    end_col    = _find_col(['end', 'End', 'stop', 'Stop'], 'end')
    strand_col = _find_col_optional(['strand', 'Strand', 'frame', 'Frame'])
    cat_col    = _find_col_optional(['category', 'Category', 'phrog_category',
                                     'annot_category', 'function'])
    prod_col   = _find_col_optional(['product', 'Product', 'annot'])

    annotations = defaultdict(list)
    for _, row in df.iterrows():
        contig = row[contig_col]
        start = int(float(row[start_col]))
        end = int(float(row[end_col]))

        if strand_col and pd.notna(row.get(strand_col, '')):
            s = str(row[strand_col])
            strand = -1 if s.startswith('-') or s == '-1' else 1
        else:
            strand = 1 if start < end else -1

        begin = min(start, end)
        stop = max(start, end)

        category = (row[cat_col].strip()
                    if cat_col and pd.notna(row.get(cat_col)) else
                    'unknown function')
        product = (row[prod_col].strip()
                   if prod_col and pd.notna(row.get(prod_col)) else '')
        cds_id = (row[cds_col].strip()
                  if pd.notna(row.get(cds_col)) else '')

        annotations[contig].append({
            'cds_id':    cds_id,
            'begin':     begin,
            'end':       stop,
            'strand':    strand,
            'length_nt': stop - begin + 1,
            'category':  category,
            'product':   product,
        })

    logger.info(f"Loaded phold annotations for {len(annotations)} contigs, "
                f"{sum(len(v) for v in annotations.values())} CDS total")
    return dict(annotations)


# ---------------------------------------------------------------------------
# Scrambling functions (from importance.py, self-contained)
# ---------------------------------------------------------------------------

def scramble_nucleotides(seq: str, start: int, end: int,
                         rng: random.Random) -> str:
    """Shuffle individual nucleotides in seq[start:end] (0-based half-open).

    Preserves the first and last 3 nt (start/stop codons).
    """
    if end - start < 6:
        return seq
    region = list(seq[start + 3:end - 3])
    rng.shuffle(region)
    return seq[:start + 3] + ''.join(region) + seq[end - 3:end] + seq[end:]


def scramble_codons(seq: str, start: int, end: int, strand: int,
                    rng: random.Random) -> str:
    """Shuffle whole codons within the protein region.

    Codon boundaries aligned from the left (+strand) or right (-strand).
    First and last codon are preserved (start/stop codons).
    """
    region = seq[start:end]
    if strand == -1:
        offset = len(region) % 3
        prefix = region[:offset]
        codons = [region[i:i+3] for i in range(offset, len(region) - 2, 3)]
        if len(codons) >= 4:
            first, last = codons[0], codons[-1]
            middle = codons[1:-1]
            rng.shuffle(middle)
            codons = [first] + middle + [last]
        else:
            rng.shuffle(codons)
        return seq[:start] + prefix + ''.join(codons) + seq[end:]
    else:
        codons = [region[i:i+3] for i in range(0, len(region) - 2, 3)]
        tail = region[len(codons) * 3:]
        if len(codons) >= 4:
            first, last = codons[0], codons[-1]
            middle = codons[1:-1]
            rng.shuffle(middle)
            codons = [first] + middle + [last]
        else:
            rng.shuffle(codons)
        return seq[:start] + ''.join(codons) + tail + seq[end:]


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def get_scores(model, tokenizer, seq: str, patch_nt_len: int,
               stride: int, temperature: float,
               device: torch.device) -> Optional[np.ndarray]:
    """Run prediction, return calibrated probabilities or None."""
    if not isinstance(seq, str):
        seq = str(seq)
    seq = seq[:len(seq) // 3 * 3]
    if len(seq) < 300:
        return None
    return predict_sequence(model, tokenizer, seq, patch_nt_len, stride,
                            temperature, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Protein importance via scrambling (lightweight version)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fasta', required=True,
                        help='Input FASTA (plain or gzipped, multi-contig OK)')
    parser.add_argument('--phold', required=True,
                        help='Phold per-CDS predictions TSV')
    parser.add_argument('--model_dir', required=True,
                        help='Model directory with calibration.json + '
                             'checkpoints/')
    parser.add_argument('--checkpoint', default=None,
                        help='Specific checkpoint path (default: auto-detect)')
    parser.add_argument('--output', '-o', default='importance_deltas.tsv',
                        help='Output TSV path')
    parser.add_argument('--min_score', type=float, default=0.3,
                        help='Min baseline top-class score to process a '
                             'contig')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    rng = random.Random(args.seed)

    # ---- Load model -------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model, calib = load_model_and_calibration(
        args.model_dir, args.checkpoint, device)

    patch_nt_len = calib['model_config']['patch_nt_len']
    stride = patch_nt_len
    temperature = calib.get('temperature', 1.0)
    host_names = calib['hosts']
    tokenizer = CodonTokenizer()

    logger.info(f"Model loaded: {len(host_names)} classes, "
                f"patch_nt_len={patch_nt_len}, device={device}")

    # ---- Load phold annotations -------------------------------------------
    phold = load_phold_annotations(args.phold)

    # ---- Load FASTA and process -------------------------------------------
    opener = gzip.open if args.fasta.endswith('.gz') else open
    records = []
    with opener(args.fasta, 'rt') as fh:
        for rec in SeqIO.parse(fh, 'fasta'):
            records.append((rec.id, str(rec.seq)))
    logger.info(f"Loaded {len(records)} contigs from {args.fasta}")

    # ---- Main loop --------------------------------------------------------
    results = []
    skipped_short = 0
    skipped_low = 0
    skipped_noanno = 0

    for ci, (contig_id, seq) in enumerate(records):
        logger.info(f"  [{ci+1}/{len(records)}] {contig_id} ({len(seq):,} nt)")

        # Match contig to phold annotations (try original + sanitized ID)
        phold_key = None
        if contig_id in phold:
            phold_key = contig_id
        else:
            sanitized = contig_id.replace(':', '_')
            if sanitized in phold:
                phold_key = sanitized

        if phold_key is None:
            logger.warning(f"    No phold annotations for {contig_id}, "
                           f"skipping")
            skipped_noanno += 1
            continue

        proteins = phold[phold_key]

        # Baseline prediction
        baseline = get_scores(model, tokenizer, seq, patch_nt_len,
                              stride, temperature, device)
        if baseline is None:
            logger.warning(f"    Sequence too short, skipping")
            skipped_short += 1
            continue

        top_class_idx = int(np.argmax(baseline))
        top_score = float(baseline[top_class_idx])
        top_class = host_names[top_class_idx]

        if top_score < args.min_score:
            logger.info(f"    Baseline score {top_score:.3f} < "
                        f"{args.min_score}, skipping")
            skipped_low += 1
            continue

        logger.info(f"    Baseline: {top_class} ({top_score:.3f}), "
                    f"{len(proteins)} proteins")

        # Scramble each protein
        for prot in proteins:
            p_start = prot['begin'] - 1   # 0-based half-open
            p_end = prot['end']

            # Nucleotide scramble
            seq_nt = scramble_nucleotides(seq, p_start, p_end, rng)
            scores_nt = get_scores(model, tokenizer, seq_nt, patch_nt_len,
                                   stride, temperature, device)
            delta_nt = (float(scores_nt[top_class_idx]) - top_score
                        if scores_nt is not None else float('nan'))

            # Codon scramble
            seq_cd = scramble_codons(seq, p_start, p_end,
                                     prot['strand'], rng)
            scores_cd = get_scores(model, tokenizer, seq_cd, patch_nt_len,
                                   stride, temperature, device)
            delta_cd = (float(scores_cd[top_class_idx]) - top_score
                        if scores_cd is not None else float('nan'))

            results.append({
                'protein_id':     prot['cds_id'],
                'contig':         contig_id,
                'start':          prot['begin'],
                'end':            prot['end'],
                'strand':         prot['strand'],
                'length_nt':      prot['length_nt'],
                'phrog_category': prot['category'],
                'phrog_product':  prot['product'],
                'baseline_top':   top_score,
                'top_class':      top_class,
                'delta_nt':       delta_nt,
                'delta_codon':    delta_cd,
            })

    # ---- Write output -----------------------------------------------------
    logger.info(f"Done. {len(results)} proteins scored across "
                f"{len(records) - skipped_short - skipped_low - skipped_noanno}"
                f" contigs.")
    if skipped_noanno:
        logger.info(f"  {skipped_noanno} contigs skipped (no phold match)")
    if skipped_short:
        logger.info(f"  {skipped_short} contigs skipped (too short)")
    if skipped_low:
        logger.info(f"  {skipped_low} contigs skipped (low baseline score)")

    if not results:
        logger.warning("No results to write.")
        return

    fieldnames = ['protein_id', 'contig', 'start', 'end', 'strand',
                  'length_nt', 'phrog_category', 'phrog_product',
                  'baseline_top', 'top_class', 'delta_nt', 'delta_codon']

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
