#!/usr/bin/env python3
"""Inference script for the hierarchical DNA classifier.

Loads a trained model checkpoint + calibration.json, processes FASTA input,
and outputs per-sequence predictions as TSV.  If hosts.csv is present in the
run directory, taxonomic lineage columns are appended automatically.

Usage:
    python predict.py --input phages.fasta --run_dir ./models/HierDNA
    python predict.py --input phages.fna.gz --run_dir ./models/HierDNA \
        --fdr 0.1 --top_k 5

Expected run_dir contents:
    calibration.json   — temperature, thresholds, model config, host list
    checkpoints/       — model checkpoint(s)
    hosts.csv          — (optional) phylum,class,order,family,genus
"""

import argparse
import csv
import gzip
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from Bio import SeqIO

from .model import HierarchicalDNAClassifier, CodonTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_fasta(path: str) -> List[dict]:
    """Read plain or gzipped FASTA. Returns list of {id, description, seq}."""
    records = []
    opener = gzip.open if path.endswith('.gz') else open
    mode = 'rt'
    with opener(path, mode) as fh:
        for rec in SeqIO.parse(fh, 'fasta'):
            records.append({
                'id': rec.id,
                'description': rec.description,
                'seq': str(rec.seq),
            })
    return records


def load_taxonomy(path: str) -> tuple:
    """Load hosts.csv -> ({genus: {rank: value, ...}}, [rank_columns]).

    Auto-detects delimiter (comma or tab).  Columns are expected in
    taxonomic order (e.g. phylum, class, order, family, genus).
    The 'genus' and 'species' columns are excluded from the lineage ranks.
    Returns (tax_dict, ordered_rank_list).
    """
    with open(path) as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = '\t' if sample.count('\t') > sample.count(',') else ','
        reader = csv.DictReader(fh, delimiter=delimiter)
        reader.fieldnames = [f.strip().lower() for f in reader.fieldnames]
        skip = {'genus', 'species'}
        ranks = [c for c in reader.fieldnames if c not in skip]
        tax = {}
        for row in reader:
            genus = row.get('genus', '').strip()
            if genus:
                tax[genus] = {k: v.strip() for k, v in row.items()
                              if k not in skip and v and v.strip()}
    return tax, ranks


# ---------------------------------------------------------------------------
# Sequence tiling (deterministic, same as eval in train.py)
# ---------------------------------------------------------------------------

def tile_sequence(seq: str, patch_nt_len: int, stride: int,
                  max_patches: int = 512) -> List[str]:
    plen = patch_nt_len
    patches = []
    for start in range(0, max(1, len(seq) - plen + 1), stride):
        patches.append(seq[start:start + plen])
        if len(patches) >= max_patches:
            break
    if len(seq) > plen:
        last = seq[len(seq) - plen:]
        if not patches or last != patches[-1]:
            patches.append(last)
    if not patches:
        patches.append(seq)
    return patches[:max_patches]


def tokenize_patches(patches: List[str], tokenizer: CodonTokenizer) -> tuple:
    """Tokenize + pad patches -> (1, N, 6, L) tensor + (1,) count."""
    toks = [tokenizer.tokenize(p) for p in patches]
    max_cl = max(t.size(1) for t in toks)
    padded = torch.zeros(len(toks), 6, max_cl, dtype=torch.long)
    for i, t in enumerate(toks):
        padded[i, :, :t.size(1)] = t
    return padded.unsqueeze(0), torch.tensor([len(toks)], dtype=torch.long)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_calibration(run_dir: str, checkpoint: Optional[str],
                                device: torch.device):
    """Load calibration.json, build model, load weights. Returns (model, calib)."""
    calib_path = os.path.join(run_dir, 'calibration.json')
    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"calibration.json not found in {run_dir}. "
            f"Run train.py --calibrate_only first.")

    with open(calib_path) as f:
        calib = json.load(f)

    cfg = calib['model_config']
    model = HierarchicalDNAClassifier(
        **cfg
    ).to(device)

    # Find checkpoint
    if checkpoint:
        ckpt_path = checkpoint
    else:
        ckpt_dir = os.path.join(run_dir, 'checkpoints')
        for name in ['best_aggregator.pt', 'last_aggregator.pt']:
            p = os.path.join(ckpt_dir, name)
            if os.path.exists(p):
                ckpt_path = p
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {ckpt_dir}. Specify --checkpoint.")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    logger.info(f"Temperature: {calib['temperature']:.4f}")

    return model, calib


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_sequence(model, tokenizer, seq: str, patch_nt_len: int,
                     stride: int, temperature: float,
                     device: torch.device,
                     max_patches: int = 512,
                     blocked_classes: Optional[List[int]] = None) -> np.ndarray:
    """Run model on one sequence, return calibrated probabilities (C,)."""
    patches = tile_sequence(seq, patch_nt_len, stride, max_patches)
    tokens, counts = tokenize_patches(patches, tokenizer)
    tokens = tokens.to(device, non_blocking=True)
    counts = counts.to(device, non_blocking=True)

    logits = model(tokens, counts)                  # (1, C)
    if blocked_classes:
        logits[:, blocked_classes] = -1e9
    probs = torch.sigmoid(logits / temperature)     # calibrated
    return probs[0].cpu().numpy()


@torch.no_grad()
def predict_batch(model, tokenizer, seqs: List[str], patch_nt_len: int,
                  stride: int, temperature: float, device: torch.device,
                  max_patches: int = 512,
                  blocked_classes: Optional[List[int]] = None) -> np.ndarray:
    """Batch prediction for multiple sequences. Returns (B, C) probabilities."""
    all_patches, all_counts = [], []
    for seq in seqs:
        patches = tile_sequence(seq, patch_nt_len, stride, max_patches)
        toks = [tokenizer.tokenize(p) for p in patches]
        all_patches.append(toks)
        all_counts.append(len(toks))

    # Pad to uniform shape
    max_n = max(all_counts)
    max_cl = max(t.size(1) for toks in all_patches for t in toks)
    B = len(seqs)
    batch_tensor = torch.zeros(B, max_n, 6, max_cl, dtype=torch.long)
    for i, toks in enumerate(all_patches):
        for j, t in enumerate(toks):
            batch_tensor[i, j, :, :t.size(1)] = t

    batch_tensor = batch_tensor.to(device, non_blocking=True)
    counts = torch.tensor(all_counts, dtype=torch.long, device=device)

    logits = model(batch_tensor, counts)
    if blocked_classes:
        logits[:, blocked_classes] = -1e9
    probs = torch.sigmoid(logits / temperature)
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_results(seq_id: str, probs: np.ndarray, hosts: List[str],
                   threshold: float, top_k: int,
                   taxonomy: Optional[Dict] = None,
                   lineage_ranks: Optional[List[str]] = None) -> List[dict]:
    """Format predictions for one sequence as list of output rows.

    Returns only predictions above threshold, sorted by score descending.
    If top_k > 0, caps output to top_k entries.
    If nothing is above threshold, returns the single best prediction
    (marked above_threshold=no).
    """
    above = np.where(probs >= threshold)[0]

    if len(above) == 0:
        # Nothing above threshold — return single best prediction
        above = np.array([np.argmax(probs)])

    # Sort by score descending
    above = above[np.argsort(probs[above])[::-1]]

    # Cap if top_k requested
    if top_k > 0:
        above = above[:top_k]

    rows = []
    for idx in above:
        genus = hosts[idx]
        score = float(probs[idx])
        row = {
            'sequence_id': seq_id,
            'genus': genus,
            'score': f"{score:.4f}",
            'above_threshold': 'yes' if score >= threshold else 'no',
        }
        if taxonomy and lineage_ranks:
            lineage = taxonomy.get(genus, {})
            parts = [lineage.get(r, '') for r in lineage_ranks]
            row['lineage'] = ';'.join(parts)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Predict phage hosts using trained hierarchical DNA model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input FASTA file (plain or .gz)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output TSV file (default: stdout)')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Training run directory (contains calibration.json)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold (default: from calibration.json)')
    parser.add_argument('--fdr', type=float, default=None,
                        help='Use FDR-calibrated threshold (e.g. 0.1 for 10%% FDR). '
                             'Overrides --threshold.')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Max predictions per sequence (0 = all above threshold)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Tiling stride in nt (default: patch_nt_len * 2/3)')
    parser.add_argument('--max_patches', type=int, default=512,
                        help='Max patches per sequence')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Sequences per batch (1 = sequential, saves memory)')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # ---- load model ------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model, calib = load_model_and_calibration(
        args.run_dir, args.checkpoint, device)

    hosts = calib['hosts']
    temperature = calib['temperature']
    blocked_classes = calib.get('blocked_classes', [])
    patch_nt_len = calib['model_config']['patch_nt_len']
    stride = args.stride or int(patch_nt_len * 2/3)
    top_k = args.top_k

    # Resolve threshold: --fdr > --threshold > calibration.json default
    if args.fdr is not None:
        fdr_key = f"fdr_{int(args.fdr * 100):02d}"
        fdr_thresholds = calib.get('fdr_thresholds', {})
        if fdr_key in fdr_thresholds:
            threshold = fdr_thresholds[fdr_key]
            logger.info(f"Using FDR {args.fdr*100:.0f}% threshold: {threshold:.4f}")
        else:
            available = list(fdr_thresholds.keys())
            logger.error(f"FDR threshold '{fdr_key}' not found in calibration. "
                         f"Available: {available}")
            sys.exit(1)
    elif args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = calib.get('threshold', 0.5)

    logger.info(f"Classes: {len(hosts)}  threshold: {threshold:.4f}  "
                f"patch_len: {patch_nt_len}  stride: {stride}")
    if blocked_classes:
        logger.info(f"  {len(blocked_classes)} blocked classes (low val precision)")

    # ---- taxonomy (hosts.csv in run_dir) ------------------------------------
    taxonomy = None
    lineage_ranks = []
    hosts_csv = os.path.join(args.run_dir, 'hosts.csv')
    if os.path.exists(hosts_csv):
        taxonomy, lineage_ranks = load_taxonomy(hosts_csv)
        logger.info(f"Loaded taxonomy for {len(taxonomy)} genera "
                    f"(ranks: {', '.join(lineage_ranks)})")
    else:
        logger.info(f"No hosts.csv in {args.run_dir} — skipping lineage")

    # ---- read input ------------------------------------------------------
    logger.info(f"Reading {args.input} …")
    records = read_fasta(args.input)
    logger.info(f"  {len(records)} sequences")

    # ---- prepare output --------------------------------------------------
    fieldnames = ['sequence_id', 'genus', 'score', 'above_threshold']
    if taxonomy:
        fieldnames.append('lineage')

    out_fh = open(args.output, 'w', newline='') if args.output else sys.stdout
    writer = csv.DictWriter(out_fh, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    # ---- predict ---------------------------------------------------------
    tokenizer = CodonTokenizer()

    if args.batch_size <= 1:
        for i, rec in enumerate(records):
            probs = predict_sequence(
                model, tokenizer, rec['seq'], patch_nt_len, stride,
                temperature, device, args.max_patches, blocked_classes)
            rows = format_results(
                rec['id'], probs, hosts, threshold, top_k,
                taxonomy, lineage_ranks)
            for row in rows:
                writer.writerow(row)

            if (i + 1) % 100 == 0:
                logger.info(f"  processed {i+1}/{len(records)}")
    else:
        for start in range(0, len(records), args.batch_size):
            batch_recs = records[start:start + args.batch_size]
            seqs = [r['seq'] for r in batch_recs]
            all_probs = predict_batch(
                model, tokenizer, seqs, patch_nt_len, stride,
                temperature, device, args.max_patches, blocked_classes)
            for rec, probs in zip(batch_recs, all_probs):
                rows = format_results(
                    rec['id'], probs, hosts, threshold, top_k,
                    taxonomy, lineage_ranks)
                for row in rows:
                    writer.writerow(row)

            done = min(start + args.batch_size, len(records))
            if done % 100 == 0 or done == len(records):
                logger.info(f"  processed {done}/{len(records)}")

    if args.output:
        out_fh.close()
        logger.info(f"Output written to {args.output}")

    logger.info("Done.")


if __name__ == '__main__':
    main()
