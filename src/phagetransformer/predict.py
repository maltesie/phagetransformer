#!/usr/bin/env python3
"""Inference script for the hierarchical DNA classifier.

Loads a trained model checkpoint + calibration.json, processes FASTA input,
and outputs per-sequence predictions as TSV.

Output columns:
    sequence_id              — FASTA record ID
    genus                    — predicted host genus (last component of lineage)
    lineage                  — full taxonomic lineage (Phylum;Class;Order;Family;Genus)
    score                    — calibrated prediction score for this genus
    bacterial_score          — score for the bacterial_fragment class (if present)
    above_host_threshold     — 'yes' if score >= host threshold
    above_bacterial_threshold — 'yes' if bacterial_score >= bacterial threshold

Host threshold:
    --threshold sets a fixed score threshold.
    --fdr (default 0.1) uses the FDR-calibrated threshold from calibration.json.
    --threshold takes priority over --fdr when both are given.

With --filter_output, only rows that are above the host threshold AND
below the bacterial threshold are written — i.e., confident phage-host
predictions that are not flagged as bacterial.

Usage:
    phagetransformer predict --input phages.fasta --model_dir ./models/PT
    phagetransformer predict --input phages.fasta --model_dir ./models/PT \
        --fdr 0.2 --top_k 5
    phagetransformer predict --input phages.fasta --model_dir ./models/PT \
        --threshold 0.3 --bacterial_threshold 0.4
    phagetransformer predict --input phages.fasta --model_dir ./models/PT \
        --filter_output

Expected model_dir contents:
    calibration.json   — temperature, thresholds, model config, host list
    checkpoints/       — model checkpoint(s)
"""

import argparse
import csv
import gzip
import json
import logging
import os
import sys
from typing import List, Optional

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



# ---------------------------------------------------------------------------
# Sequence tiling (deterministic, same as eval in train.py)
# ---------------------------------------------------------------------------

def tile_sequence(seq: str, patch_nt_len: int, stride: int,
                  max_patches: int = 512) -> tuple:
    """Tile a sequence into overlapping patches.

    Returns (patches, starts) where patches is a list of subsequence
    strings and starts is a list of 0-based nucleotide start positions.
    """
    plen = patch_nt_len
    patches = []
    starts = []
    for start in range(0, max(1, len(seq) - plen + 1), stride):
        patches.append(seq[start:start + plen])
        starts.append(start)
        if len(patches) >= max_patches:
            break
    if len(seq) > plen:
        last_start = len(seq) - plen
        last = seq[last_start:]
        if not patches or last != patches[-1]:
            patches.append(last)
            starts.append(last_start)
    if not patches:
        patches.append(seq)
        starts.append(0)
    patches = patches[:max_patches]
    starts = starts[:max_patches]
    return patches, starts


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

def load_model_and_calibration(model_dir: str, checkpoint: Optional[str],
                                device: torch.device):
    """Load calibration.json, build model, load weights. Returns (model, calib)."""
    calib_path = os.path.join(model_dir, 'calibration.json')
    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"calibration.json not found in {model_dir}. "
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
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
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
    if 'temperature_host' in calib and 'temperature_bacterial' in calib:
        logger.info(f"Temperature: host={calib['temperature_host']:.4f}  "
                    f"bacterial={calib['temperature_bacterial']:.4f}")
    else:
        logger.info(f"Temperature: {calib['temperature']:.4f}")

    return model, calib


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_sequence(model, tokenizer, seq: str, patch_nt_len: int,
                     stride: int, temperature,
                     device: torch.device,
                     max_patches: int = 512,
                     blocked_classes: Optional[List[int]] = None) -> np.ndarray:
    """Run model on one sequence, return calibrated probabilities (C,).

    ``temperature`` can be a scalar float or a (C,) tensor for per-class
    temperatures (e.g. separate host / bacterial temperatures).
    """
    patches, _ = tile_sequence(seq, patch_nt_len, stride, max_patches)
    tokens, counts = tokenize_patches(patches, tokenizer)
    tokens = tokens.to(device, non_blocking=True)
    counts = counts.to(device, non_blocking=True)

    logits = model(tokens, counts)                  # (1, C)
    if blocked_classes:
        logits[:, blocked_classes] = -1e9
    if torch.is_tensor(temperature):
        temperature = temperature.to(logits.device)
    probs = torch.sigmoid(logits / temperature)     # calibrated
    return probs[0].cpu().numpy()


@torch.no_grad()
def predict_batch(model, tokenizer, seqs: List[str], patch_nt_len: int,
                  stride: int, temperature, device: torch.device,
                  max_patches: int = 512,
                  blocked_classes: Optional[List[int]] = None) -> np.ndarray:
    """Batch prediction for multiple sequences. Returns (B, C) probabilities.

    ``temperature`` can be a scalar float or a (C,) tensor for per-class
    temperatures (e.g. separate host / bacterial temperatures).
    """
    all_patches, all_counts = [], []
    for seq in seqs:
        patches, _ = tile_sequence(seq, patch_nt_len, stride, max_patches)
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
    if torch.is_tensor(temperature):
        temperature = temperature.to(logits.device)
    probs = torch.sigmoid(logits / temperature)
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_results(seq_id: str, probs: np.ndarray, hosts: List[str],
                   threshold: float, bacterial_threshold: float,
                   top_k: int) -> List[dict]:
    """Format predictions for one sequence as list of output rows.

    Returns only genus predictions above threshold, sorted by score
    descending.  If top_k > 0, caps output to top_k entries.
    If nothing is above threshold, returns the single best prediction
    (marked above_host_threshold=no).

    The ``bacterial_fragment`` class (if present as the last host) is
    reported separately as ``bacterial_score`` on every row, not as a
    genus prediction.  ``above_bacterial_threshold`` indicates whether
    the bacterial score exceeds ``bacterial_threshold``.
    """
    # Separate bacterial_fragment from genus predictions
    has_bact = len(hosts) > 0 and hosts[-1] == 'bacterial_fragment'
    if has_bact:
        bact_score = float(probs[-1])
        above_bact = bact_score >= bacterial_threshold
        genus_probs = probs[:-1]
        genus_hosts = hosts[:-1]
    else:
        bact_score = None
        above_bact = False
        genus_probs = probs
        genus_hosts = hosts

    above = np.where(genus_probs >= threshold)[0]

    if len(above) == 0:
        # Nothing above threshold — return single best prediction
        above = np.array([np.argmax(genus_probs)])

    # Sort by score descending
    above = above[np.argsort(genus_probs[above])[::-1]]

    # Cap if top_k requested
    if top_k > 0:
        above = above[:top_k]

    rows = []
    for idx in above:
        host = genus_hosts[idx]
        genus = host.split(';')[-1] if ';' in host else host
        score = float(genus_probs[idx])
        row = {
            'sequence_id': seq_id,
            'genus': genus,
            'lineage': host,
            'score': f"{score:.4f}",
            'bacterial_score': f"{bact_score:.4f}" if bact_score is not None else '',
            'above_host_threshold': 'yes' if score >= threshold else 'no',
            'above_bacterial_threshold': 'yes' if above_bact else 'no',
        }
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
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory (contains calibration.json and checkpoints/)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed score threshold (overrides --fdr)')
    parser.add_argument('--fdr', type=float, default=0.1,
                        help='FDR level for threshold from calibration '
                             '(e.g. 0.1 for 10%% FDR, 0.2 for 20%%)')
    parser.add_argument('--bacterial_threshold', type=float, default=0.5,
                        help='Score threshold for bacterial_fragment class')
    parser.add_argument('--filter_output', action='store_true',
                        help='Only report predictions above the host '
                             'threshold and below the bacterial threshold')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Max predictions per sequence (0 = all above threshold)')
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
        args.model_dir, args.checkpoint, device)

    hosts = calib['hosts']
    blocked_classes = calib.get('blocked_classes', [])
    patch_nt_len = calib['model_config']['patch_nt_len']
    stride = calib.get('eval_stride') or patch_nt_len // 2
    top_k = args.top_k

    # Build per-class temperature: use split host/bacterial temperatures
    # when available, otherwise fall back to the single scalar.
    n_classes = calib['model_config']['num_classes']
    has_bact = len(hosts) > 0 and hosts[-1] == 'bacterial_fragment'
    if calib.get('temperature_host') is not None and has_bact:
        T_host = calib['temperature_host']
        T_bact = calib.get('temperature_bacterial', T_host)
        n_host = n_classes - 1
        temperature = torch.ones(n_classes)
        temperature[:n_host] = T_host
        temperature[n_host:] = T_bact
    else:
        temperature = calib['temperature']

    # Resolve threshold: --threshold overrides --fdr
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Using fixed threshold: {threshold:.4f}")
    else:
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

    logger.info(f"Classes: {len(hosts)}  threshold: {threshold:.4f}  "
                f"bacterial_threshold: {args.bacterial_threshold:.4f}  "
                f"patch_len: {patch_nt_len}  stride: {stride}")
    if args.filter_output:
        logger.info(f"  --filter_output: only reporting rows above host "
                    f"threshold and below bacterial threshold")
    if blocked_classes:
        logger.info(f"  {len(blocked_classes)} blocked classes (low val precision)")

    # ---- read input ------------------------------------------------------
    logger.info(f"Reading {args.input} …")
    records = read_fasta(args.input)
    logger.info(f"  {len(records)} sequences")

    # ---- prepare output --------------------------------------------------
    fieldnames = ['sequence_id', 'genus', 'lineage', 'score',
                  'bacterial_score', 'above_host_threshold',
                  'above_bacterial_threshold']

    out_fh = open(args.output, 'w', newline='') if args.output else sys.stdout
    writer = csv.DictWriter(out_fh, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    # ---- predict ---------------------------------------------------------
    tokenizer = CodonTokenizer()
    bact_thresh = args.bacterial_threshold
    filter_out = args.filter_output

    def _write_rows(rows):
        for row in rows:
            if filter_out:
                if row['above_host_threshold'] != 'yes':
                    continue
                if row['above_bacterial_threshold'] == 'yes':
                    continue
            writer.writerow(row)

    if args.batch_size <= 1:
        for i, rec in enumerate(records):
            probs = predict_sequence(
                model, tokenizer, rec['seq'], patch_nt_len, stride,
                temperature, device, args.max_patches, blocked_classes)
            rows = format_results(
                rec['id'], probs, hosts, threshold, bact_thresh, top_k)
            _write_rows(rows)

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
                    rec['id'], probs, hosts, threshold, bact_thresh, top_k)
                _write_rows(rows)

            done = min(start + args.batch_size, len(records))
            if done % 100 == 0 or done == len(records):
                logger.info(f"  processed {done}/{len(records)}")

    if args.output:
        out_fh.close()
        logger.info(f"Output written to {args.output}")

    logger.info("Done.")


if __name__ == '__main__':
    main()
