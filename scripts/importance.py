#!/usr/bin/env python3
"""Protein importance analysis via systematic scrambling.

For each genome, calls proteins with pyrodigal, then measures the effect of
scrambling each protein region on the host prediction scores.  Three scramble
types are applied to every protein:

  nucleotide  — shuffle individual nucleotides within the protein boundaries
  codon       — shuffle whole codons (triplets in the protein's reading frame)
  random      — shuffle nucleotides in a random region of the same length
                (length-matched control)

The script loads training data from --dataset_dir (same layout as train.py)
and subsamples N genomes per genus from the top-K most abundant genera.

Usage:
    python scripts/importance.py \
        --dataset_dir /path/to/dataset \
        --model_dir ./models/PT \
        --output importance_results \
        --n_genera 100 --n_per_genus 10

Expected dataset_dir contents:
    train.fna.gz           — training FASTA (gzipped)
    phages_hosts_tani.csv  — metadata with host_genus, in_testset columns
"""

import argparse
import csv
import gzip
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import pyrodigal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio import SeqIO

# ---------------------------------------------------------------------------
# Imports from the package — adjust if your layout differs
# ---------------------------------------------------------------------------
try:
    from phagetransformer.model import CodonTokenizer
    from phagetransformer.predict import (
        load_model_and_calibration, predict_sequence,
    )
    from phagetransformer.dataset import load_phage_host_merged
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model import CodonTokenizer
    from predict import load_model_and_calibration, predict_sequence
    from dataset import load_phage_host_merged

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling — shared colour palette and font sizes
# ---------------------------------------------------------------------------
from eval_utils import (
    COLORS, FONT_SIZES,
    setup_style, _save_figure, _suptitle, enable_presentation_mode,
)


# ---------------------------------------------------------------------------
# Dataset loading (matches evaluate_phages.py / train.py layout)
# ---------------------------------------------------------------------------

def read_fasta_ids(fasta_path: str) -> List[str]:
    """Read only the sequence IDs from a gzipped or plain FASTA."""
    ids = []
    opener = gzip.open if fasta_path.endswith('.gz') else open
    with opener(fasta_path, 'rt') as fh:
        for rec in SeqIO.parse(fh, 'fasta'):
            ids.append(rec.id)
    return ids


def load_dataset(dataset_dir: str) -> List[dict]:
    """Load training sequences with IDs and host genus labels.

    Uses ``load_phage_host_merged`` (same loader as train.py / evaluate) to
    read sequences and multi-hot labels, then reads the FASTA a second time
    to recover sequence IDs (which the shared loader discards).

    Returns list of {id, seq, genus} dicts.
    """
    train_seqs, train_labels, _, _, hosts_from_data = \
        load_phage_host_merged(dataset_dir)

    # Recover IDs from the FASTA header lines
    fasta_path = os.path.join(dataset_dir, 'train.fna.gz')
    if os.path.exists(fasta_path):
        ids = read_fasta_ids(fasta_path)
    else:
        ids = [f"seq_{i}" for i in range(len(train_seqs))]

    if len(ids) != len(train_seqs):
        logger.warning(f"ID count ({len(ids)}) != sequence count "
                       f"({len(train_seqs)}), using indices as fallback")
        ids = [f"seq_{i}" for i in range(len(train_seqs))]

    # Decode primary genus from multi-hot label matrix
    records = []
    for i, seq in enumerate(train_seqs):
        label_row = train_labels[i]
        primary_idx = int(np.argmax(label_row))
        genus = hosts_from_data[primary_idx] if label_row[primary_idx] > 0 else 'unknown'
        records.append({
            'id': ids[i],
            'seq': str(seq),          # explicit str — guards against Seq objects
            'genus': genus,
        })

    return records


def subsample(records: List[dict], n_genera: int, n_per_genus: int,
              rng: random.Random) -> List[dict]:
    """Pick top-K genera by count, then sample N genomes per genus."""
    genus_counts = Counter(r['genus'] for r in records)
    top_genera = set(g for g, _ in genus_counts.most_common(n_genera))

    genus_to_recs = defaultdict(list)
    for r in records:
        if r['genus'] in top_genera:
            genus_to_recs[r['genus']].append(r)

    selected = []
    for genus, recs in genus_to_recs.items():
        if len(recs) <= n_per_genus:
            selected.extend(recs)
        else:
            selected.extend(rng.sample(recs, n_per_genus))

    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Gene calling
# ---------------------------------------------------------------------------

def call_proteins(seq: str) -> List[dict]:
    """Run pyrodigal meta-mode on a nucleotide sequence.

    Returns list of dicts with keys: begin, end, strand, length_nt, category.
    Coordinates are 1-based inclusive, matching pyrodigal convention.
    """
    finder = pyrodigal.GeneFinder(meta=True)
    genes = finder.find_genes(seq.encode())
    proteins = []
    for g in genes:
        proteins.append({
            'begin': g.begin,       # 1-based
            'end': g.end,           # 1-based inclusive
            'strand': g.strand,     # +1 or -1
            'length_nt': g.end - g.begin + 1,
            'category': 'unknown function',
            'product': '',
        })
    return proteins


def load_pharokka_annotations(tsv_path: str) -> Dict[str, List[dict]]:
    """Parse pharokka CDS TSV into per-contig protein lists.

    Returns {contig_id: [protein dicts]} with the same keys as
    call_proteins plus 'category' and 'product'.
    """
    import pandas as pd
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)

    # pharokka column names (may vary slightly by version)
    contig_col = None
    for c in ['contig', 'Contig', 'contig_id']:
        if c in df.columns:
            contig_col = c
            break
    if contig_col is None:
        raise ValueError(f"Cannot find contig column in {tsv_path}. "
                         f"Columns: {list(df.columns)}")

    start_col = next(c for c in ['start', 'Start'] if c in df.columns)
    end_col = next(c for c in ['end', 'End', 'stop', 'Stop'] if c in df.columns)
    strand_col = next((c for c in ['strand', 'Strand', 'frame', 'Frame']
                       if c in df.columns), None)
    cat_col = next((c for c in ['category', 'Category', 'phrog_category',
                                'annot_category', 'function']
                    if c in df.columns), None)
    prod_col = next((c for c in ['product', 'Product', 'annot']
                     if c in df.columns), None)

    annotations = defaultdict(list)
    for _, row in df.iterrows():
        contig = row[contig_col]
        start = int(float(row[start_col]))
        end = int(float(row[end_col]))

        if strand_col and row.get(strand_col, '') != '':
            s = row[strand_col]
            strand = -1 if str(s).startswith('-') or str(s) == '-1' else 1
        else:
            strand = 1 if start < end else -1

        # Ensure start < end for coordinates
        begin = min(start, end)
        stop = max(start, end)

        category = row[cat_col].strip() if cat_col and pd.notna(row.get(cat_col)) else 'unknown function'
        product = row[prod_col].strip() if prod_col and pd.notna(row.get(prod_col)) else ''

        annotations[contig].append({
            'begin': begin,
            'end': stop,
            'strand': strand,
            'length_nt': stop - begin + 1,
            'category': category,
            'product': product,
        })

    logger.info(f"Loaded pharokka annotations for {len(annotations)} contigs, "
                f"{sum(len(v) for v in annotations.values())} CDS total")

    # Log category distribution
    all_cats = Counter(p['category']
                       for prots in annotations.values() for p in prots)
    for cat, n in all_cats.most_common():
        logger.info(f"    {cat}: {n}")

    return dict(annotations)


# ---------------------------------------------------------------------------
# Scrambling functions
# ---------------------------------------------------------------------------

def scramble_nucleotides(seq: str, start: int, end: int, rng: random.Random,
                         preserve_termini: bool = True) -> str:
    """Shuffle individual nucleotides in seq[start:end] (0-based half-open).

    If ``preserve_termini`` is True, the first three and last three
    nucleotides of the region are left untouched — this preserves the
    start and stop codons of a CDS whose coordinates include both
    terminators. This applies symmetrically for ``+`` and ``-`` strand
    genes: on the forward strand the start codon is at the 5' end of
    the slice (first 3 nt) and the stop codon is at the 3' end (last
    3 nt); on the reverse strand the stop codon sits at the 5' end of
    the slice and the start codon at the 3' end, but both are still
    the first/last three nucleotides of the slice and therefore both
    are preserved by the same rule.

    For regions shorter than 6 nt the full slice is returned unchanged
    (too short to meaningfully scramble between preserved termini).
    """
    if preserve_termini:
        if end - start < 6:
            return seq
        region = list(seq[start + 3:end - 3])
        rng.shuffle(region)
        return (seq[:start + 3] + ''.join(region)
                + seq[end - 3:end] + seq[end:])
    region = list(seq[start:end])
    rng.shuffle(region)
    return seq[:start] + ''.join(region) + seq[end:]


def scramble_codons(seq: str, start: int, end: int, strand: int,
                    rng: random.Random,
                    preserve_termini: bool = True) -> str:
    """Shuffle whole codons within the protein region.

    For +strand genes, codon boundaries are aligned from the left (start).
    For -strand genes, codon boundaries are aligned from the right (end).
    Partial terminal codons (0-2 nt) are left untouched.

    If ``preserve_termini`` is True, the first and last codon of the
    shuffled list are held in place (they are not included in the
    shuffle). For a CDS whose coordinates span both terminators this
    keeps the start codon and the stop codon at their original
    positions. The rule is frame-correct for both strands because
    the codon split is already aligned to whichever end carries the
    start codon.
    """
    region = seq[start:end]
    if strand == -1:
        # Align from the right: leftover nts go at the start
        offset = len(region) % 3
        prefix = region[:offset]
        codons = [region[i:i+3] for i in range(offset, len(region) - 2, 3)]
        if preserve_termini and len(codons) >= 4:
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
        if preserve_termini and len(codons) >= 4:
            first, last = codons[0], codons[-1]
            middle = codons[1:-1]
            rng.shuffle(middle)
            codons = [first] + middle + [last]
        else:
            rng.shuffle(codons)
        return seq[:start] + ''.join(codons) + tail + seq[end:]


def scramble_random_region(seq: str, length: int, rng: random.Random,
                           excluded_start: int = -1,
                           excluded_end: int = -1) -> str:
    """Shuffle nucleotides in a random region of ``length`` nt.

    Tries to avoid overlapping with [excluded_start, excluded_end) but
    falls back to any position if the genome is too short.

    Note: termini preservation is intentionally disabled for the
    random control — the random window has no semantic start/stop
    codons, so preserving its first/last 3 nt would introduce an
    asymmetry relative to the protein scrambles without any
    biological meaning.
    """
    seq_len = len(seq)
    if length > seq_len:
        length = seq_len

    # Try up to 50 times to find a non-overlapping position
    for _ in range(50):
        pos = rng.randint(0, seq_len - length)
        if excluded_start < 0 or pos + length <= excluded_start or pos >= excluded_end:
            return scramble_nucleotides(seq, pos, pos + length, rng,
                                        preserve_termini=False)

    # Fallback: just pick any position
    pos = rng.randint(0, seq_len - length)
    return scramble_nucleotides(seq, pos, pos + length, rng,
                                preserve_termini=False)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def get_scores(model, tokenizer, seq: str, patch_nt_len: int,
               stride: int, temperature: float,
               device: torch.device,
               max_patches: int = 512) -> Optional[np.ndarray]:
    """Thin wrapper around the package's predict_sequence.

    Returns shape (num_classes,) array of calibrated probabilities,
    or None if sequence is too short.
    """
    if not isinstance(seq, str):
        seq = str(seq)
    seq = seq[:len(seq) // 3 * 3]
    if len(seq) < 300:
        return None
    return predict_sequence(model, tokenizer, seq, patch_nt_len, stride,
                            temperature, device, max_patches)


# ---------------------------------------------------------------------------
# Post-processing: paired analysis
# ---------------------------------------------------------------------------

def build_paired_table(results: List[dict]) -> List[dict]:
    """Join each protein's scramble results with its matched random control.

    Returns one row per protein per replicate with fields:
        genome_id, protein_idx, protein_length_nt, genome_length,
        frac_of_genome, baseline_top, top_class,
        delta_nt, delta_cd, delta_rnd,
        paired_nt (= delta_nt - delta_rnd),
        paired_cd (= delta_cd - delta_rnd),
        norm_paired_nt (= paired_nt / protein_length_nt * 1000),
        norm_paired_cd (= paired_cd / protein_length_nt * 1000),
    """
    # Index: (genome_id, protein_idx, replicate) -> {scramble_type: row}
    grouped = defaultdict(dict)
    for r in results:
        key = (r['genome_id'], r['protein_idx'], r['replicate'])
        grouped[key][r['scramble_type']] = r

    paired = []
    for key, types in grouped.items():
        if not all(t in types for t in ('nucleotide', 'codon', 'random')):
            continue
        nt = types['nucleotide']
        cd = types['codon']
        rnd = types['random']

        p_len = nt['protein_length_nt']
        g_len = nt.get('genome_length', 0)
        if isinstance(g_len, str):
            g_len = int(float(g_len)) if g_len else 0
        frac = p_len / g_len if g_len > 0 else 0

        paired_nt = nt['delta_top'] - rnd['delta_top']
        paired_cd = cd['delta_top'] - rnd['delta_top']

        paired.append({
            'genome_id': nt['genome_id'],
            'protein_idx': nt['protein_idx'],
            'protein_start': nt['protein_start'],
            'protein_end': nt['protein_end'],
            'protein_length_nt': p_len,
            'genome_length': g_len,
            'frac_of_genome': frac,
            'category': nt.get('category', 'unknown function'),
            'product': nt.get('product', ''),
            'replicate': nt['replicate'],
            'baseline_top': nt['baseline_top'],
            'top_class': nt['top_class'],
            'delta_nt': nt['delta_top'],
            'delta_cd': cd['delta_top'],
            'delta_rnd': rnd['delta_top'],
            'paired_nt': paired_nt,
            'paired_cd': paired_cd,
            # Per-kb normalization for comparability across protein sizes
            'norm_paired_nt': paired_nt / p_len * 1000 if p_len > 0 else 0,
            'norm_paired_cd': paired_cd / p_len * 1000 if p_len > 0 else 0,
        })

    return paired


# ---------------------------------------------------------------------------
# Locality analysis
# ---------------------------------------------------------------------------

def _mean_nearest_neighbor_ranks(selected_ranks: np.ndarray,
                                  n_total: int) -> float:
    """Mean nearest-neighbor distance in gene-rank space (linear genome).

    ``selected_ranks`` are the positional indices (0-based) of the selected
    genes within the full sorted gene list.  Distance between two genes is
    the number of genes between them: |rank_i - rank_j| - 1.
    """
    n = len(selected_ranks)
    if n < 2:
        return 0.0
    sr = np.sort(selected_ranks)
    nn_dists = []
    for i in range(n):
        min_d = n_total
        for j in range(n):
            if i == j:
                continue
            d = abs(int(sr[i]) - int(sr[j])) - 1  # intervening genes
            if d < min_d:
                min_d = d
        nn_dists.append(min_d)
    return float(np.mean(nn_dists))


def locality_analysis(results: List[dict],
                      cumulative_results: List[dict],
                      target_delta: float = -0.3,
                      n_permutations: int = 1000,
                      seed: int = 42) -> List[dict]:
    """Test whether impactful proteins are spatially clustered.

    Uses cumulative scrambling results to find, for each genome, the minimum
    number of top proteins (min_k) whose simultaneous scrambling crosses
    ``target_delta``.  Then tests spatial clustering of both:
      - 'min_k': the smallest set that reaches the target
      - 'top5':  the full top-5 set

    Distance is measured as the number of intervening genes between selected
    proteins (gene-rank distance, linear genome), compared to a permutation
    null that randomly selects the same number of proteins from the genome.
    """
    rng = random.Random(seed)

    # Group individual protein results per genome (nt scramble, rep 0)
    genome_proteins = defaultdict(list)
    for r in results:
        if r['scramble_type'] == 'nucleotide' and r.get('replicate', 0) == 0:
            genome_proteins[r['genome_id']].append(r)

    # Build cumulative delta lookup: (genome_id, top_k) -> delta
    cum_lookup = defaultdict(dict)
    for cr in cumulative_results:
        cum_lookup[cr['genome_id']][cr['top_k']] = cr['delta_top']

    # Also add k=1 from individual results (worst single protein per genome)
    for genome_id, prots in genome_proteins.items():
        if prots:
            worst = min(p['delta_top'] for p in prots)
            cum_lookup[genome_id][1] = worst

    locality_results = []

    for genome_id, prots in genome_proteins.items():
        if len(prots) < 5:
            continue

        n_total = len(prots)

        # Sort all proteins by genomic position to get rank order
        pos_sorted = sorted(range(n_total),
                            key=lambda i: prots[i]['protein_start'])
        # Map: original index -> positional rank
        rank_of = {orig_idx: rank for rank, orig_idx in enumerate(pos_sorted)}

        # Sort by delta ascending (most negative first) for selection
        delta_order = sorted(range(n_total),
                             key=lambda i: prots[i]['delta_top'])

        g_len = prots[0].get('genome_length', 0)
        if isinstance(g_len, str):
            g_len = int(float(g_len)) if g_len else 0

        # Find min_k that crosses target_delta
        cum = cum_lookup.get(genome_id, {})
        min_k = None
        for k in [1, 2, 3, 4, 5]:
            if k in cum and cum[k] <= target_delta:
                min_k = k
                break

        # Build selection sets
        selections = {}

        if min_k is not None and min_k >= 2:
            sel_orig_idx = delta_order[:min_k]
            sel_ranks = np.array([rank_of[i] for i in sel_orig_idx])
            selections[f'min_k={min_k}'] = sel_ranks

        # Always test top-5
        top_k = min(5, n_total)
        if top_k >= 2:
            sel_orig_idx = delta_order[:top_k]
            sel_ranks = np.array([rank_of[i] for i in sel_orig_idx])
            selections['top5'] = sel_ranks

        # Fixed top-k values 2..5 — primary material for the figure panel
        for fixed_k in (2, 3, 4, 5):
            if n_total < fixed_k:
                continue
            sel_orig_idx = delta_order[:fixed_k]
            sel_ranks = np.array([rank_of[i] for i in sel_orig_idx])
            selections[f'top_k={fixed_k}'] = sel_ranks

        for method, sel_ranks in selections.items():
            k = len(sel_ranks)
            observed = _mean_nearest_neighbor_ranks(sel_ranks, n_total)

            # Count adjacent pairs (0 intervening genes)
            sr = np.sort(sel_ranks)
            n_adjacent = int(np.sum(np.diff(sr) == 1))
            # Expected adjacent pairs for k random from n (linear):
            # E = k*(k-1)/n
            expected_adjacent = k * (k - 1) / n_total if n_total > 0 else 0

            # Permutation null: random k proteins from all n
            null_dists = []
            null_adj = []
            for _ in range(n_permutations):
                perm_ranks = np.array(rng.sample(range(n_total), k))
                null_dists.append(
                    _mean_nearest_neighbor_ranks(perm_ranks, n_total))
                null_adj.append(int(np.sum(np.diff(np.sort(perm_ranks)) == 1)))

            null_arr = np.array(null_dists)
            null_mean = null_arr.mean()
            null_std = null_arr.std()
            z = ((observed - null_mean) / null_std
                 if null_std > 0 else 0.0)
            p_val = (null_arr <= observed).sum() / n_permutations

            # Adjacency p-value from permutation
            null_adj_arr = np.array(null_adj)
            adj_p_val = (null_adj_arr >= n_adjacent).sum() / n_permutations

            locality_results.append({
                'genome_id': genome_id,
                'method': method,
                'k': k,
                'min_k': min_k if min_k is not None else -1,
                'cumulative_delta': cum.get(k, float('nan')),
                'observed_mnn': observed,
                'null_mean': null_mean,
                'null_std': null_std,
                'z_score': z,
                'p_value': p_val,
                'n_adjacent': n_adjacent,
                'expected_adjacent': expected_adjacent,
                'adj_p_value': adj_p_val,
                'genome_length': g_len,
                'n_proteins_total': n_total,
            })

    return locality_results


# ---------------------------------------------------------------------------
# Figure: Combined 3x2 summary figure
# ---------------------------------------------------------------------------

def plot_combined(results: List[dict], cumulative_results: List[dict],
                  locality_results: List[dict], output_dir: str):
    """Single 8-panel figure summarising protein importance.

    Row 1 — distributions:
        A. Per-protein score drop (boxes for nt / codon / random)
        B. Spatial clustering of worst-k important proteins:
           observed MNN / null mean per genome at k in {2, 3, 4, 5}.
    Row 2 — worst-protein enrichment (count-based fold enrichment of
        the single most-affected protein per genome, paired nt/codon
        bars):
        C. By PHROG category.
        D. By PHROG product, coloured by modal category.
    Row 3 — per-genome importance-concentration ridges (|\u0394| share
        divided by count share, one ratio per genome, full distribution
        as paired nt/codon KDEs, log x-axis):
        E. By PHROG category.
        F. By PHROG product (top 10 by mean ratio).
    Row 4 — per-protein percentile-rank histograms (each protein ranked
        by signed Δ within its genome into 20 bins, pooled across
        genomes):
        G. By PHROG category.
        H. By PHROG product.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    nt_results = [r for r in results
                  if r['scramble_type'] == 'nucleotide'
                  and r.get('replicate', 0) == 0]
    cd_results = [r for r in results
                  if r['scramble_type'] == 'codon'
                  and r.get('replicate', 0) == 0]
    rnd_results = [r for r in results
                   if r['scramble_type'] == 'random'
                   and r.get('replicate', 0) == 0]

    if not nt_results:
        logger.warning("plot_combined: no nucleotide results — skipping")
        return

    has_cats = any(r.get('category', 'unknown function') != 'unknown function'
                   for r in nt_results)
    if not has_cats:
        logger.warning("plot_combined: no pharokka categories present — "
                       "skipping figure")
        return

    structural_cats = {'head and packaging', 'tail', 'connector'}

    def _shorten(c):
        s = c.replace('moron, auxiliary metabolic gene and host takeover',
                       'moron/AMG')
        return s[:23] + '..' if len(s) > 25 else s

    def _cat_color(c):
        if c.lower() in structural_cats:
            return COLORS['secondary']    # structural
        elif 'moron' in c.lower():
            return COLORS['tertiary']     # moron / AMG
        return COLORS['primary']          # other

    # ------------------------------------------------------------------
    # Shared computations
    # ------------------------------------------------------------------
    # Per-genome worst (most negative) delta for each scramble type
    def _worst_per_genome(rows):
        out = {}
        for r in rows:
            gid = r['genome_id']
            if gid not in out or r['delta_top'] < out[gid]:
                out[gid] = r['delta_top']
        return out

    nt_worst = _worst_per_genome(nt_results)
    cd_worst = _worst_per_genome(cd_results)
    rnd_worst = _worst_per_genome(rnd_results)

    # Per-genome worst-protein category (and product) for enrichment panels
    genome_prots_nt = defaultdict(list)
    for r in nt_results:
        genome_prots_nt[r['genome_id']].append(r)
    genome_prots_cd = defaultdict(list)
    for r in cd_results:
        genome_prots_cd[r['genome_id']].append(r)

    all_cats = [r.get('category', 'unknown function') or 'unknown function'
                for r in nt_results]
    all_prods = [r.get('product', '') or '' for r in nt_results]
    all_abs_nt = [abs(r['delta_top']) for r in nt_results]
    all_abs_cd = [abs(r['delta_top']) for r in cd_results]

    top1_cats_nt = []
    top1_cats_cd = []
    top1_prods_nt = []
    top1_prods_cd = []
    top1_absd_nt = []
    top1_absd_cd = []
    for gid, prots in genome_prots_nt.items():
        worst = min(prots, key=lambda x: x['delta_top'])
        top1_cats_nt.append(worst.get('category', 'unknown function')
                            or 'unknown function')
        top1_prods_nt.append(worst.get('product', '') or '')
        top1_absd_nt.append(abs(worst['delta_top']))
    for gid, prots in genome_prots_cd.items():
        worst = min(prots, key=lambda x: x['delta_top'])
        top1_cats_cd.append(worst.get('category', 'unknown function')
                            or 'unknown function')
        top1_prods_cd.append(worst.get('product', '') or '')
        top1_absd_cd.append(abs(worst['delta_top']))

    # Modal category per product (pharokka assigns category from product,
    # so this is essentially a 1-to-1 lookup with graceful tie-handling)
    prod_to_cat_counts = defaultdict(Counter)
    for r in nt_results:
        prod = r.get('product', '') or ''
        cat = r.get('category', 'unknown function') or 'unknown function'
        if prod:
            prod_to_cat_counts[prod][cat] += 1
    prod_to_cat = {p: c.most_common(1)[0][0]
                   for p, c in prod_to_cat_counts.items()}

    # ------------------------------------------------------------------
    # Enrichment helper (local re-implementation, kept compact)
    # ------------------------------------------------------------------
    def _enrichment(selected, background, min_count=0,
                    selected_weights=None, background_weights=None):
        """Fold enrichment of groups in ``selected`` vs ``background``.

        Default is count-based. If weights are provided for both sides
        (lists parallel to ``selected`` / ``background``), enrichment is
        computed on the summed weight per group instead — "share of
        total weight in selected" over "share of total weight in
        background". In this module's usage, weights are |\u0394| so the
        bars read as "share of total score-drop magnitude attributable
        to this group, relative to its background share."
        """
        sel_counts = Counter(selected)
        bg_counts = Counter(background)

        if selected_weights is not None and background_weights is not None:
            sel_w = defaultdict(float)
            bg_w = defaultdict(float)
            for s, w in zip(selected, selected_weights):
                sel_w[s] += float(w)
            for s, w in zip(background, background_weights):
                bg_w[s] += float(w)
            sel_total = sum(sel_w.values())
            bg_total = sum(bg_w.values())
            weighted = True
        else:
            sel_w = sel_counts
            bg_w = bg_counts
            sel_total = len(selected)
            bg_total = len(background)
            weighted = False

        out = []
        if bg_total == 0 or sel_total == 0:
            return out
        for k in sorted(bg_counts.keys()):
            if bg_counts[k] < min_count:
                continue
            bg_frac = bg_w[k] / bg_total if bg_total > 0 else 0.0
            sel_frac = sel_w.get(k, 0.0) / sel_total if sel_total > 0 else 0.0
            fold = sel_frac / bg_frac if bg_frac > 0 else 0.0
            out.append({'key': k,
                        'n_sel': sel_counts.get(k, 0),
                        'n_bg': bg_counts[k],
                        'fold': fold,
                        'weighted': weighted})
        out.sort(key=lambda x: x['fold'], reverse=True)
        return out

    def _plot_paired_bars(ax, enrich_nt, enrich_cd, title, color_fn,
                          is_product=False, max_labels=0):
        """Paired horizontal bar plot: two bars per group — nt and codon.

        Bars share a single y-axis of category/product names (right-aligned
        for readability). The nt scramble bar is drawn at full colour
        opacity; the codon bar uses a lighter hue (same base colour,
        lower alpha) to distinguish the two scramble types without
        doubling the palette. Categories are ordered by the mean fold
        enrichment across the two scramble types, most-enriched on top.
        """
        # Build a dict lookup keyed by category/product label.
        nt_by_key = {e['key']: e for e in enrich_nt}
        cd_by_key = {e['key']: e for e in enrich_cd}
        keys_all = set(nt_by_key) | set(cd_by_key)

        def _mean_fold(k):
            vals = [nt_by_key.get(k, {}).get('fold', 0.0),
                    cd_by_key.get(k, {}).get('fold', 0.0)]
            return sum(vals) / max(1, sum(1 for v in vals if v > 0))

        ordered_keys = sorted(keys_all, key=_mean_fold, reverse=True)
        if max_labels > 0:
            ordered_keys = ordered_keys[:max_labels]

        weighted = bool(
            (enrich_nt and enrich_nt[0].get('weighted')) or
            (enrich_cd and enrich_cd[0].get('weighted'))
        )

        # Reverse so the highest-enriched sits at the top of the axis.
        ordered_keys = list(reversed(ordered_keys))

        y = np.arange(len(ordered_keys))
        bar_h = 0.38
        folds_nt = [nt_by_key.get(k, {}).get('fold', 0.0)
                    for k in ordered_keys]
        folds_cd = [cd_by_key.get(k, {}).get('fold', 0.0)
                    for k in ordered_keys]
        ns_nt = [nt_by_key.get(k, {}).get('n_sel', 0) for k in ordered_keys]
        ns_cd = [cd_by_key.get(k, {}).get('n_sel', 0) for k in ordered_keys]
        bgs = [max(nt_by_key.get(k, {}).get('n_bg', 0),
                   cd_by_key.get(k, {}).get('n_bg', 0))
               for k in ordered_keys]
        colors = [color_fn(k) for k in ordered_keys]

        # nt = full opacity, codon = lighter (lower alpha) of the same hue.
        ax.barh(y + bar_h / 2, folds_nt, height=bar_h,
                color=colors, alpha=0.85, edgecolor='white')
        ax.barh(y - bar_h / 2, folds_cd, height=bar_h,
                color=colors, alpha=0.4, edgecolor='white')

        # Annotate bar-end counts as n_selected/n_background so the
        # reader can see both the worst-pick count and the total
        # occurrences driving the fold enrichment.
        xmax = max(folds_nt + folds_cd + [1.0])
        offset = xmax * 0.01
        for i, (fn, fc, nn, nc, bg) in enumerate(zip(
                folds_nt, folds_cd, ns_nt, ns_cd, bgs)):
            lbl_n = f'{nn}/{bg}'
            lbl_c = f'{nc}/{bg}'
            ax.text(fn + offset, i + bar_h / 2, lbl_n,
                    va='center', ha='left', fontsize=ANNOT_FS)
            ax.text(fc + offset, i - bar_h / 2, lbl_c,
                    va='center', ha='left', fontsize=ANNOT_FS,
                    color=COLORS['dark_overlay'])

        ax.set_yticks(y)
        ax.set_yticklabels([_shorten(k) for k in ordered_keys],
                           fontsize=TICK_FS)
        ax.tick_params(axis='x', labelsize=TICK_FS)
        ax.axvline(1.0, color=COLORS['text_light'], linewidth=1, linestyle='--')
        if weighted:
            ax.set_xlabel('Magnitude-weighted fold enrichment '
                          '(\u03a3|\u0394|)', fontsize=TICK_FS)
        else:
            ax.set_xlabel('Fold enrichment', fontsize=TICK_FS)
        ax.set_title(title, fontweight='bold', loc='left',
                     fontsize=TITLE_FS)
        # Extend x-lim for the bar-end count labels.
        ax.set_xlim(0, xmax * 1.12)

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    # Apply the shared matplotlib style before building the figure.
    setup_style()

    # Font-size aliases sourced from eval_utils.FONT_SIZES so this figure
    # stays visually consistent with the rest of the PhageTransformer
    # figure family.
    TICK_FS = FONT_SIZES['label']    # tick labels + axis labels
    TITLE_FS = FONT_SIZES['title']   # panel titles
    ANNOT_FS = FONT_SIZES['overlay'] # small in-panel annotations
    LEGEND_FS = FONT_SIZES['legend'] # figure-level legend

    # Semantic colour aliases. All bar, box and accent colours are drawn
    # from eval_utils.COLORS so the figure matches the shared palette.
    COL_NT     = COLORS['primary']       # nucleotide-scramble bars / box
    COL_CD     = COLORS['quaternary']    # codon-scramble box
    COL_RND    = COLORS['text_light']    # random-region control
    COL_STRUCT = COLORS['secondary']     # structural categories
    COL_MORON  = COLORS['tertiary']      # moron / AMG
    COL_OTHER  = COLORS['primary']       # everything else
    COL_GREY   = COLORS['dark_overlay']  # annotation text + neutral legend

    fig = plt.figure(figsize=(14 * 0.7, 19 * 0.7))
    # Four equal-height rows: A/B distributions, C/D worst-protein
    # enrichment, E/F per-genome concentration ridges, G/H per-protein
    # z-score ridges.
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.35, wspace=0.45,
                           left=0.08, right=0.97, top=0.97, bottom=0.08)
    rng_plot = random.Random(0)

    # ── A. Per-protein score drop distribution (boxplot, 3 groups) ──────
    # Every (genome, protein) contributes one point per scramble type.
    ax_a = fig.add_subplot(gs[0, 0])
    groups = [
        ('Nucleotide', [r['delta_top'] for r in nt_results],  COL_NT),
        ('Codon',      [r['delta_top'] for r in cd_results],  COL_CD),
        ('Random',     [r['delta_top'] for r in rnd_results], COL_RND),
    ]
    data = [g[1] for g in groups]
    positions = [1, 2, 3]
    bp = ax_a.boxplot(data, positions=positions, widths=0.5,
                      patch_artist=True, showfliers=False, zorder=2,
                      medianprops=dict(color='black', linewidth=1.5))
    for patch, (_, _, col) in zip(bp['boxes'], groups):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for i, (name, vals, col) in enumerate(groups):
        x = [positions[i] + rng_plot.uniform(-0.15, 0.15) for _ in vals]
        ax_a.scatter(x, vals, s=4, alpha=0.25, color=col, zorder=1,
                     edgecolors='none')
    ax_a.set_xticks(positions)
    ax_a.set_xticklabels([g[0] for g in groups], fontsize=TICK_FS)
    ax_a.tick_params(axis='y', labelsize=TICK_FS)
    ax_a.axhline(0, color=COLORS['text_light'], linewidth=0.8, linestyle='--', zorder=0)
    ax_a.set_ylabel('Per-protein score drop (\u0394)', fontsize=TICK_FS)
    ax_a.set_title('A. Per-protein score drop',
                   fontweight='bold', loc='left', fontsize=TITLE_FS)
    # Clip extreme outliers for readability and leave headroom at top
    # for the med/mean annotation labels.
    all_vals = np.concatenate([np.asarray(v) for v in data if len(v) > 0])
    if len(all_vals) > 0:
        ylo = float(np.percentile(all_vals, 0.5))
        span = abs(ylo) + 0.02
        margin_lo = span * 0.1
        # Headroom sized so the two-line annotation fits comfortably.
        headroom = span * 0.22
        ax_a.set_ylim(ylo - margin_lo, headroom)

    # Annotate median + mean above each box (after ylim is set so text
    # sits just inside the top of the panel).
    ytop = ax_a.get_ylim()[1]
    for i, (name, vals, col) in enumerate(groups):
        if not vals:
            continue
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        ax_a.text(positions[i], ytop,
                  f'med={med:.3f}\nmean={mean:.3f}',
                  ha='center', va='top', fontsize=ANNOT_FS, color=col,
                  fontweight='bold')

    # ── B. Locality: observed / expected MNN ratio of top-k important ──
    #      proteins at k in {2, 3, 4, 5}.
    #
    # For each genome, observed_mnn is compared against that same
    # genome's permutation null (mean across n_permutations random
    # draws of k proteins from the genome's n_total proteins). The
    # ratio observed_mnn / null_mean is < 1 when the top-k proteins
    # sit closer together than random — clustered — and > 1 when
    # they sit further apart. A horizontal line at 1 marks the null
    # expectation. Log y-axis makes below/above 1 symmetric.
    #
    # The per-genome p-value in the TSV is the left-tailed fraction of
    # permutations at least as extreme as observed; we summarise per
    # k-group as the fraction of genomes with p < 0.05.
    ax_b = fig.add_subplot(gs[0, 1])
    ks = [2, 3, 4, 5]
    P_CLUSTERED = 0.05

    ratios_by_k = {k: [] for k in ks}
    n_clustered_by_k = {k: 0 for k in ks}
    n_total_by_k = {k: 0 for k in ks}
    for r in locality_results or []:
        method = r.get('method', '')
        if not method.startswith('top_k='):
            continue
        try:
            k = int(method.split('=')[1])
        except (ValueError, IndexError):
            continue
        if k not in ratios_by_k:
            continue
        null_mean = r.get('null_mean', 0.0)
        if null_mean <= 0:
            # Degenerate null (vanishingly rare); skip to avoid div-by-zero.
            continue
        ratio = r['observed_mnn'] / null_mean
        ratios_by_k[k].append(ratio)
        n_total_by_k[k] += 1
        if r.get('p_value', 1.0) < P_CLUSTERED:
            n_clustered_by_k[k] += 1

    positions_b = list(range(1, len(ks) + 1))
    data_b = [ratios_by_k[k] for k in ks]

    if any(len(d) > 0 for d in data_b):
        bp = ax_b.boxplot(data_b, positions=positions_b, widths=0.55,
                          patch_artist=True, showfliers=False, zorder=2,
                          medianprops=dict(color='black', linewidth=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor(COL_NT)
            patch.set_alpha(0.75)
        for i, k in enumerate(ks):
            y = ratios_by_k[k]
            x = [positions_b[i] + rng_plot.uniform(-0.15, 0.15) for _ in y]
            ax_b.scatter(x, y, s=4, alpha=0.2, color=COL_NT, zorder=1,
                         edgecolors='none')

    ax_b.axhline(1.0, color=COLORS['text_light'], linewidth=0.8, linestyle='--', zorder=0)
    ax_b.set_yscale('log')
    ax_b.set_xticks(positions_b)
    ax_b.set_xticklabels([f'k={k}' for k in ks], fontsize=TICK_FS)
    ax_b.tick_params(axis='y', labelsize=TICK_FS)
    ax_b.set_xlabel('Number of worst proteins', fontsize=TICK_FS)
    ax_b.set_ylabel('Observed MNN / null mean (per genome)',
                    fontsize=TICK_FS)
    ax_b.set_title('B. Worst-k protein spatial clustering',
                   fontweight='bold', loc='left', fontsize=TITLE_FS)

    # Give the log axis extra headroom on top for the 3-line annotation.
    if any(len(d) > 0 for d in data_b):
        ylo_b, yhi_b = ax_b.get_ylim()
        ax_b.set_ylim(ylo_b, yhi_b * 1.8)

    # Annotate median ratio and clustered fraction above each box.
    ytop_b = ax_b.get_ylim()[1]
    for i, k in enumerate(ks):
        vals = ratios_by_k[k]
        if not vals:
            continue
        med = float(np.median(vals))
        n_clust = n_clustered_by_k[k]
        n_tot = n_total_by_k[k]
        pct = 100.0 * n_clust / n_tot if n_tot > 0 else 0.0
        ax_b.text(positions_b[i], ytop_b,
                  f'med={med:.2f}\n{n_clust}/{n_tot} ({pct:.0f}%)\np<0.05',
                  ha='center', va='top', fontsize=ANNOT_FS,
                  color=COL_GREY, fontweight='bold')

    # ── C. Merged category enrichment (nt + codon as paired bars) ───────
    # Count-based fold enrichment: category k's share of the worst-per-
    # genome picks divided by its share of the background population.
    enrich_cat_nt = _enrichment(top1_cats_nt, all_cats, min_count=0)
    enrich_cat_cd = _enrichment(top1_cats_cd, all_cats, min_count=0)

    ax_c = fig.add_subplot(gs[1, 0])
    _plot_paired_bars(ax_c, enrich_cat_nt, enrich_cat_cd,
                      f'C. Worst protein / genome — category '
                      f'(n={len(top1_cats_nt)} genomes)',
                      _cat_color, is_product=False)

    # ── D. Merged product enrichment (nt + codon as paired bars) ────────
    # Count-based, same metric as panel C but resolved to PHROG product
    # annotation rather than PHROG top-level category. Larger min_count
    # filter so rare names don't flood the top 10.
    prod_min_count = 50
    enrich_prod_nt = _enrichment(top1_prods_nt, all_prods,
                                 min_count=prod_min_count)
    enrich_prod_cd = _enrichment(top1_prods_cd, all_prods,
                                 min_count=prod_min_count)

    # Exclude empty product string and 'hypothetical protein' for readability
    def _filter_prods(enrich):
        return [e for e in enrich
                if e['key']
                and e['key'].lower() not in ('hypothetical protein',
                                             'unknown function', '')]
    enrich_prod_nt = _filter_prods(enrich_prod_nt)
    enrich_prod_cd = _filter_prods(enrich_prod_cd)

    def _prod_color(p):
        cat = prod_to_cat.get(p, 'unknown function')
        return _cat_color(cat)

    ax_d = fig.add_subplot(gs[1, 1])
    _plot_paired_bars(ax_d, enrich_prod_nt, enrich_prod_cd,
                      f'D. Worst product / genome '
                      f'(n={len(top1_prods_nt)} genomes)',
                      _prod_color, is_product=True, max_labels=10)

    # ── E / F. Per-genome importance-concentration ratio ─────────────────
    #
    # For every (genome, scramble_type) pair, group proteins by their
    # PHROG category / product and compute:
    #
    #     ratio = (Σ|Δ| in group k in genome g) / (Σ|Δ| in genome g)
    #             ÷ (n_k_in_g / n_in_g)
    #
    # A ratio > 1 means group k concentrates more importance mass than
    # its headcount in that genome would predict; < 1 means the opposite.
    # Unlike the worst-protein panels C/D, every protein contributes via
    # its group — this is a distributional statement, not a single-winner
    # statement. We aggregate by building one distribution per (group,
    # scramble_type) across the 999 genomes.
    #
    # Note: this uses count share as the null, not length share. That
    # means long proteins have more opportunity to accumulate |Δ| and
    # categories dominated by long proteins will tend to look enriched.
    # The alternative (length share) over-corrects for motif-driven
    # signals where a short region inside a long protein carries the
    # importance, so count share is the more biologically conservative
    # choice.
    def _per_genome_ratios(genome_prots, keyfn):
        """Return {group_key: [|Δ|-share-over-count-share ratios]}."""
        out = defaultdict(list)
        for gid, prots in genome_prots.items():
            total_abs = sum(abs(r['delta_top']) for r in prots)
            total_n   = len(prots)
            if total_abs <= 0 or total_n == 0:
                continue
            # Aggregate within this genome
            grp_abs = defaultdict(float)
            grp_n   = defaultdict(int)
            for r in prots:
                k = keyfn(r)
                if not k:
                    continue
                grp_abs[k] += abs(r['delta_top'])
                grp_n[k]   += 1
            for k, na in grp_n.items():
                if na <= 0:
                    continue
                share_abs = grp_abs[k] / total_abs
                share_n   = na / total_n
                out[k].append(share_abs / share_n)
        return out

    cat_ratios_nt = _per_genome_ratios(
        genome_prots_nt,
        lambda r: (r.get('category', 'unknown function')
                   or 'unknown function'))
    cat_ratios_cd = _per_genome_ratios(
        genome_prots_cd,
        lambda r: (r.get('category', 'unknown function')
                   or 'unknown function'))

    prod_ratios_nt = _per_genome_ratios(
        genome_prots_nt,
        lambda r: (r.get('product', '') or ''))
    prod_ratios_cd = _per_genome_ratios(
        genome_prots_cd,
        lambda r: (r.get('product', '') or ''))

    # ── G / H. Per-protein percentile-rank histograms by group ──────────
    #
    # For every genome g, rank its proteins by Δ (signed). The rank is
    # expressed as a percentile: 0 = most-negative Δ (most-important),
    # 100 = most-positive. Collect ranks across genomes, bucket per
    # group into 20 bins (5-percentile-wide), and plot a histogram.
    #
    # Under the null (group k's proteins are indistinguishable from
    # the rest), the histogram is flat at 1/20 = 5% per bin. Groups
    # that carry consistent importance pile up at low percentiles;
    # mixed-signal groups show mass at both ends (bimodality);
    # uninvolved groups stay flat.
    def _per_protein_percentile_ranks(genome_prots, keyfn, min_prots=10):
        """Return {group_key: [percentile ranks across genomes]}.

        Proteins are ranked within their genome by raw signed Δ. Rank is
        0-100: 0 = most-negative Δ (most-important), 100 = most-positive.
        Ties broken by average rank.
        """
        out = defaultdict(list)
        for gid, prots in genome_prots.items():
            if len(prots) < min_prots:
                continue
            deltas = np.array([r['delta_top'] for r in prots], dtype=float)
            # Average-rank: tied Δ values get the mean of the ranks they
            # would have received. scipy.stats.rankdata does this.
            from scipy.stats import rankdata
            ranks = rankdata(deltas, method='average')  # 1..n, ascending
            n = len(deltas)
            if n < 2:
                continue
            percentiles = (ranks - 1) / (n - 1) * 100.0
            for r, p in zip(prots, percentiles):
                k = keyfn(r)
                if not k:
                    continue
                out[k].append(float(p))
        return out

    cat_pct_nt = _per_protein_percentile_ranks(
        genome_prots_nt,
        lambda r: (r.get('category', 'unknown function')
                   or 'unknown function'))
    cat_pct_cd = _per_protein_percentile_ranks(
        genome_prots_cd,
        lambda r: (r.get('category', 'unknown function')
                   or 'unknown function'))

    prod_pct_nt = _per_protein_percentile_ranks(
        genome_prots_nt,
        lambda r: (r.get('product', '') or ''))
    prod_pct_cd = _per_protein_percentile_ranks(
        genome_prots_cd,
        lambda r: (r.get('product', '') or ''))

    def _plot_ridges(ax, ratios_nt, ratios_cd, title, color_fn,
                     min_genomes=50, max_rows=0, exclude_keys=(),
                     log_x=True, null_x=1.0, xlabel=None,
                     ordered_keys=None):
        """Paired ridge plot of per-group distributions.

        For each group key surviving filters, draw two horizontal KDEs
        stacked in one row: nt on top (full alpha), codon on bottom
        (half alpha). KDEs are height-normalised to unit peak so rare
        groups don't vanish beside common ones.

        log_x=True evaluates the KDE in log10 space (for multiplicative
        quantities like ratios); log_x=False uses linear space (for
        additive quantities like z-scores). null_x sets the reference
        line position.
        """
        from scipy.stats import gaussian_kde

        if ordered_keys is not None:
            # Caller specified the row order (used so panels G/H can
            # render the same rows in the same order as E/F). Still
            # filter to rows with enough data in *this* panel's dicts,
            # so a group absent here silently drops out rather than
            # producing a blank ridge.
            ordered = [k for k in ordered_keys
                       if len(ratios_nt.get(k, [])) >= min_genomes
                       and len(ratios_cd.get(k, [])) >= min_genomes]
        else:
            # Filter: need enough contributing entries in both nt and
            # codon distributions, and key must not be in exclude list.
            ex = {e.lower() for e in exclude_keys}
            common_keys = [k for k in set(ratios_nt) | set(ratios_cd)
                           if k
                           and k.lower() not in ex
                           and len(ratios_nt.get(k, [])) >= min_genomes
                           and len(ratios_cd.get(k, [])) >= min_genomes]

            # Order by the mean of nt and codon medians (higher on top).
            def _score(k):
                m_nt = np.median(ratios_nt.get(k, [null_x]))
                m_cd = np.median(ratios_cd.get(k, [null_x]))
                return 0.5 * (m_nt + m_cd)
            ordered = sorted(common_keys, key=_score, reverse=True)
            if max_rows > 0:
                ordered = ordered[:max_rows]

            # Reverse so highest-scoring row sits at the top of the panel.
            ordered = list(reversed(ordered))

        # Transform values for KDE. Log-space for ratios (multiplicative);
        # linear for z-scores (additive).
        def _vals_for_kde(vals):
            if log_x:
                return np.log10([v for v in vals if v > 0])
            return np.asarray([v for v in vals if np.isfinite(v)])

        # Build shared grid across all rows for comparability.
        all_vals = []
        for k in ordered:
            all_vals.extend(_vals_for_kde(ratios_nt.get(k, [])))
            all_vals.extend(_vals_for_kde(ratios_cd.get(k, [])))
        if len(all_vals) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            return ordered
        xlo_g, xhi_g = np.percentile(all_vals, [1, 99])
        pad = (xhi_g - xlo_g) * 0.1
        xlo_g, xhi_g = xlo_g - pad, xhi_g + pad
        grid_t = np.linspace(xlo_g, xhi_g, 256)
        # xgrid is always in *data* coords (so we can plot directly);
        # grid_t is the KDE evaluation space (log or linear).
        xgrid = 10 ** grid_t if log_x else grid_t

        row_height = 0.9
        half = row_height / 2.0

        for row, k in enumerate(ordered):
            color = color_fn(k)
            for side, rmap, alpha in (('nt', ratios_nt, 0.85),
                                      ('cd', ratios_cd, 0.4)):
                vals_t = _vals_for_kde(rmap.get(k, []))
                if len(vals_t) < min_genomes:
                    continue
                kde = gaussian_kde(vals_t, bw_method='scott')
                dens = kde(grid_t)
                if dens.max() > 0:
                    dens = dens / dens.max() * half * 0.92
                y_centre = row
                if side == 'nt':
                    y0 = y_centre
                    y1 = y_centre + dens
                else:
                    y0 = y_centre
                    y1 = y_centre - dens
                ax.fill_between(xgrid, y0, y1, color=color, alpha=alpha,
                                linewidth=0)
                # Median tick inside each half-ridge
                m_data = float(np.median([v for v in rmap.get(k, [])
                                          if np.isfinite(v)
                                          and (v > 0 or not log_x)]))
                m_t = np.log10(m_data) if log_x else m_data
                if grid_t[0] <= m_t <= grid_t[-1]:
                    m_dens = float(kde(m_t)[0])
                    if m_dens > 0 and dens.max() > 0:
                        tick = m_dens / kde(grid_t).max() * half * 0.92
                        if side == 'nt':
                            ax.plot([m_data, m_data],
                                    [y_centre, y_centre + tick],
                                    color=COLORS['text'], linewidth=1.0,
                                    alpha=0.9)
                        else:
                            ax.plot([m_data, m_data],
                                    [y_centre, y_centre - tick],
                                    color=COLORS['text'], linewidth=1.0,
                                    alpha=0.9)

        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels([_shorten(k) for k in ordered],
                           fontsize=TICK_FS)
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for row, k in enumerate(ordered):
            n = len(ratios_nt.get(k, []))
            ax.text(0.985, row, f'n={n}',
                    transform=trans,
                    ha='right', va='center', fontsize=ANNOT_FS,
                    color=COL_GREY)

        if log_x:
            ax.set_xscale('log')
            ax.set_xlim(10 ** xlo_g, 10 ** xhi_g)
        else:
            ax.set_xlim(xlo_g, xhi_g)
        ax.axvline(null_x, color=COLORS['text_light'], linewidth=1,
                   linestyle='--')
        default_label = ('Per-genome importance concentration '
                         '(|\u0394| share / count share)'
                         if log_x else 'z-score')
        ax.set_xlabel(xlabel or default_label, fontsize=TICK_FS)
        ax.tick_params(axis='x', labelsize=TICK_FS)
        ax.set_ylim(-0.7, len(ordered) - 0.3)
        ax.set_title(title, fontweight='bold', loc='left',
                     fontsize=TITLE_FS)
        return ordered

    def _plot_histogram_ridges(ax, dists_nt, dists_cd, title, color_fn,
                               ordered_keys, n_bins=20,
                               xlabel='Percentile rank of \u0394 within '
                                      'genome (0 = most-negative)',
                               null_density=None):
        """Paired per-row histograms of 0-100 percentile distributions.

        Each row shows two histograms stacked around a centre line: nt
        bars rise above the row centre, codon bars hang below. All rows
        share the same fixed bin edges (0..100), so visual comparisons
        between rows and between panels are honest.
        """
        ordered = [k for k in ordered_keys
                   if k in dists_nt or k in dists_cd]

        bin_edges = np.linspace(0.0, 100.0, n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_lefts = bin_edges[:-1]

        if null_density is None:
            null_density = 1.0 / n_bins

        # Find the max density across all rows + sides to scale the
        # half-row height consistently. This way a row with a huge peak
        # doesn't dwarf the others — every row's tallest bar reaches the
        # same fraction of the row slot.
        peak = 0.0
        densities = {}
        for k in ordered:
            for side, dmap in (('nt', dists_nt), ('cd', dists_cd)):
                vals = np.asarray(dmap.get(k, []), dtype=float)
                if vals.size == 0:
                    densities[(k, side)] = np.zeros(n_bins)
                    continue
                counts, _ = np.histogram(vals, bins=bin_edges)
                d = counts / counts.sum() if counts.sum() > 0 \
                    else np.zeros_like(counts, dtype=float)
                densities[(k, side)] = d
                peak = max(peak, float(d.max()))
        if peak <= 0:
            peak = null_density * 2

        row_height = 0.9
        half = row_height / 2.0
        scale = half * 0.92 / peak

        for row, k in enumerate(ordered):
            color = color_fn(k)
            for side, alpha, sign in (('nt', 0.85, +1),
                                      ('cd', 0.4, -1)):
                d = densities.get((k, side))
                if d is None or d.sum() == 0:
                    continue
                heights = d * scale * sign
                ax.bar(bin_lefts, heights, width=bin_width,
                       bottom=row, align='edge',
                       color=color, alpha=alpha,
                       edgecolor='white', linewidth=0.3)

        # Null reference line: flat density, in the same scaled units.
        null_h = null_density * scale
        for row in range(len(ordered)):
            ax.plot([0, 100], [row + null_h, row + null_h],
                    color=COLORS['text_light'], linewidth=0.8,
                    linestyle='--', alpha=0.7)
            ax.plot([0, 100], [row - null_h, row - null_h],
                    color=COLORS['text_light'], linewidth=0.8,
                    linestyle='--', alpha=0.7)

        # Row labels and per-row n-annotations
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels([_shorten(k) for k in ordered],
                           fontsize=TICK_FS)
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for row, k in enumerate(ordered):
            n = len(dists_nt.get(k, []))
            ax.text(0.985, row, f'n={n}',
                    transform=trans,
                    ha='right', va='center', fontsize=ANNOT_FS,
                    color=COL_GREY)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.7, len(ordered) - 0.3)
        ax.set_xlabel(xlabel, fontsize=TICK_FS)
        ax.tick_params(axis='x', labelsize=TICK_FS)
        ax.set_title(title, fontweight='bold', loc='left',
                     fontsize=TITLE_FS)

    ax_e = fig.add_subplot(gs[2, 0])
    cat_order = _plot_ridges(
        ax_e, cat_ratios_nt, cat_ratios_cd,
        'E. Per-genome concentration — category',
        _cat_color, min_genomes=50)

    ax_f = fig.add_subplot(gs[2, 1])
    prod_order = _plot_ridges(
        ax_f, prod_ratios_nt, prod_ratios_cd,
        'F. Per-genome concentration — product',
        _prod_color, min_genomes=50, max_rows=10,
        exclude_keys=('hypothetical protein', 'unknown function', ''))

    # Panels G/H render their own top-scoring rows by combined first-
    # and last-bin mass. Candidate pool is every group that passes the
    # same filters E/F would use (min sample size, exclude list).
    def _tail_score(k, dists):
        vals = np.asarray(dists.get(k, []), dtype=float)
        if vals.size == 0:
            return 0.0
        # Fraction of proteins in the first 5% or last 5% of within-
        # genome Δ rank. Equivalent to the first-bin + last-bin mass
        # of the 20-bin histogram.
        tail_mass = np.mean((vals < 5.0) | (vals >= 95.0))
        return float(tail_mass)

    def _rank_by_tails(dists_nt, dists_cd, min_n=50, max_rows=0,
                       exclude_keys=()):
        ex = {e.lower() for e in exclude_keys}
        keys = [k for k in set(dists_nt) | set(dists_cd)
                if k and k.lower() not in ex
                and len(dists_nt.get(k, [])) >= min_n
                and len(dists_cd.get(k, [])) >= min_n]
        scored = [(0.5 * (_tail_score(k, dists_nt) +
                          _tail_score(k, dists_cd)), k)
                  for k in keys]
        scored.sort(key=lambda x: x[0], reverse=True)
        keys = [k for _, k in scored]
        if max_rows > 0:
            keys = keys[:max_rows]
        # Reverse so highest-scoring row sits at the top of the panel.
        return list(reversed(keys))

    # For categories there are only ~10 anyway, so just reorder the set
    # that panel E displays. For products the candidate pool is much
    # larger than F's top-10, so we score the whole pool.
    cat_order_gh = _rank_by_tails(
        cat_pct_nt, cat_pct_cd, min_n=50)
    prod_order_gh = _rank_by_tails(
        prod_pct_nt, prod_pct_cd, min_n=50, max_rows=10,
        exclude_keys=('hypothetical protein', 'unknown function', ''))

    # Log candidate-pool sizes so it's obvious when G/H overlap with
    # E/F by coincidence rather than by construction.
    cat_pool_n = len({k for k in set(cat_pct_nt) | set(cat_pct_cd)
                      if k
                      and len(cat_pct_nt.get(k, [])) >= 50
                      and len(cat_pct_cd.get(k, [])) >= 50})
    prod_pool_n = len({
        k for k in set(prod_pct_nt) | set(prod_pct_cd)
        if k
        and k.lower() not in {'hypothetical protein', 'unknown function', ''}
        and len(prod_pct_nt.get(k, [])) >= 50
        and len(prod_pct_cd.get(k, [])) >= 50
    })
    logger.info(f"  Panel G candidate pool: {cat_pool_n} categories "
                f"(min 50 proteins each in nt and codon)")
    logger.info(f"  Panel H candidate pool: {prod_pool_n} products "
                f"(min 50 proteins each in nt and codon, excluding "
                f"hypothetical/unknown); ranked by first+last-bin mass, "
                f"top 10 shown")

    # Diagnostic: what products would dominate the last-bin (rank
    # 95-100) side specifically, at a more permissive min_n=20? If the
    # main selection above misses interesting last-bin-heavy products
    # because they're below the threshold, they'll show up here.
    def _last_bin_mass(k, dists):
        vals = np.asarray(dists.get(k, []), dtype=float)
        return float(np.mean(vals >= 95.0)) if vals.size else 0.0

    diag_min_n = 20
    ex_prod = {'hypothetical protein', 'unknown function', ''}
    diag_prods = [
        k for k in set(prod_pct_nt) | set(prod_pct_cd)
        if k and k.lower() not in ex_prod
        and len(prod_pct_nt.get(k, [])) >= diag_min_n
        and len(prod_pct_cd.get(k, [])) >= diag_min_n
    ]
    diag_scored = [
        (0.5 * (_last_bin_mass(k, prod_pct_nt) +
                _last_bin_mass(k, prod_pct_cd)),
         len(prod_pct_nt.get(k, [])), k)
        for k in diag_prods
    ]
    diag_scored.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Diagnostic: top-10 products by last-bin "
                f"(rank 95-100) mass at min_n={diag_min_n} "
                f"({len(diag_scored)} candidates):")
    for score, n, k in diag_scored[:10]:
        logger.info(f"    {score:.3f}  n={n:5d}  {k}")

    ax_g = fig.add_subplot(gs[3, 0])
    _plot_histogram_ridges(ax_g, cat_pct_nt, cat_pct_cd,
                           'G. Per-protein Δ-rank within genome — category',
                           _cat_color, ordered_keys=cat_order_gh)

    ax_h = fig.add_subplot(gs[3, 1])
    _plot_histogram_ridges(ax_h, prod_pct_nt, prod_pct_cd,
                           'H. Per-protein Δ-rank within genome — product',
                           _prod_color, ordered_keys=prod_order_gh)

    # Shared legend below the figure: category colours (C/D bars) on the
    # left, scramble-type alpha swatches + expected line on the right.
    from matplotlib.lines import Line2D as _Line2DLocal
    legend_elements = [
        Patch(facecolor=COL_STRUCT, alpha=0.75, label='Structural'),
        Patch(facecolor=COL_MORON,  alpha=0.75, label='Moron/AMG'),
        Patch(facecolor=COL_OTHER,  alpha=0.75, label='Other'),
        Patch(facecolor=COL_GREY,   alpha=0.85, label='Nucleotide'),
        Patch(facecolor=COL_GREY,   alpha=0.4,  label='Codon'),
        _Line2DLocal([0], [0], color=COLORS['text_light'], linewidth=1,
                     linestyle='--', label='Expected (1.0)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
               fontsize=LEGEND_FS, frameon=False,
               bbox_to_anchor=(0.5, -0.03))

    path = os.path.join(output_dir, 'importance_combined.png')
    _save_figure(fig, path, dpi=200)
    logger.info(f"Combined figure saved to {path}")

    # ── Reduced figure (C D E F only) ────────────────────────────────────
    # Same 70% scaling convention as the full figure: two rows instead of
    # four, so height is halved. Helpers are closures so we can reuse
    # _plot_paired_bars and _plot_ridges directly.
    fig_r = plt.figure(figsize=(14 * 0.7, 9.5 * 0.7))
    gs_r = gridspec.GridSpec(2, 2, figure=fig_r,
                             hspace=0.35, wspace=0.6,
                             left=0.1, right=0.96, top=0.93, bottom=0.15)

    ax_c_r = fig_r.add_subplot(gs_r[0, 0])
    _plot_paired_bars(ax_c_r, enrich_cat_nt, enrich_cat_cd,
                      'A. Worst protein / genome — category',
                      _cat_color, is_product=False)

    ax_d_r = fig_r.add_subplot(gs_r[0, 1])
    _plot_paired_bars(ax_d_r, enrich_prod_nt, enrich_prod_cd,
                      'B. Worst product / genome',
                      _prod_color, is_product=True, max_labels=10)

    ax_e_r = fig_r.add_subplot(gs_r[1, 0])
    _plot_ridges(ax_e_r, cat_ratios_nt, cat_ratios_cd,
                 'C. Per-genome concentration — category',
                 _cat_color, min_genomes=50)

    ax_f_r = fig_r.add_subplot(gs_r[1, 1])
    _plot_ridges(ax_f_r, prod_ratios_nt, prod_ratios_cd,
                 'D. Per-genome concentration — product',
                 _prod_color, min_genomes=50, max_rows=10,
                 exclude_keys=('hypothetical protein', 'unknown function', ''))

    legend_elements_r = [
        Patch(facecolor=COL_STRUCT, alpha=0.75, label='Structural'),
        Patch(facecolor=COL_MORON,  alpha=0.75, label='Moron/AMG'),
        Patch(facecolor=COL_OTHER,  alpha=0.75, label='Other'),
        Patch(facecolor=COL_GREY,   alpha=0.85, label='Nucleotide'),
        Patch(facecolor=COL_GREY,   alpha=0.4,  label='Codon'),
        _Line2DLocal([0], [0], color=COLORS['text_light'], linewidth=1,
                     linestyle='--', label='Expected (1.0)'),
    ]
    fig_r.legend(handles=legend_elements_r, loc='lower center', ncol=6,
                 fontsize=LEGEND_FS, frameon=False,
                 bbox_to_anchor=(0.5, 0.02))

    path_r = os.path.join(output_dir, 'importance_reduced.png')
    _save_figure(fig_r, path_r, dpi=200)
    logger.info(f"Reduced figure saved to {path_r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _log_structural_baseline(results: List[dict]):
    """Log baseline score comparison between genomes with/without structural genes."""
    structural_cats = {'head and packaging', 'tail', 'connector'}

    # Check if categories are present
    has_cats = any(r.get('category', 'unknown function') != 'unknown function'
                   for r in results)
    if not has_cats:
        return

    # Group by genome, check if any protein is structural (nt scramble, rep 0)
    genome_info = defaultdict(lambda: {'baseline': None, 'has_structural': False})
    for r in results:
        if r['scramble_type'] == 'nucleotide' and r.get('replicate', 0) == 0:
            gid = r['genome_id']
            genome_info[gid]['baseline'] = r['baseline_top']
            if r.get('category', '').lower() in structural_cats:
                genome_info[gid]['has_structural'] = True

    with_struct = [g['baseline'] for g in genome_info.values()
                   if g['has_structural'] and g['baseline'] is not None]
    without_struct = [g['baseline'] for g in genome_info.values()
                      if not g['has_structural'] and g['baseline'] is not None]

    if with_struct and without_struct:
        logger.info("Baseline score comparison (structural vs non-structural):")
        logger.info(f"  With structural genes:    n={len(with_struct):4d}, "
                    f"median={np.median(with_struct):.4f}, "
                    f"mean={np.mean(with_struct):.4f}")
        logger.info(f"  Without structural genes: n={len(without_struct):4d}, "
                    f"median={np.median(without_struct):.4f}, "
                    f"mean={np.mean(without_struct):.4f}")
        diff = np.median(with_struct) - np.median(without_struct)
        logger.info(f"  Median difference: {diff:+.4f} "
                    f"({'phages more confident' if diff > 0 else 'MGEs more confident'})")


def _read_results_tsv(tsv_path: str) -> List[dict]:
    """Read back a previously saved importance.tsv with correct types."""
    results = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            for key in ('protein_idx', 'protein_start', 'protein_end',
                        'protein_strand', 'protein_length_nt',
                        'genome_length', 'replicate', 'top_class_idx'):
                if key in row and row[key] != '':
                    row[key] = int(float(row[key]))
            for key in ('baseline_top', 'perturbed_top', 'delta_top',
                        'max_pos_delta', 'max_pos_baseline',
                        'max_neg_delta', 'max_neg_baseline'):
                if key in row and row[key] != '':
                    row[key] = float(row[key])
            results.append(row)
    return results


def _read_cumulative_tsv(tsv_path: str) -> List[dict]:
    """Read back importance_cumulative.tsv with correct types."""
    if not os.path.exists(tsv_path):
        return []
    results = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            for key in ('top_k', 'total_scrambled_nt'):
                if key in row and row[key] != '':
                    row[key] = int(float(row[key]))
            for key in ('baseline_top', 'perturbed_top', 'delta_top',
                        'sum_individual_deltas', 'frac_scrambled'):
                if key in row and row[key] != '':
                    row[key] = float(row[key])
            results.append(row)
    return results


def _run_inference(args, tsv_path: str, scores_path: str,
                   rng: random.Random) -> List[dict]:
    """Load model + data, run all scrambling experiments, save TSV + scores."""

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model, calib = load_model_and_calibration(
        args.model_dir, args.checkpoint, device)

    patch_nt_len = calib['model_config']['patch_nt_len']
    stride = patch_nt_len  # non-overlapping for speed
    host_names = calib['hosts']
    num_classes = len(host_names)
    tokenizer = CodonTokenizer()

    # Build per-class temperature vector when split temperatures are available
    has_bact = num_classes > 0 and host_names[-1] == 'bacterial_fragment'
    if calib.get('temperature_host') is not None and has_bact:
        T_host = calib['temperature_host']
        T_bact = calib.get('temperature_bacterial', T_host)
        temperature = torch.ones(num_classes)
        temperature[:num_classes - 1] = T_host
        temperature[num_classes - 1:] = T_bact
    else:
        temperature = calib.get('temperature', 1.0)

    logger.info(f"Model loaded: {num_classes} classes, "
                f"patch_nt_len={patch_nt_len}")
    logger.info("Scrambling mode: protein nt/codon scrambles preserve "
                "start and stop codons (first and last 3 nt / codons of "
                "each CDS); random-region control shuffles flatly.")

    # ---- Load dataset ----------------------------------------------------
    logger.info(f"Loading dataset from {args.dataset_dir} ...")
    records = load_dataset(args.dataset_dir)
    logger.info(f"  {len(records)} training sequences loaded")

    records = subsample(records, args.n_genera, args.n_per_genus, rng)
    logger.info(f"  Subsampled to {len(records)} genomes "
                f"({args.n_per_genus}/genus x {args.n_genera} genera)")

    # ---- Load pharokka annotations (optional) ----------------------------
    pharokka = None
    if args.protein_annotations:
        pharokka = load_pharokka_annotations(args.protein_annotations)

    # ---- Main loop -------------------------------------------------------
    results = []
    cumulative_results = []
    skipped_low = 0
    skipped_noprot = 0
    total_proteins = 0
    all_baselines = []
    all_perturbed = []

    for gi, rec in enumerate(records):
        seq = str(rec['seq'])
        genome_id = rec['id']

        if gi % 50 == 0:
            logger.info(f"  [{gi+1}/{len(records)}] {genome_id} "
                        f"({len(seq):,} nt, {total_proteins} proteins so far)")

        baseline = get_scores(model, tokenizer, seq, patch_nt_len,
                              stride, temperature, device)
        if baseline is None:
            skipped_low += 1
            continue

        top_class = int(np.argmax(baseline))
        top_score = float(baseline[top_class])
        genome_length = len(seq)

        if top_score < args.min_score:
            skipped_low += 1
            continue

        all_baselines.append(baseline)
        baseline_idx = len(all_baselines) - 1

        # Get proteins: pharokka annotations or pyrodigal fallback
        # Try both original and sanitized ID (colons replaced for phold compat)
        pharokka_key = None
        if pharokka:
            if genome_id in pharokka:
                pharokka_key = genome_id
            else:
                sanitized_id = genome_id.replace(':', '_')
                if sanitized_id in pharokka:
                    pharokka_key = sanitized_id

        if pharokka_key:
            proteins = pharokka[pharokka_key]
        else:
            if pharokka:
                logger.debug(f"  {genome_id}: not in pharokka annotations, "
                             f"falling back to pyrodigal")
            proteins = call_proteins(seq)
        if not proteins:
            skipped_noprot += 1
            continue

        if args.max_proteins > 0 and len(proteins) > args.max_proteins:
            proteins = rng.sample(proteins, args.max_proteins)

        for pi, prot in enumerate(proteins):
            p_start = prot['begin'] - 1  # 0-based half-open
            p_end = prot['end']
            p_len = p_end - p_start

            for rep in range(args.n_replicates):
                # Shared fields for all three scramble types
                base = {
                    'genome_id': genome_id,
                    'baseline_idx': baseline_idx,
                    'protein_idx': pi,
                    'protein_start': prot['begin'],
                    'protein_end': prot['end'],
                    'protein_strand': prot['strand'],
                    'protein_length_nt': p_len,
                    'genome_length': genome_length,
                    'category': prot.get('category', 'unknown function'),
                    'product': prot.get('product', ''),
                    'replicate': rep,
                    'top_class': host_names[top_class],
                    'top_class_idx': top_class,
                    'baseline_top': top_score,
                }

                # --- Nucleotide scramble ---
                seq_nt = scramble_nucleotides(seq, p_start, p_end, rng)
                scores_nt = get_scores(model, tokenizer, seq_nt,
                                       patch_nt_len, stride, temperature,
                                       device)
                if scores_nt is not None:
                    delta_all = scores_nt - baseline
                    ipos = int(np.argmax(delta_all))
                    ineg = int(np.argmin(delta_all))
                    results.append({**base, 'scramble_type': 'nucleotide',
                                    'perturbed_top': float(scores_nt[top_class]),
                                    'delta_top': float(scores_nt[top_class]) - top_score,
                                    'max_pos_delta': float(delta_all[ipos]),
                                    'max_pos_class': host_names[ipos],
                                    'max_pos_baseline': float(baseline[ipos]),
                                    'max_neg_delta': float(delta_all[ineg]),
                                    'max_neg_class': host_names[ineg],
                                    'max_neg_baseline': float(baseline[ineg]),
                                    })
                    all_perturbed.append(scores_nt)

                # --- Codon scramble ---
                seq_cd = scramble_codons(seq, p_start, p_end,
                                         prot['strand'], rng)
                scores_cd = get_scores(model, tokenizer, seq_cd,
                                       patch_nt_len, stride, temperature,
                                       device)
                if scores_cd is not None:
                    delta_all = scores_cd - baseline
                    ipos = int(np.argmax(delta_all))
                    ineg = int(np.argmin(delta_all))
                    results.append({**base, 'scramble_type': 'codon',
                                    'perturbed_top': float(scores_cd[top_class]),
                                    'delta_top': float(scores_cd[top_class]) - top_score,
                                    'max_pos_delta': float(delta_all[ipos]),
                                    'max_pos_class': host_names[ipos],
                                    'max_pos_baseline': float(baseline[ipos]),
                                    'max_neg_delta': float(delta_all[ineg]),
                                    'max_neg_class': host_names[ineg],
                                    'max_neg_baseline': float(baseline[ineg]),
                                    })
                    all_perturbed.append(scores_cd)

                # --- Random region control ---
                seq_rnd = scramble_random_region(seq, p_len, rng,
                                                 excluded_start=p_start,
                                                 excluded_end=p_end)
                scores_rnd = get_scores(model, tokenizer, seq_rnd,
                                        patch_nt_len, stride, temperature,
                                        device)
                if scores_rnd is not None:
                    delta_all = scores_rnd - baseline
                    ipos = int(np.argmax(delta_all))
                    ineg = int(np.argmin(delta_all))
                    results.append({**base, 'scramble_type': 'random',
                                    'perturbed_top': float(scores_rnd[top_class]),
                                    'delta_top': float(scores_rnd[top_class]) - top_score,
                                    'max_pos_delta': float(delta_all[ipos]),
                                    'max_pos_class': host_names[ipos],
                                    'max_pos_baseline': float(baseline[ipos]),
                                    'max_neg_delta': float(delta_all[ineg]),
                                    'max_neg_class': host_names[ineg],
                                    'max_neg_baseline': float(baseline[ineg]),
                                    })
                    all_perturbed.append(scores_rnd)

            total_proteins += 1

        # ---- Cumulative scrambling: top-K most impactful proteins --------
        # Collect per-protein nt deltas for this genome
        genome_nt_deltas = []
        for r in results:
            if (r['genome_id'] == genome_id and
                    r['scramble_type'] == 'nucleotide' and
                    r['replicate'] == 0):
                genome_nt_deltas.append((r['delta_top'], r['protein_idx']))

        if len(genome_nt_deltas) >= 2:
            # Sort by delta ascending (most negative first)
            genome_nt_deltas.sort(key=lambda x: x[0])
            ranked_indices = [idx for _, idx in genome_nt_deltas]

            for top_k in [2, 3, 4, 5]:
                if top_k > len(ranked_indices):
                    break
                # Scramble top-k proteins simultaneously
                seq_cum = seq
                total_scrambled_nt = 0
                for pi_rank in ranked_indices[:top_k]:
                    prot = proteins[pi_rank]
                    p_start = prot['begin'] - 1
                    p_end = prot['end']
                    seq_cum = scramble_nucleotides(seq_cum, p_start, p_end, rng)
                    total_scrambled_nt += p_end - p_start

                scores_cum = get_scores(model, tokenizer, seq_cum,
                                        patch_nt_len, stride, temperature,
                                        device)
                if scores_cum is not None:
                    # Sum of individual deltas for additivity comparison
                    sum_individual = sum(d for d, _ in genome_nt_deltas[:top_k])
                    cumulative_results.append({
                        'genome_id': genome_id,
                        'top_k': top_k,
                        'baseline_top': top_score,
                        'perturbed_top': float(scores_cum[top_class]),
                        'delta_top': float(scores_cum[top_class]) - top_score,
                        'sum_individual_deltas': sum_individual,
                        'total_scrambled_nt': total_scrambled_nt,
                        'frac_scrambled': total_scrambled_nt / genome_length
                                          if genome_length > 0 else 0,
                        'top_class': host_names[top_class],
                    })

    logger.info(f"Done. {total_proteins} proteins across {len(all_baselines)} "
                f"genomes. {skipped_low} skipped (low score), "
                f"{skipped_noprot} skipped (no proteins).")
    logger.info(f"  {len(results)} result rows total.")

    if not results:
        logger.warning("No results — nothing to write.")
        return results, cumulative_results

    # ---- Write TSV -------------------------------------------------------
    fieldnames = ['genome_id', 'protein_idx', 'protein_start', 'protein_end',
                  'protein_strand', 'protein_length_nt', 'genome_length',
                  'category', 'product', 'scramble_type', 'replicate',
                  'top_class',
                  'baseline_top', 'perturbed_top', 'delta_top',
                  'max_pos_delta', 'max_pos_class', 'max_pos_baseline',
                  'max_neg_delta', 'max_neg_class', 'max_neg_baseline']

    with open(tsv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t',
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"TSV written to {tsv_path}")

    # ---- Write cumulative TSV --------------------------------------------
    cum_tsv = os.path.join(os.path.dirname(tsv_path), 'importance_cumulative.tsv')
    if cumulative_results:
        cum_fields = ['genome_id', 'top_k', 'baseline_top', 'perturbed_top',
                      'delta_top', 'sum_individual_deltas',
                      'total_scrambled_nt', 'frac_scrambled', 'top_class']
        with open(cum_tsv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cum_fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(cumulative_results)
        logger.info(f"Cumulative TSV written to {cum_tsv} "
                    f"({len(cumulative_results)} rows)")

    # ---- Save full score arrays ------------------------------------------
    np.savez_compressed(
        scores_path,
        baselines=np.array(all_baselines),
        perturbed=np.array(all_perturbed) if all_perturbed else np.array([]),
        host_names=np.array(host_names),
    )
    logger.info(f"Full score matrices saved to {scores_path}")

    return results, cumulative_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Protein importance via systematic scrambling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_dir', default=None,
                        help='Dataset directory containing train.fna.gz and '
                             'phages_hosts_tani.csv')
    parser.add_argument('--model_dir', default=None,
                        help='Model directory with calibration.json + checkpoints/')
    parser.add_argument('--checkpoint', default=None,
                        help='Specific checkpoint path (default: auto-detect)')
    parser.add_argument('--output', '-o', default='importance_results',
                        help='Output directory for TSV + plot')
    parser.add_argument('--recompute', action='store_true',
                        help='Force recomputation even if results exist')
    parser.add_argument('--protein_annotations', default=None,
                        help='Pharokka CDS TSV (pharokka_cds_final_merged_output.tsv). '
                             'When provided, uses pharokka CDS instead of pyrodigal '
                             'and carries functional categories through the analysis.')

    # Subsampling
    parser.add_argument('--n_genera', type=int, default=100,
                        help='Number of top genera to include')
    parser.add_argument('--n_per_genus', type=int, default=10,
                        help='Genomes to sample per genus')

    # Analysis
    parser.add_argument('--max_proteins', type=int, default=0,
                        help='Max proteins to scramble per genome (0 = all)')
    parser.add_argument('--min_score', type=float, default=0.3,
                        help='Min baseline top-class score to include a genome')
    parser.add_argument('--n_replicates', type=int, default=1,
                        help='Scrambling replicates per protein (for variance)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')

    # Locality
    parser.add_argument('--target_delta', type=float, default=-0.3,
                        help='Cumulative score drop threshold for locality '
                             'analysis (finds min proteins to reach this)')
    parser.add_argument('--n_permutations', type=int, default=1000,
                        help='Permutations for locality null distribution')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    rng = random.Random(args.seed)
    os.makedirs(args.output, exist_ok=True)

    tsv_path = os.path.join(args.output, 'importance.tsv')
    cum_tsv_path = os.path.join(args.output, 'importance_cumulative.tsv')
    scores_path = os.path.join(args.output, 'importance_scores.npz')

    # ==================================================================
    # Decide: load cached results or compute from scratch
    # ==================================================================
    if os.path.exists(tsv_path) and not args.recompute:
        logger.info(f"Found existing {tsv_path} — loading cached results. "
                    f"(Use --recompute to force re-running inference.)")
        results = _read_results_tsv(tsv_path)
        cumulative_results = _read_cumulative_tsv(cum_tsv_path)
        logger.info(f"  Loaded {len(results)} rows, "
                    f"{len(cumulative_results)} cumulative rows")

    else:
        if not args.dataset_dir or not args.model_dir:
            parser.error("--dataset_dir and --model_dir are required "
                         "when no cached results exist (or with --recompute)")

        results, cumulative_results = _run_inference(
            args, tsv_path, scores_path, rng)

    # ==================================================================
    # Paired analysis + plots (always runs, regardless of source)
    # ==================================================================
    logger.info("Building paired analysis ...")
    paired = build_paired_table(results)
    logger.info(f"  {len(paired)} paired observations")

    if paired:
        # Save paired table
        paired_path = os.path.join(args.output, 'importance_paired.tsv')
        paired_fields = [
            'genome_id', 'protein_idx', 'protein_start', 'protein_end',
            'protein_length_nt', 'genome_length', 'frac_of_genome',
            'category', 'product', 'replicate', 'baseline_top', 'top_class',
            'delta_nt', 'delta_cd', 'delta_rnd',
            'paired_nt', 'paired_cd', 'norm_paired_nt', 'norm_paired_cd',
        ]
        with open(paired_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=paired_fields,
                                    delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(paired)
        logger.info(f"Paired TSV written to {paired_path}")

        # Summary stats
        pnt = [p['paired_nt'] for p in paired]
        pcd = [p['paired_cd'] for p in paired]
        logger.info(f"  Paired \u0394 nucleotide: "
                    f"median={np.median(pnt):.4f}  "
                    f"mean={np.mean(pnt):.4f}")
        logger.info(f"  Paired \u0394 codon:      "
                    f"median={np.median(pcd):.4f}  "
                    f"mean={np.mean(pcd):.4f}")

    # ---- Locality analysis -----------------------------------------------
    # Computed before plotting so the spatial clustering panel in
    # plot_combined can consume the top_k=2..5 rows.
    logger.info("Running locality analysis ...")
    loc_results = locality_analysis(
        results, cumulative_results,
        target_delta=args.target_delta,
        n_permutations=args.n_permutations, seed=args.seed)
    logger.info(f"  {len(loc_results)} genome/method combinations tested")

    if loc_results:
        loc_path = os.path.join(args.output, 'importance_locality.tsv')
        loc_fields = ['genome_id', 'method', 'k', 'min_k',
                      'cumulative_delta', 'observed_mnn',
                      'null_mean', 'null_std', 'z_score', 'p_value',
                      'n_adjacent', 'expected_adjacent', 'adj_p_value',
                      'genome_length', 'n_proteins_total']
        with open(loc_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=loc_fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(loc_results)
        logger.info(f"Locality TSV written to {loc_path}")

        for method in sorted(set(r['method'] for r in loc_results)):
            zs = [r['z_score'] for r in loc_results
                  if r['method'] == method]
            if zs:
                logger.info(f"  {method}: median z={np.median(zs):.2f}, "
                            f"clustered (z<-1.96): "
                            f"{sum(1 for z in zs if z < -1.96)}/{len(zs)}")

    # ---- Plots -----------------------------------------------------------
    plot_combined(results, cumulative_results, loc_results, args.output)

    # ---- Baseline score comparison: structural vs non-structural ---------
    _log_structural_baseline(results)


if __name__ == '__main__':
    main()
