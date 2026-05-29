#!/usr/bin/env python3
"""Align test phages to host bacterial genomes via minimap2.

For each genus that is both (a) listed as a host in the phage dataset
and (b) represented as a training genome in the bacterial spike-in set,
this script:

    1. Collects the test phages whose host_genus_lineage ends in that genus.
    2. Runs minimap2 (-x asm20, 10 threads) against that genus's
       bacterial genome.
    3. Filters alignments by query coverage fraction (the share of the
       phage covered by the alignment), default ≥0.2.
    4. Tallies how many phages aligned to the *train* region vs the *val*
       region of the genome. The val region was held out from bacterial
       model training, so alignments landing there are the cleanest signal
       of genuine phage→host nucleotide homology — as opposed to potential
       leakage from train-region content seen during training.

Inputs
------
--bacterial_species_tsv : the bacterial_species.tsv written by a training
                          run (logs/bacterial_species.tsv). Provides per
                          genus the chosen species, genome length, and
                          val region coordinates.
--dataset_path          : phage dataset dir containing test.fna.gz and
                          phages_hosts.csv.
--host_genome_dir       : bacterial genome dir containing
                          host_genome_manifest.tsv.
--output_dir            : where per-genus PAFs, per-genus query FASTAs,
                          and the summary TSV are written.

Outputs
-------
output_dir/
├── per_genus/<genus>.paf              # minimap2 PAF with CIGAR + cs
├── per_genus_phages/<genus>.fa        # query FASTA used for that genus
└── summary.tsv                        # per-genus train/val counts

Resume: a genus whose PAF already exists and is non-empty is skipped
(the summary is still recomputed from the on-disk PAF). Delete the PAF
to force re-alignment.

Assumes minimap2 is on PATH and the bacterial set has one species per
genus; multi-species genera fall back to the first row in the TSV.
"""

import argparse
import gzip
import logging
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# Fixed minimap2 settings.
# asm20: assembly-to-assembly preset tolerating up to ~20% divergence —
# the right regime for phage–host alignments (prophages, integrated MGEs,
# shared genes) which typically sit in the 5–20% divergence range.
# -c --cs: emit CIGAR + difference strings for downstream parsing.
MINIMAP2_PRESET = 'asm20'
MINIMAP2_THREADS = 10


# ============================================================================
# Parsing
# ============================================================================

def read_fasta_with_headers(path: str) -> List[Tuple[str, str]]:
    """Read a (possibly gzipped) FASTA and return ``[(header, seq), ...]``.

    The header is the first whitespace-delimited token after ``>``.
    """
    opener = gzip.open if path.endswith('.gz') else open
    records: List[Tuple[str, str]] = []
    header = None
    parts: List[str] = []
    with opener(path, 'rt') as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(parts)))
                header = line[1:].split()[0]
                parts = []
            elif line:
                parts.append(line.upper())
        if header is not None:
            records.append((header, ''.join(parts)))
    return records


def parse_bacterial_species_tsv(path: str) -> Dict[str, dict]:
    """Parse the run's bacterial_species.tsv → ``{genus: meta}``.

    When a genus has multiple species, keep the first row encountered
    (TSV file order) and log a summary of the others.
    """
    df = pd.read_csv(path, sep='\t')
    out: Dict[str, dict] = {}
    extras: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        g = row['genus']
        if g in out:
            extras.setdefault(g, []).append(row['species'])
        else:
            out[g] = {
                'species': row['species'],
                'genome_len': int(row['genome_len']),
                'val_start': int(row['val_start']),
                'val_end': int(row['val_end']),
            }
    if extras:
        n_extra = sum(len(v) for v in extras.values())
        logger.warning(f'  {len(extras)} genera have multiple species '
                       f'({n_extra} extras); using the first per genus.')
        for g, sps in list(extras.items())[:5]:
            logger.warning(f'    {g}: kept {out[g]["species"]!r}, '
                           f'ignored {sps}')
        if len(extras) > 5:
            logger.warning(f'    ... ({len(extras) - 5} more)')
    logger.info(f'  bacterial_species.tsv: {len(out)} unique genera')
    return out


def parse_host_genome_manifest(host_genome_dir: str) -> Dict[str, str]:
    """Parse host_genome_manifest.tsv → ``{species: absolute_genome_path}``."""
    path = os.path.join(host_genome_dir, 'host_genome_manifest.tsv')
    df = pd.read_csv(path, sep='\t')
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        p = row['genome_path']
        if not os.path.isabs(p):
            p = os.path.join(host_genome_dir, p)
        out[row['species']] = p
    logger.info(f'  manifest: {len(out)} species → genome paths')
    return out


def get_test_phages_by_genus(
        dataset_path: str
) -> Tuple[List[Tuple[str, str]], Dict[str, List[int]]]:
    """Load test phages and group them by host genus.

    Returns:
        records: ``[(header, sequence), ...]`` for in_testset==1 rows, in
                 the same positional order as the CSV / FASTA.
        genus_to_idx: ``{genus: sorted [indices into records]}``. A phage
                      with multiple hosts appears under each of its
                      host genera.
    """
    csv_path = os.path.join(dataset_path, 'phages_hosts.csv')
    fasta_path = os.path.join(dataset_path, 'test.fna.gz')

    df = pd.read_csv(csv_path, delimiter=',', dtype=str)
    test_df = df[df['in_testset'] == '1'].reset_index(drop=True)
    logger.info(f'  test phages in CSV: {len(test_df)}')

    records = read_fasta_with_headers(fasta_path)
    logger.info(f'  test sequences in FASTA: {len(records)}')

    if len(records) != len(test_df):
        raise ValueError(
            f'Test FASTA count ({len(records)}) != in_testset==1 row '
            f'count ({len(test_df)}) — positional alignment broken.')

    headers = [h for h, _ in records]
    if len(set(headers)) != len(headers):
        raise ValueError(
            f'Test FASTA contains duplicate headers; minimap2 query '
            f'names would collide. Fix the FASTA before re-running.')

    # Group test phages by trailing genus of each host_genus_lineage entry.
    # A phage with several hosts contributes to several groups; dedupe
    # within a group in case the same genus appears twice in its lineages.
    genus_to_idx_set: Dict[str, set] = {}
    for i, lineages in enumerate(test_df['host_genus_lineage']):
        if not isinstance(lineages, str) or not lineages.strip():
            continue
        for lin in lineages.split('|'):
            genus = lin.split(';')[-1] if ';' in lin else lin
            genus_to_idx_set.setdefault(genus, set()).add(i)
    genus_to_idx = {g: sorted(s) for g, s in genus_to_idx_set.items()}

    n_with_host = sum(1 for ls in test_df['host_genus_lineage']
                      if isinstance(ls, str) and ls.strip())
    logger.info(f'  genera referenced in test set: {len(genus_to_idx)} '
                f'({n_with_host}/{len(test_df)} phages have ≥1 host)')

    return records, genus_to_idx


# ============================================================================
# Alignment + PAF summarisation
# ============================================================================

def write_phage_fasta(path: str, records: List[Tuple[str, str]],
                      indices: List[int]) -> None:
    """Write the selected phage records out as FASTA (80-char wrapped)."""
    with open(path, 'w') as fh:
        for idx in indices:
            h, s = records[idx]
            fh.write(f'>{h}\n')
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + '\n')


def run_minimap2(genome_path: str, phages_fasta: str,
                 output_paf: str) -> None:
    """Run minimap2 with the fixed asm20 / -c --cs / 10-thread settings."""
    cmd = ['/home/groups/VEO/tools/minimap2/v2.26/minimap2',
           '-x', MINIMAP2_PRESET,
           '-c', '--cs',
           '-t', str(MINIMAP2_THREADS),
           genome_path, phages_fasta]
    with open(output_paf, 'w') as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE,
                                text=True)
    if result.returncode != 0:
        err_tail = '\n'.join(result.stderr.strip().split('\n')[-3:])
        raise RuntimeError(
            f'minimap2 failed (exit {result.returncode}): {err_tail}')


def classify_alignment_region(tstart: int, tend: int,
                              val_start: int, val_end: int,
                              genome_len: int) -> str:
    """Classify an alignment by majority bp overlap with train vs val."""
    overlap_val = max(0, min(tend, val_end) - max(tstart, val_start))
    overlap_train1 = max(0, min(tend, val_start) - max(tstart, 0))
    overlap_train2 = max(0, min(tend, genome_len) - max(tstart, val_end))
    overlap_train = overlap_train1 + overlap_train2
    return 'val' if overlap_val > overlap_train else 'train'


def summarise_paf(paf_path: str, val_start: int, val_end: int,
                  genome_len: int, n_queried: int,
                  min_query_aln_frac: float = 0.2) -> dict:
    """Tally per-phage and per-alignment train/val hits from a PAF.

    Only alignments covering at least ``min_query_aln_frac`` of the query
    (phage) length — i.e. ``(qend - qstart) / qlen >= cutoff`` — are
    counted. Sub-threshold alignments are dropped silently. The raw PAF
    is left untouched, so re-tallying with a different cutoff later is
    cheap.

    A phage is counted as a "train-hit" phage if at least one of its
    passing alignments has its majority bp overlap in a train region
    (analogously for val). A phage can therefore be in both buckets —
    the disjoint ``n_phages_with_both`` column makes that explicit.
    """
    train_aln = 0
    val_aln = 0
    n_total = 0
    n_dropped = 0
    phages_with_train: set = set()
    phages_with_val: set = set()

    with open(paf_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            cols = line.split('\t')
            qname = cols[0]
            qlen = int(cols[1])
            qstart = int(cols[2])
            qend = int(cols[3])
            tstart = int(cols[7])
            tend = int(cols[8])
            n_total += 1
            qaln_frac = (qend - qstart) / max(qlen, 1)
            if qaln_frac < min_query_aln_frac:
                n_dropped += 1
                continue
            region = classify_alignment_region(
                tstart, tend, val_start, val_end, genome_len)
            if region == 'val':
                val_aln += 1
                phages_with_val.add(qname)
            else:
                train_aln += 1
                phages_with_train.add(qname)

    any_hit = phages_with_train | phages_with_val
    both = phages_with_train & phages_with_val
    return {
        'n_phages_queried': n_queried,
        'n_phages_with_any_hit': len(any_hit),
        'n_phages_with_train_hit': len(phages_with_train),
        'n_phages_with_val_hit': len(phages_with_val),
        'n_phages_with_both': len(both),
        'n_alignments_train': train_aln,
        'n_alignments_val': val_aln,
        'n_alignments_dropped_below_cutoff': n_dropped,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Align test phages to host bacterial genomes via minimap2.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bacterial_species_tsv', required=True,
                        help='Path to bacterial_species.tsv written by a training run.')
    parser.add_argument('--dataset_path', required=True,
                        help='Phage dataset dir (test.fna.gz + phages_hosts.csv).')
    parser.add_argument('--host_genome_dir', required=True,
                        help='Bacterial genome dir with host_genome_manifest.tsv.')
    parser.add_argument('--output_dir', required=True,
                        help='Where per-genus PAFs and summary.tsv are written.')
    parser.add_argument('--min_query_aln_frac', type=float, default=0.2,
                        help='Minimum fraction of the phage query length '
                             'that an alignment must cover to be counted '
                             'as a hit. Applied per PAF row at tally time; '
                             'the raw PAF is unfiltered. Set to 0.0 to '
                             'count every minimap2 alignment.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paf_dir = os.path.join(args.output_dir, 'per_genus')
    fa_dir = os.path.join(args.output_dir, 'per_genus_phages')
    os.makedirs(paf_dir, exist_ok=True)
    os.makedirs(fa_dir, exist_ok=True)

    logger.info('Parsing inputs...')
    logger.info(f'  min query alignment fraction: '
                f'{args.min_query_aln_frac}')
    genus_meta = parse_bacterial_species_tsv(args.bacterial_species_tsv)
    species_to_path = parse_host_genome_manifest(args.host_genome_dir)
    test_records, genus_to_idx = get_test_phages_by_genus(args.dataset_path)

    # Inner-join genus → species → genome path. Drop genera whose
    # genome file is missing on disk (warn loudly).
    genus_to_genome: Dict[str, dict] = {}
    for genus, meta in genus_meta.items():
        path = species_to_path.get(meta['species'])
        if path is None:
            logger.warning(f'  Genus {genus}: species '
                           f'{meta["species"]!r} not in manifest, skipping')
            continue
        if not os.path.exists(path):
            logger.warning(f'  Genus {genus}: genome file missing at '
                           f'{path}, skipping')
            continue
        genus_to_genome[genus] = {**meta, 'genome_path': path}

    common = sorted(set(genus_to_idx) & set(genus_to_genome))
    only_phage = set(genus_to_idx) - set(genus_to_genome)
    only_bact = set(genus_to_genome) - set(genus_to_idx)
    covered_phages = set().union(*[genus_to_idx[g] for g in common]) \
        if common else set()
    logger.info(f'  genera to align: {len(common)}')
    logger.info(f'  phage genera with no bacterial reference: '
                f'{len(only_phage)}')
    logger.info(f'  bacterial genera not phage-targeted: {len(only_bact)}')
    logger.info(f'  test phages covered by ≥1 alignable host: '
                f'{len(covered_phages)}/{len(test_records)}')

    # =========================================================================
    # Per-genus alignment
    # =========================================================================
    summary_rows = []
    for i, genus in enumerate(common, 1):
        meta = genus_to_genome[genus]
        phage_indices = genus_to_idx[genus]
        n_phages = len(phage_indices)

        paf_path = os.path.join(paf_dir, f'{genus}.paf')
        fa_path = os.path.join(fa_dir, f'{genus}.fa')

        logger.info(f'[{i}/{len(common)}] {genus}: {n_phages} phages → '
                    f'{meta["species"]} (genome {meta["genome_len"]:,} nt, '
                    f'val [{meta["val_start"]:,}, {meta["val_end"]:,}))')

        if os.path.exists(paf_path) and os.path.getsize(paf_path) > 0:
            logger.info(f'  PAF exists, skipping alignment')
        else:
            write_phage_fasta(fa_path, test_records, phage_indices)
            try:
                run_minimap2(meta['genome_path'], fa_path, paf_path)
            except RuntimeError as e:
                logger.error(f'  {e}')
                continue

        stats = summarise_paf(paf_path, meta['val_start'],
                              meta['val_end'], meta['genome_len'], n_phages,
                              min_query_aln_frac=args.min_query_aln_frac)
        logger.info(
            f'  → {stats["n_phages_with_any_hit"]}/{n_phages} phages with '
            f'≥1 alignment; train-hit {stats["n_phages_with_train_hit"]}, '
            f'val-hit {stats["n_phages_with_val_hit"]} '
            f'({stats["n_alignments_train"]} train / '
            f'{stats["n_alignments_val"]} val alignments)')
        summary_rows.append({
            'genus': genus,
            'bacterial_species': meta['species'],
            'bacterial_genome_len': meta['genome_len'],
            'val_start': meta['val_start'],
            'val_end': meta['val_end'],
            **stats,
        })

    # =========================================================================
    # Summary
    # =========================================================================
    summary_path = os.path.join(args.output_dir, 'summary.tsv')
    df = pd.DataFrame(summary_rows)
    df.to_csv(summary_path, sep='\t', index=False)
    logger.info(f'Summary → {summary_path}')

    if not df.empty:
        logger.info(
            f'Totals across {len(df)} genera: '
            f'{int(df["n_phages_with_any_hit"].sum())} phage-genus pairs '
            f'with alignments, '
            f'{int(df["n_phages_with_train_hit"].sum())} train-hit, '
            f'{int(df["n_phages_with_val_hit"].sum())} val-hit '
            f'({int(df["n_alignments_train"].sum())} train / '
            f'{int(df["n_alignments_val"].sum())} val alignments)')


if __name__ == '__main__':
    main()
