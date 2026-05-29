#!/usr/bin/env python3
"""Compute bacterial-genome regions matched by phage alignments.

One-time preprocessing step. For each bacterial genome in
``host_genome_manifest.tsv``, this script aligns every phage whose
declared host genus matches that genome's genus and records the union
of alignment regions on the genome. The output TSV is then read by
``BacterialGenomeStore`` at training-data-load time to excise those
regions from the bacterial training material — preventing the model
from learning ``this region = bacterial`` from sequence that is in
fact phage-derived (prophages, integrated mobile genetic elements,
shared genes).

Coordinates are emitted in the genome store's concatenated coordinate
system. Concretely: each bacterial FASTA is concatenated contig-by-contig
into a single string via the same ``_read_fasta_concat`` function the
genome store uses, this concatenated string is written to a temporary
single-record FASTA, and minimap2 aligns against *that* — so the
script's output coordinates are directly usable downstream without
any per-contig translation step.

Usage::

    python compute_phage_hit_regions.py \\
        --dataset_path /path/to/phages \\
        --host_genome_dir /path/to/bacterial_genomes \\
        --output_dir analyses/phage_hit_regions

Outputs::

    output_dir/phage_hit_regions.tsv      # the deliverable
    output_dir/per_species/<sanitized>.paf  # raw PAFs, kept for resume

This script supersedes the older ``align_phages_to_hosts.py``.
"""

import argparse
import gzip
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# IMPORTANT: this function must produce byte-for-byte the same string as
# ``dataset._read_fasta_concat`` so the alignment coordinates this script
# emits land in the genome store's coordinate system. If you change one,
# change the other.
def _read_fasta_concat(path: str) -> str:
    """Read a (possibly gzipped) FASTA file and return its concatenated
    sequence as a single uppercase string. Multi-contig genomes are
    concatenated in FASTA file order with no separator."""
    opener = gzip.open if path.endswith('.gz') else open
    parts = []
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                continue
            parts.append(line.strip().upper())
    return ''.join(parts)


# Fixed minimap2 settings.
# asm20: assembly-to-assembly preset tolerating up to ~20% divergence
# — the right regime for phage–host nucleotide homology (prophages,
# integrated MGEs, shared genes) which typically sits in the 5–20% range.
# -c emits CIGAR; we don't need --cs here (no downstream tooling).
MINIMAP2_PRESET = 'asm20'
MINIMAP2_THREADS = 10


# ============================================================================
# FASTA / CSV parsing
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


def parse_host_genome_manifest(host_genome_dir: str) -> List[Tuple[str, str]]:
    """Parse host_genome_manifest.tsv → ``[(species, abs_genome_path), ...]``
    in manifest file order."""
    path = os.path.join(host_genome_dir, 'host_genome_manifest.tsv')
    df = pd.read_csv(path, sep='\t')
    out: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        p = row['genome_path']
        if not os.path.isabs(p):
            p = os.path.join(host_genome_dir, p)
        out.append((row['species'], p))
    logger.info(f'  manifest: {len(out)} species → genome paths')
    return out


def load_all_phages_by_genus(
        dataset_path: str, host_column: str
) -> Tuple[List[Tuple[str, str]], Dict[str, List[int]]]:
    """Load every phage (train + test) and group them by host genus.

    Returns:
        records: ``[(header, sequence), ...]`` across the full CSV, in CSV
                 row order. Train and test FASTAs are interleaved back into
                 CSV order by matching ``in_testset`` rows positionally,
                 mirroring how ``load_phage_host_data`` re-pairs them.
        genus_to_idx: ``{genus: [sorted indices into records]}``.
    """
    csv_path = os.path.join(dataset_path, 'phages_hosts.csv')
    train_fa = os.path.join(dataset_path, 'train.fna.gz')
    test_fa = os.path.join(dataset_path, 'test.fna.gz')

    df = pd.read_csv(csv_path, delimiter=',', dtype=str)
    logger.info(f'  CSV rows: {len(df)} '
                f'({(df["in_testset"] == "0").sum()} train, '
                f'{(df["in_testset"] == "1").sum()} test)')

    train_records = read_fasta_with_headers(train_fa)
    test_records = read_fasta_with_headers(test_fa)
    logger.info(f'  FASTA: {len(train_records)} train + '
                f'{len(test_records)} test sequences')

    n_train_csv = (df['in_testset'] == '0').sum()
    n_test_csv = (df['in_testset'] == '1').sum()
    if len(train_records) != n_train_csv:
        raise ValueError(
            f'train.fna.gz has {len(train_records)} sequences but CSV has '
            f'{n_train_csv} in_testset==0 rows — positional pairing broken.')
    if len(test_records) != n_test_csv:
        raise ValueError(
            f'test.fna.gz has {len(test_records)} sequences but CSV has '
            f'{n_test_csv} in_testset==1 rows — positional pairing broken.')

    # Walk the CSV in order, pulling from train_records or test_records.
    # Result: ``records[i]`` aligns with ``df.iloc[i]``.
    records: List[Tuple[str, str]] = []
    ti, ei = 0, 0
    for _, row in df.iterrows():
        if row['in_testset'] == '1':
            records.append(test_records[ei])
            ei += 1
        else:
            records.append(train_records[ti])
            ti += 1
    assert ti == len(train_records) and ei == len(test_records)

    # Header uniqueness across train+test
    headers = [h for h, _ in records]
    if len(set(headers)) != len(headers):
        raise ValueError(
            'Phage FASTA contains duplicate headers across train+test; '
            'minimap2 query names would collide.')

    # Group by trailing genus of each host_<level> entry
    if host_column not in df.columns:
        raise ValueError(
            f'CSV has no column {host_column!r}. Available: {list(df.columns)}')
    genus_to_idx_set: Dict[str, set] = {}
    for i, lineages in enumerate(df[host_column]):
        if not isinstance(lineages, str) or not lineages.strip():
            continue
        for lin in lineages.split('|'):
            genus = lin.split(';')[-1] if ';' in lin else lin
            genus_to_idx_set.setdefault(genus, set()).add(i)
    genus_to_idx = {g: sorted(s) for g, s in genus_to_idx_set.items()}
    logger.info(f'  distinct host genera: {len(genus_to_idx)}')
    return records, genus_to_idx


# ============================================================================
# Alignment + region merging
# ============================================================================

_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._-]+')


def sanitize_for_filename(s: str) -> str:
    """Replace anything outside ``[A-Za-z0-9._-]`` with underscores so the
    string is safe to use as a filename across filesystems."""
    return _SANITIZE_RE.sub('_', s)


def write_fasta(path: str, records, line_wrap: int = 80) -> None:
    """Write FASTA records (``[(header, seq), ...]``) to ``path`` with
    line wrapping (default 80 cols)."""
    with open(path, 'w') as fh:
        for header, seq in records:
            fh.write(f'>{header}\n')
            for i in range(0, len(seq), line_wrap):
                fh.write(seq[i:i + line_wrap] + '\n')


def run_minimap2(target_fasta: str, query_fasta: str,
                 output_paf: str) -> None:
    """Run minimap2 with the fixed asm20 / -c / 10-thread settings."""
    cmd = ['/home/groups/VEO/tools/minimap2/v2.26/minimap2',
           '-x', MINIMAP2_PRESET,
           '-c',
           '-t', str(MINIMAP2_THREADS),
           target_fasta, query_fasta]
    with open(output_paf, 'w') as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE,
                                text=True)
    if result.returncode != 0:
        err_tail = '\n'.join(result.stderr.strip().split('\n')[-3:])
        raise RuntimeError(
            f'minimap2 failed (exit {result.returncode}): {err_tail}')


def collect_intervals_from_paf(paf_path: str,
                               min_query_aln_frac: float,
                               min_alignment_len: int
                               ) -> List[Tuple[int, int]]:
    """Parse PAF, filter by query coverage fraction *and* alignment
    block length, return target-side intervals as ``[(tstart, tend),
    ...]`` (unsorted, possibly overlapping).

    Two filters apply per row (both must pass):
      - ``(qend - qstart) / qlen >= min_query_aln_frac``
      - ``alignment_block_length >= min_alignment_len`` (PAF column 11:
        matches + mismatches + indels in the alignment matrix)
    """
    intervals: List[Tuple[int, int]] = []
    with open(paf_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            cols = line.split('\t')
            qlen = int(cols[1])
            qstart = int(cols[2])
            qend = int(cols[3])
            tstart = int(cols[7])
            tend = int(cols[8])
            block_len = int(cols[10])
            qaln_frac = (qend - qstart) / max(qlen, 1)
            if qaln_frac >= min_query_aln_frac \
                    and block_len >= min_alignment_len:
                intervals.append((tstart, tend))
    return intervals


def merge_intervals(intervals: List[Tuple[int, int]]
                    ) -> List[Tuple[int, int]]:
    """Merge half-open intervals: sort by start, fuse anything that
    touches or overlaps the running interval."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute bacterial-genome regions matched by phage '
                    'alignments (preprocessing step for bacterial '
                    'training-data excision).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', required=True,
                        help='Phage dataset dir (train.fna.gz, test.fna.gz, '
                             'phages_hosts.csv).')
    parser.add_argument('--host_genome_dir', required=True,
                        help='Bacterial genome dir with '
                             'host_genome_manifest.tsv.')
    parser.add_argument('--output_dir', required=True,
                        help='Where phage_hit_regions.tsv and per-species '
                             'PAFs are written.')
    parser.add_argument('--host_column', type=str,
                        default='host_genus_lineage',
                        help='CSV column with pipe-separated host lineages '
                             '(trailing token taken as the genus).')
    parser.add_argument('--min_query_aln_frac', type=float, default=0.2,
                        help='Minimum fraction of the phage query length '
                             'that an alignment must cover to count. '
                             'Applied per PAF row; raw PAFs are unfiltered.')
    parser.add_argument('--min_alignment_len', type=int, default=2000,
                        help='Minimum alignment block length (PAF column '
                             '11: matches + mismatches + indels) for an '
                             'alignment to count. Combined with '
                             '--min_query_aln_frac via AND. Set to 0 to '
                             'disable.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paf_dir = os.path.join(args.output_dir, 'per_species')
    os.makedirs(paf_dir, exist_ok=True)
    output_tsv = os.path.join(args.output_dir, 'phage_hit_regions.tsv')

    logger.info('Parsing inputs...')
    logger.info(f'  min query alignment fraction: '
                f'{args.min_query_aln_frac}')
    logger.info(f'  min alignment block length:   '
                f'{args.min_alignment_len}')
    manifest = parse_host_genome_manifest(args.host_genome_dir)
    phage_records, genus_to_phage_idx = load_all_phages_by_genus(
        args.dataset_path, args.host_column)

    # Group manifest species by genus, preserving manifest order so the
    # "first species per genus" tie-breaker (irrelevant here, but stable)
    # is deterministic.
    genus_to_species: Dict[str, List[Tuple[str, str]]] = {}
    for species, path in manifest:
        g = species.split()[0]
        genus_to_species.setdefault(g, []).append((species, path))

    # Only process genera that have at least one phage targeting them.
    aligned_genera = sorted(set(genus_to_phage_idx) & set(genus_to_species))
    phage_only = set(genus_to_phage_idx) - set(genus_to_species)
    bact_only = set(genus_to_species) - set(genus_to_phage_idx)
    logger.info(f'  genera with phages and ≥1 bacterial genome: '
                f'{len(aligned_genera)}')
    logger.info(f'  phage genera with no bacterial reference: '
                f'{len(phage_only)}')
    logger.info(f'  bacterial genera not phage-targeted (skipped): '
                f'{len(bact_only)}')

    # Flatten (genus, species, path, phage_indices) for a stable
    # per-species progress count.
    work: List[Tuple[str, str, str, List[int]]] = []
    for genus in aligned_genera:
        idxs = genus_to_phage_idx[genus]
        for species, path in genus_to_species[genus]:
            work.append((genus, species, path, idxs))
    logger.info(f'  total (genus, species) pairs to align: {len(work)}')

    # =========================================================================
    # Per-species alignment
    # =========================================================================
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Cache per-genus phage FASTAs inside the temp dir — multiple
        # species share the same phage set, no point rewriting it.
        genus_phage_fa: Dict[str, str] = {}

        for i, (genus, species, genome_path, phage_indices) in enumerate(
                work, 1):
            sanitized = sanitize_for_filename(species)
            paf_path = os.path.join(paf_dir, f'{sanitized}.paf')

            # Build the temp concatenated genome that mirrors what
            # BacterialGenomeStore loads — so alignment coords land
            # directly in its coordinate system. Empty headers are fine;
            # minimap2 just needs a record name.
            concat_seq = _read_fasta_concat(genome_path)
            genome_len = len(concat_seq)

            logger.info(
                f'[{i}/{len(work)}] {species} (genus {genus}): '
                f'{len(phage_indices)} phages → genome {genome_len:,} nt')

            if os.path.exists(paf_path) and os.path.getsize(paf_path) > 0:
                logger.info(f'  PAF exists, skipping alignment')
            else:
                # Genome FASTA: a single record with the concatenated seq
                concat_fa = os.path.join(
                    tmpdir, f'{sanitized}.concat.fa')
                write_fasta(concat_fa, [(sanitized, concat_seq)])

                # Phage FASTA, cached per genus
                phage_fa = genus_phage_fa.get(genus)
                if phage_fa is None:
                    phage_fa = os.path.join(
                        tmpdir, f'phages.{sanitize_for_filename(genus)}.fa')
                    write_fasta(
                        phage_fa,
                        [phage_records[idx] for idx in phage_indices])
                    genus_phage_fa[genus] = phage_fa

                try:
                    run_minimap2(concat_fa, phage_fa, paf_path)
                except RuntimeError as e:
                    logger.error(f'  {e}')
                    continue
                finally:
                    # Genome temp file can go; PAF is what we care about
                    if os.path.exists(concat_fa):
                        os.remove(concat_fa)

            # Tally intervals
            intervals = collect_intervals_from_paf(
                paf_path, args.min_query_aln_frac, args.min_alignment_len)
            merged = merge_intervals(intervals)
            covered_bp = sum(e - s for s, e in merged)
            coverage_ratio = covered_bp / max(genome_len, 1)
            regions_str = ';'.join(f'{s}-{e}' for s, e in merged)
            logger.info(
                f'  → {len(intervals)} alignments above cutoff '
                f'→ {len(merged)} merged regions covering '
                f'{covered_bp:,} bp ({100 * coverage_ratio:.3f}%)')
            rows.append({
                'species': species,
                'genome_len': genome_len,
                'n_regions': len(merged),
                'coverage_ratio': coverage_ratio,
                'regions': regions_str,
            })

    # =========================================================================
    # Write TSV
    # =========================================================================
    df = pd.DataFrame(rows).sort_values('species').reset_index(drop=True)
    df.to_csv(output_tsv, sep='\t', index=False,
              float_format='%.6f')
    logger.info(f'Wrote {len(df)} rows → {output_tsv}')

    if not df.empty:
        logger.info(
            f'Totals across {len(df)} genomes: '
            f'{int(df["n_regions"].sum())} merged regions, '
            f'mean coverage {100 * df["coverage_ratio"].mean():.3f}% '
            f'(max {100 * df["coverage_ratio"].max():.3f}%)')


if __name__ == '__main__':
    main()
