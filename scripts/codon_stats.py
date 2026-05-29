#!/usr/bin/env python3
"""Phage-host codon usage analysis (genus level, single figure).

Standalone script — not part of the installed package.

Calls genes with pyrodigal on both phage and host genomes, pools
codon counts to the genus level, and produces a single 3x3 figure
summarising phage-host codon adaptation:

    Row 1  Biological decomposition   — AA usage, full codon, RSCU
                                        (matched vs shuffled null)
    Row 2  Position decomposition     — positions 1, 2, 3 per genus
                                        (matched vs shuffled null)
    Row 3  Position decomposition     — positions 1, 2, 3 per phage
                                        against genus-pooled host
                                        (matched vs random-genus null)

Species is the atomic unit: species-level stats are computed first,
then pooled into genus-level profiles. Selection is genus-first:
pick genera that have enough species with enough phages, then
include all qualifying species from those genera.

Use --count_only for a quick diagnostic of the data landscape
without running gene calling.

Requirements:
    pip install pyrodigal

Usage:
    # Quick data landscape summary (no gene calling)
    python codon_stats.py \\
        --dataset_dir ./data \\
        --host_genome_dir ./genomes \\
        --count_only

    # Full analysis
    python codon_stats.py \\
        --dataset_dir ./data \\
        --host_genome_dir ./genomes \\
        --min_phages_per_species 3 \\
        --min_species_per_genus 3 \\
        --max_phages_per_species 50 \\
        --cache codon_cache.pkl \\
        --output codon_stats.png
"""

import argparse
import gzip
import logging
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pyrodigal
from Bio import SeqIO
from scipy import stats as sp_stats

from eval_utils import (
    COLORS, FONT_SIZES, FIG_WIDTH, FIG_HEIGHT_ROW,
    setup_style, _save_figure, _suptitle,
    enable_presentation_mode,
)

logger = logging.getLogger(__name__)

# Genetic code: 61 sense codons (stop codons excluded)
STOP_CODONS = {'TAA', 'TAG', 'TGA'}
CODON_PREFIXES = sorted({a + b for a in 'ACGT' for b in 'ACGT'})  # 16
WOBBLE_BASES = list('ACGT')

CODON_TO_AA = {
    'TTT': 'Phe', 'TTC': 'Phe', 'TTA': 'Leu', 'TTG': 'Leu',
    'CTT': 'Leu', 'CTC': 'Leu', 'CTA': 'Leu', 'CTG': 'Leu',
    'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile', 'ATG': 'Met',
    'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val',
    'TCT': 'Ser', 'TCC': 'Ser', 'TCA': 'Ser', 'TCG': 'Ser',
    'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    'GCT': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    'TAT': 'Tyr', 'TAC': 'Tyr',
    'CAT': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
    'AAT': 'Asn', 'AAC': 'Asn', 'AAA': 'Lys', 'AAG': 'Lys',
    'GAT': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
    'TGT': 'Cys', 'TGC': 'Cys', 'TGG': 'Trp',
    'CGT': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
    'AGT': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
    'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
}
SENSE_CODONS = sorted(CODON_TO_AA.keys())  # 61
AA_LIST = sorted(set(CODON_TO_AA.values()))  # 20 amino acids
# Number of synonymous codons per amino acid
_AA_NSYN = {}
for _c, _a in CODON_TO_AA.items():
    _AA_NSYN.setdefault(_a, 0)
    _AA_NSYN[_a] += 1

# ---------------------------------------------------------------------------
# Gene calling
# ---------------------------------------------------------------------------

def _extract_codons(seq: str, genes, min_codons: int = 30):
    """Extract codon lists from predicted genes.

    Returns list of codon lists (each codon is a 3-char string).
    Excludes partial genes and genes shorter than min_codons.
    """
    codon_lists = []
    for gene in genes:
        if gene.partial_begin or gene.partial_end:
            continue
        start = gene.begin - 1  # 1-based to 0-based
        end = gene.end
        if gene.strand == -1:
            nt = _revcomp(seq[start:end])
        else:
            nt = seq[start:end]
        codons = [nt[i:i+3].upper() for i in range(0, len(nt) - 2, 3)]
        codons = [c for c in codons if len(c) == 3
                  and all(b in 'ACGT' for b in c)
                  and c not in STOP_CODONS]
        if len(codons) >= min_codons:
            codon_lists.append(codons)
    return codon_lists


def call_genes(seq: str, gene_finder: pyrodigal.GeneFinder,
               min_codons: int = 30):
    """Call genes and extract codons using a shared GeneFinder instance.

    The GeneFinder should be created once with ``meta=True`` and
    reused across all sequences.
    """
    genes = gene_finder.find_genes(seq.encode())
    return _extract_codons(seq, genes, min_codons)


_COMP = str.maketrans('ACGTacgt', 'TGCAtgca')

def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


# ---------------------------------------------------------------------------
# Codon statistics
# ---------------------------------------------------------------------------

def codon_counts_from_genes(codon_lists):
    """Aggregate codon counts from a list of codon lists.

    Returns dict {codon: count} for all 61 sense codons.
    """
    counts = defaultdict(int)
    for codons in codon_lists:
        for c in codons:
            counts[c] += 1
    return dict(counts)


def wobble_profile(counts: dict) -> np.ndarray:
    """64-dimensional wobble profile: for each of 16 codon prefixes,
    the fraction of each wobble base (A, C, G, T).

    Returns (64,) array ordered as AA_A, AA_C, AA_G, AA_T, AC_A, ...
    """
    profile = np.zeros(64)
    for pi, prefix in enumerate(CODON_PREFIXES):
        total = sum(counts.get(prefix + b, 0) for b in WOBBLE_BASES)
        if total > 0:
            for bi, base in enumerate(WOBBLE_BASES):
                profile[pi * 4 + bi] = counts.get(prefix + base, 0) / total
    return profile


def position_profile(counts: dict, position: int) -> np.ndarray:
    """64-dimensional profile for any codon position (0, 1, or 2).

    Groups codons by the two OTHER positions, then computes the
    fraction of each nucleotide at the target position.

    Position 0: group by suffix (pos1+pos2), vary pos0
    Position 1: group by pos0+pos2, vary pos1
    Position 2: group by prefix (pos0+pos1), vary pos2  (= wobble_profile)

    Returns (64,) array with 16 groups × 4 nucleotides.
    """
    groups = sorted({_context_key(c, position) for c in counts if len(c) == 3})
    # Pad to 16 if some contexts are missing
    profile = np.zeros(64)
    group_idx = {g: i for i, g in enumerate(groups)}
    for codon, count in counts.items():
        if len(codon) != 3:
            continue
        ctx = _context_key(codon, position)
        if ctx not in group_idx:
            continue
        gi = group_idx[ctx]
        bi = WOBBLE_BASES.index(codon[position]) if codon[position] in WOBBLE_BASES else -1
        if bi >= 0 and gi < 16:
            profile[gi * 4 + bi] += count
    # Normalise each group to fractions
    for gi in range(min(len(groups), 16)):
        total = profile[gi * 4: gi * 4 + 4].sum()
        if total > 0:
            profile[gi * 4: gi * 4 + 4] /= total
    return profile


def _context_key(codon: str, position: int) -> str:
    """Return the two-letter context for a codon excluding the given position."""
    if position == 0:
        return codon[1] + codon[2]
    elif position == 1:
        return codon[0] + codon[2]
    else:
        return codon[0] + codon[1]


def aa_profile(counts: dict) -> np.ndarray:
    """20-dimensional amino acid usage profile.

    For each amino acid, the fraction of total sense codons encoding it.
    """
    total = sum(counts.get(c, 0) for c in SENSE_CODONS)
    if total == 0:
        return np.zeros(len(AA_LIST))
    profile = np.zeros(len(AA_LIST))
    aa_idx = {aa: i for i, aa in enumerate(AA_LIST)}
    for codon in SENSE_CODONS:
        aa = CODON_TO_AA[codon]
        profile[aa_idx[aa]] += counts.get(codon, 0)
    return profile / total


def rscu_profile(counts: dict) -> np.ndarray:
    """61-dimensional RSCU (Relative Synonymous Codon Usage) profile.

    RSCU(c) = observed(c) * n_synonyms(aa) / total(aa)
    Values >1 indicate preference, <1 indicate avoidance.
    Codons belonging to 1-fold degenerate AAs (Met, Trp) always have RSCU=1.
    """
    # Total counts per amino acid
    aa_totals = defaultdict(int)
    for codon in SENSE_CODONS:
        aa_totals[CODON_TO_AA[codon]] += counts.get(codon, 0)

    profile = np.zeros(len(SENSE_CODONS))
    for i, codon in enumerate(SENSE_CODONS):
        aa = CODON_TO_AA[codon]
        n_syn = _AA_NSYN[aa]
        aa_total = aa_totals[aa]
        if aa_total > 0:
            profile[i] = counts.get(codon, 0) * n_syn / aa_total
        else:
            profile[i] = 0.0
    return profile


def full_codon_profile(counts: dict) -> np.ndarray:
    """61-dimensional codon frequency profile (one entry per sense codon)."""
    total = sum(counts.get(c, 0) for c in SENSE_CODONS)
    if total == 0:
        return np.zeros(len(SENSE_CODONS))
    return np.array([counts.get(c, 0) / total for c in SENSE_CODONS])


def _profile_corr(p: np.ndarray, h: np.ndarray) -> tuple:
    """Pearson and Spearman correlation between two profiles, ignoring zero entries.

    Returns (pearson_r, spearman_r).
    """
    mask = (p > 0) | (h > 0)
    if mask.sum() > 2:
        pearson = np.corrcoef(p[mask], h[mask])[0, 1]
        spearman = sp_stats.spearmanr(p[mask], h[mask]).statistic
        return pearson, spearman
    return np.nan, np.nan


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phage_sequences_with_ids(dataset_dir: str, include_test: bool = False):
    """Load phage sequences with IDs from FASTA files.

    Parameters
    ----------
    include_test : bool
        If True, also load test.fna.gz and concatenate after training
        sequences.  Ordering matches the row order in phages_hosts.csv
        when include_test=True (all train rows first, then all test rows).

    Returns list of (seq_id, sequence) tuples.
    """
    records = []
    # Always load training sequences first
    train_path = os.path.join(dataset_dir, 'train.fna.gz')
    with gzip.open(train_path, 'rt') as fh:
        for rec in SeqIO.parse(fh, 'fasta'):
            records.append((rec.id, str(rec.seq)))
    n_train = len(records)

    if include_test:
        test_path = os.path.join(dataset_dir, 'test.fna.gz')
        if os.path.exists(test_path):
            with gzip.open(test_path, 'rt') as fh:
                for rec in SeqIO.parse(fh, 'fasta'):
                    records.append((rec.id, str(rec.seq)))
            logger.info(f"  Loaded {n_train} train + "
                        f"{len(records) - n_train} test sequences")
        else:
            logger.warning(f"  test.fna.gz not found, using train only")

    return records


def load_phage_host_map(dataset_dir: str, rank: str = 'genus',
                        include_test: bool = False):
    """Load phage → taxon mapping from phages_hosts.csv.

    Parameters
    ----------
    rank : str
        'genus' — extract genus from host_genus_lineage (original behaviour).
        'species' — extract species from host_species column.
    include_test : bool
        If True, include test-set phages.  Indices are assigned so that
        all training phages come first (matching train.fna.gz order),
        then test phages (matching test.fna.gz order).

    Returns dict {sequence_index: taxon_name}.
    For multi-host phages, uses the first host.
    """
    csv_df = pd.read_csv(os.path.join(dataset_dir, 'phages_hosts.csv'),
                         delimiter=',', dtype=str)
    is_test = np.array([b == '1' for b in csv_df['in_testset']])

    idx_to_taxon = {}
    train_counter = 0
    test_counter = 0
    n_train = int((~is_test).sum())  # needed for test offset

    for i, row in csv_df.iterrows():
        if is_test[i] and not include_test:
            continue

        if rank == 'species':
            # host_species may contain pipe-separated multi-hosts
            species_raw = str(row.get('host_species', '')).split('|')[0].strip()
            taxon = species_raw if species_raw else None
        else:
            lineage = str(row.get('host_genus_lineage', '')).split('|')[0]
            parts = lineage.split(';')
            taxon = parts[-1].strip() if parts else lineage.strip()

        if not taxon:
            if is_test[i]:
                test_counter += 1
            else:
                train_counter += 1
            continue

        if is_test[i]:
            idx = n_train + test_counter
            test_counter += 1
        else:
            idx = train_counter
            train_counter += 1

        idx_to_taxon[idx] = taxon

    return idx_to_taxon


def load_host_genomes(host_genome_dir: str,
                      taxa_needed: set = None,
                      max_per_taxon: int = 3,
                      rank: str = 'genus'):
    """Load host genomes from manifest.

    Parameters
    ----------
    taxa_needed : set or None
        Only load genomes for these taxa (genus names or species names).
    max_per_taxon : int
        Max genomes per taxon.  For species-level, 1 is typical
        (one representative genome per species).
    rank : str
        'genus' — key genomes by genus (first word of species).
        'species' — key genomes by full species name.

    Returns dict {taxon: [sequence, ...]}
    """
    from phagetransformer.dataset import _read_fasta_raw
    import csv

    manifest = os.path.join(host_genome_dir, 'host_genome_manifest.tsv')

    # First pass: collect entries by taxon
    entries_by_taxon = defaultdict(list)
    with open(manifest) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            species = row['species']
            if rank == 'species':
                taxon = species
            else:
                taxon = species.split()[0]
            if taxa_needed is not None and taxon not in taxa_needed:
                continue
            genome_path = row['genome_path']
            if not os.path.isabs(genome_path):
                genome_path = os.path.join(host_genome_dir, genome_path)
            entries_by_taxon[taxon].append(genome_path)

    # Second pass: load up to max_per_taxon per taxon
    taxon_genomes = {}
    n_total = sum(min(len(v), max_per_taxon)
                  for v in entries_by_taxon.values())
    logger.info(f"  Loading {n_total} genomes for "
                f"{len(entries_by_taxon)} {rank} taxa ...")
    n_loaded = 0
    for taxon, paths in sorted(entries_by_taxon.items()):
        seqs = []
        for path in paths[:max_per_taxon]:
            if not os.path.exists(path):
                continue
            try:
                seqs.append(_read_fasta_raw(path))
                n_loaded += 1
            except Exception:
                continue
        if seqs:
            taxon_genomes[taxon] = seqs
        if n_loaded % 200 == 0 and n_loaded > 0:
            logger.info(f"    {n_loaded}/{n_total} genomes loaded ...")

    logger.info(f"  Loaded {n_loaded} genomes for "
                f"{len(taxon_genomes)} {rank} taxa")
    return taxon_genomes


# ---------------------------------------------------------------------------
# Quick data landscape (--count_only)
# ---------------------------------------------------------------------------

def count_only_summary(dataset_dir: str, host_genome_dir: str):
    """Print a diagnostic summary of species/genus phage counts.

    No gene calling — just reads the CSV and manifest to show:
      - Total phages (train / test)
      - Number of unique species and genera
      - Species-per-genus distribution
      - Phages-per-species distribution at various thresholds
      - How many genera have N+ qualifying species
      - Overlap with available host genomes
    """
    import csv

    # ---- Phage side ----
    csv_df = pd.read_csv(os.path.join(dataset_dir, 'phages_hosts.csv'),
                         delimiter=',', dtype=str)
    is_test = np.array([b == '1' for b in csv_df['in_testset']])

    species_phage_counts = defaultdict(int)
    genus_phage_counts = defaultdict(int)
    species_to_genus = {}

    for i, row in csv_df.iterrows():
        species_raw = str(row.get('host_species', '')).split('|')[0].strip()
        lineage = str(row.get('host_genus_lineage', '')).split('|')[0]
        parts = lineage.split(';')
        genus = parts[-1].strip() if parts else lineage.strip()

        if species_raw:
            species_phage_counts[species_raw] += 1
            species_to_genus[species_raw] = genus
        if genus:
            genus_phage_counts[genus] += 1

    n_train = int((~is_test).sum())
    n_test = int(is_test.sum())
    n_species = len(species_phage_counts)
    n_genera = len(genus_phage_counts)

    print(f"\n{'='*70}")
    print(f"DATA LANDSCAPE SUMMARY")
    print(f"{'='*70}")
    print(f"\nPhages:  {n_train} train + {n_test} test = "
          f"{n_train + n_test} total")
    print(f"Species: {n_species} unique (with host_species annotation)")
    print(f"Genera:  {n_genera} unique")

    # ---- Phages per species distribution ----
    counts = sorted(species_phage_counts.values(), reverse=True)
    print(f"\n--- Phages per species ---")
    print(f"  Max:    {counts[0]}")
    print(f"  Median: {np.median(counts):.0f}")
    print(f"  Mean:   {np.mean(counts):.1f}")
    for threshold in [1, 2, 3, 5, 10, 20, 50]:
        n_pass = sum(1 for c in counts if c >= threshold)
        print(f"  >= {threshold:3d} phages: {n_pass:5d} species "
              f"({100*n_pass/n_species:.1f}%)")

    # ---- Species per genus ----
    genus_species_counts = defaultdict(int)
    for sp, genus in species_to_genus.items():
        genus_species_counts[genus] += 1
    sp_per_genus = sorted(genus_species_counts.values(), reverse=True)

    print(f"\n--- Species per genus ---")
    print(f"  Max:    {sp_per_genus[0]}")
    print(f"  Median: {np.median(sp_per_genus):.0f}")
    for threshold in [2, 3, 5, 10, 20]:
        n_pass = sum(1 for c in sp_per_genus if c >= threshold)
        print(f"  >= {threshold:2d} species: {n_pass:4d} genera")

    # ---- Top N species scenario ----
    print(f"\n--- Top-N species by phage count ---")
    for top_n in [100, 200, 300, 500]:
        if top_n > n_species:
            continue
        top_species = sorted(species_phage_counts,
                             key=species_phage_counts.get,
                             reverse=True)[:top_n]
        top_genera = set(species_to_genus[s] for s in top_species)
        # Genera with 2+ species in the top set
        genus_sp_in_top = defaultdict(int)
        for s in top_species:
            genus_sp_in_top[species_to_genus[s]] += 1
        multi_sp_genera = sum(1 for c in genus_sp_in_top.values() if c >= 2)
        min_phages = species_phage_counts[top_species[-1]]
        print(f"  Top {top_n:4d}: {len(top_genera)} genera, "
              f"{multi_sp_genera} with 2+ species, "
              f"min {min_phages} phages/species")

    # ---- Host genome availability ----
    manifest = os.path.join(host_genome_dir, 'host_genome_manifest.tsv')
    available_species = set()
    available_genera = set()
    if os.path.exists(manifest):
        with open(manifest) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                sp = row['species']
                available_species.add(sp)
                available_genera.add(sp.split()[0])

        sp_overlap = set(species_phage_counts.keys()) & available_species
        gen_overlap = set(genus_phage_counts.keys()) & available_genera

        print(f"\n--- Host genome overlap ---")
        print(f"  Manifest: {len(available_species)} species, "
              f"{len(available_genera)} genera")
        print(f"  Phage species with genomes: {len(sp_overlap)} / "
              f"{n_species}")
        print(f"  Phage genera with genomes:  {len(gen_overlap)} / "
              f"{n_genera}")

    # ---- Genus-first selection grid (the recommended approach) ----
    print(f"\n--- Genus-first selection (recommended) ---")
    print(f"  {'min_phages/sp':>14s}  {'min_sp/genus':>12s}  "
          f"{'genera':>7s}  {'species':>8s}  {'total phages':>13s}  "
          f"{'w/ genome':>10s}")
    for min_phages in [2, 3, 5, 10]:
        # Species passing the phage threshold + having a genome
        qualifying_species = {
            sp for sp, cnt in species_phage_counts.items()
            if cnt >= min_phages and sp in available_species
        }
        # Group by genus
        genus_qual_species = defaultdict(list)
        for sp in qualifying_species:
            genus_qual_species[species_to_genus[sp]].append(sp)

        for min_sp in [2, 3, 5]:
            if min_sp > min_phages:
                continue  # nonsensical combo
            passing_genera = {
                g: sps for g, sps in genus_qual_species.items()
                if len(sps) >= min_sp
            }
            n_gen = len(passing_genera)
            n_sp = sum(len(sps) for sps in passing_genera.values())
            n_phages_total = sum(
                species_phage_counts[sp]
                for sps in passing_genera.values() for sp in sps
            )
            # How many of those species have genomes (should be all
            # since we filtered above, but show for confirmation)
            n_with_genome = sum(
                1 for sps in passing_genera.values()
                for sp in sps if sp in available_species
            )
            print(f"  {min_phages:>14d}  {min_sp:>12d}  "
                  f"{n_gen:>7d}  {n_sp:>8d}  {n_phages_total:>13,d}  "
                  f"{n_with_genome:>10d}")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Per-species analysis (genus is derived by pooling)
# ---------------------------------------------------------------------------

def compute_species_stats(phage_records, idx_to_species, host_genomes,
                          min_codons=500, max_phages_per_species=50):
    """Compute wobble statistics for all species.

    Parameters
    ----------
    idx_to_species : dict
        {sequence_index: species_name}.
    host_genomes : dict
        {species_name: [sequence, ...]}.
    min_codons : int
        Skip species with fewer total codons on either side.
    max_phages_per_species : int
        Cap phage sequences per species (randomly sampled if more).

    Returns (species_df, per_phage_df).
    species_df stores raw codon count dicts (phage_counts_raw,
    host_counts_raw) alongside normalized profiles, enabling
    lossless genus-level pooling later.
    """
    # Group phage sequences by species
    phage_by_species = defaultdict(list)
    for idx, (seq_id, seq) in enumerate(phage_records):
        if idx in idx_to_species:
            phage_by_species[idx_to_species[idx]].append(seq)

    # Find species present in both phage and host data
    shared_species = sorted(set(phage_by_species.keys()) &
                            set(host_genomes.keys()))
    logger.info(f"  {len(shared_species)} species with both phage and "
                f"host data")

    logger.info("  Initialising pyrodigal GeneFinder (meta mode) ...")
    gene_finder = pyrodigal.GeneFinder(meta=True)

    rng = np.random.RandomState(42)
    rows = []
    per_phage_rows = []
    for gi, species in enumerate(shared_species):
        t_sp = time.time()
        logger.info(f"    [{gi+1}/{len(shared_species)}] {species}")

        phage_seqs = phage_by_species[species]
        n_total_phages = len(phage_seqs)
        if len(phage_seqs) > max_phages_per_species:
            indices = rng.choice(len(phage_seqs), max_phages_per_species,
                                 replace=False)
            phage_seqs = [phage_seqs[i] for i in indices]

        phage_codon_lists_per_seq = []
        t_phage = time.time()
        for pi, seq in enumerate(phage_seqs):
            seq_codons = call_genes(seq, gene_finder)
            phage_codon_lists_per_seq.append(seq_codons)
            if (pi + 1) % 10 == 0:
                elapsed = time.time() - t_phage
                logger.info(f"      Phage {pi+1}/{len(phage_seqs)} "
                            f"({len(seq)/1000:.0f} kb) — "
                            f"{elapsed:.1f}s elapsed")
        elapsed_phage = time.time() - t_phage
        phage_codon_lists = [c for per_seq in phage_codon_lists_per_seq
                             for c in per_seq]
        logger.info(f"      Phage done: {len(phage_seqs)}/{n_total_phages} "
                    f"seqs, {len(phage_codon_lists)} genes, "
                    f"{elapsed_phage:.1f}s")

        phage_counts = codon_counts_from_genes(phage_codon_lists)
        if sum(phage_counts.values()) < min_codons:
            logger.info(f"      Skipped: {sum(phage_counts.values())} phage "
                        f"codons < {min_codons}")
            continue

        host_codon_lists = []
        t_host = time.time()
        for hi, seq in enumerate(host_genomes[species]):
            t_one = time.time()
            try:
                host_codon_lists.extend(call_genes(seq, gene_finder))
            except Exception as e:
                logger.debug(f"      Host {hi+1} gene calling failed: {e}")
                continue
            elapsed_one = time.time() - t_one
            logger.info(f"      Host {hi+1}/{len(host_genomes[species])} "
                        f"({len(seq)/1e6:.1f} Mb) — {elapsed_one:.1f}s, "
                        f"{len(host_codon_lists)} genes so far")
        elapsed_host = time.time() - t_host

        host_counts = codon_counts_from_genes(host_codon_lists)
        if sum(host_counts.values()) < min_codons:
            logger.info(f"      Skipped: {sum(host_counts.values())} host "
                        f"codons < {min_codons}")
            continue

        row = _build_stats_row(species, phage_counts, host_counts)
        row['n_phages'] = len(phage_seqs)
        rows.append(row)

        host_prof_wobble = wobble_profile(host_counts)
        for per_seq_codons in phage_codon_lists_per_seq:
            per_seq_counts = codon_counts_from_genes(per_seq_codons)
            if sum(per_seq_counts.values()) < 100:
                continue
            pp = wobble_profile(per_seq_counts)
            per_phage_rows.append({
                'group': species,
                'parent_genus': species.split()[0],
                'phage_profile': pp,
                'phage_profile_pos0': position_profile(per_seq_counts, 0),
                'phage_profile_pos1': position_profile(per_seq_counts, 1),
                'host_profile': host_prof_wobble,
            })

        elapsed_sp = time.time() - t_sp
        spear = row['corr_pos2']
        logger.info(
            f"      Total: {elapsed_sp:.1f}s "
            f"(phage {elapsed_phage:.1f}s, host {elapsed_host:.1f}s) "
            f"({row['phage_codons']} / {row['host_codons']} codons)  "
            f"wobble Spearman={spear:.3f}")

    species_df = pd.DataFrame(rows)
    per_phage_df = pd.DataFrame(per_phage_rows)
    logger.info(f"  {len(species_df)} species with sufficient data, "
                f"{len(per_phage_df)} individual phage profiles")
    return species_df, per_phage_df


def _build_stats_row(group_name, phage_counts, host_counts):
    """Build a stats row dict from raw codon counts.

    Used for both species-level (directly) and genus-level (from
    pooled counts).  Stores raw counts for later re-pooling.
    Only emits fields consumed by make_figure_combined, the shuffled
    / decomposition null computations, and the per-species log line.
    """
    phage_pos_profiles = {}
    host_pos_profiles = {}
    spearman_by_pos = {}
    for pos in range(3):
        pp = position_profile(phage_counts, pos)
        hp = position_profile(host_counts, pos)
        phage_pos_profiles[pos] = pp
        host_pos_profiles[pos] = hp
        _, spear = _profile_corr(pp, hp)
        spearman_by_pos[pos] = spear

    phage_aa = aa_profile(phage_counts)
    host_aa = aa_profile(host_counts)
    phage_rscu = rscu_profile(phage_counts)
    host_rscu = rscu_profile(host_counts)
    phage_full = full_codon_profile(phage_counts)
    host_full = full_codon_profile(host_counts)
    _, corr_aa = _profile_corr(phage_aa, host_aa)
    _, corr_rscu = _profile_corr(phage_rscu, host_rscu)
    _, corr_full = _profile_corr(phage_full, host_full)

    # Wobble (pos 2) profiles are stored under the unadorned keys
    # 'phage_profile' / 'host_profile' for back-compat with the
    # per-phage code path.
    row = {
        'group': group_name,
        'parent_genus': group_name.split()[0],
        'phage_counts_raw': phage_counts,
        'host_counts_raw': host_counts,
        'phage_profile': phage_pos_profiles[2],
        'host_profile': host_pos_profiles[2],
        'phage_aa_profile': phage_aa,
        'host_aa_profile': host_aa,
        'phage_rscu_profile': phage_rscu,
        'host_rscu_profile': host_rscu,
        'phage_full_profile': phage_full,
        'host_full_profile': host_full,
        'corr_aa': corr_aa,
        'corr_rscu': corr_rscu,
        'corr_full': corr_full,
        'phage_codons': sum(phage_counts.values()),
        'host_codons': sum(host_counts.values()),
    }
    for pos in range(3):
        row[f'phage_profile_pos{pos}'] = phage_pos_profiles[pos]
        row[f'host_profile_pos{pos}'] = host_pos_profiles[pos]
        row[f'corr_pos{pos}'] = spearman_by_pos[pos]
    return row


def pool_to_genus(species_df):
    """Derive genus-level stats by summing raw codon counts per genus.

    Returns genus_df with the same schema as species_df (minus
    species-specific fields), properly weighted by codon count.
    """
    if 'phage_counts_raw' not in species_df.columns:
        raise ValueError("species_df missing phage_counts_raw — "
                         "regenerate cache with current code")

    genus_rows = []
    for genus, grp in species_df.groupby('parent_genus'):
        phage_pooled = defaultdict(int)
        host_pooled = defaultdict(int)
        n_species = 0
        total_phages = 0
        for _, sp_row in grp.iterrows():
            for codon, cnt in sp_row['phage_counts_raw'].items():
                phage_pooled[codon] += cnt
            for codon, cnt in sp_row['host_counts_raw'].items():
                host_pooled[codon] += cnt
            n_species += 1
            total_phages += sp_row.get('n_phages', 0)

        row = _build_stats_row(genus, dict(phage_pooled), dict(host_pooled))
        row['n_species'] = n_species
        row['n_phages'] = total_phages
        genus_rows.append(row)

    genus_df = pd.DataFrame(genus_rows)
    logger.info(f"  Pooled {len(species_df)} species into "
                f"{len(genus_df)} genera")
    return genus_df


def compute_shuffled_correlations(df, position=2, n_shuffles=1000):
    """Compute profile Spearman correlations with shuffled host assignments.

    Parameters
    ----------
    position : int
        Codon position (0, 1, or 2). Default 2 = wobble.
    """
    if position == 2:
        profiles_phage = np.stack(df['phage_profile'].values)
        profiles_host = np.stack(df['host_profile'].values)
    else:
        profiles_phage = np.stack(df[f'phage_profile_pos{position}'].values)
        profiles_host = np.stack(df[f'host_profile_pos{position}'].values)
    n = len(df)
    rng = np.random.RandomState(42 + position)

    shuffled_corrs = []
    for _ in range(n_shuffles):
        perm = rng.permutation(n)
        for i in range(n):
            j = perm[i]
            p = profiles_phage[i]
            h = profiles_host[j]
            mask = (p > 0) | (h > 0)
            if mask.sum() > 2:
                shuffled_corrs.append(
                    sp_stats.spearmanr(p[mask], h[mask]).statistic)
    return np.array(shuffled_corrs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_figure_combined(genus_df, per_phage_df, shuffled_by_pos,
                         shuffled_decomp, out_path, dpi=200):
    """Single 3x3 figure summarising phage-host codon adaptation (genus level).

    Row 1: biological decomposition of the signal — amino acid usage,
        full codon profile, RSCU — each matched vs shuffled null.
    Row 2: position decomposition, genus-averaged — per-genus profile
        correlation at positions 1, 2, 3 vs shuffled null.
    Row 3: position decomposition, per-phage — each phage's profile
        correlated against its genus-pooled host (matched) and a
        randomly chosen genus (null), separately at each position.

    Panels are colour-coded by column (panels in column 1 use
    COLORS['primary'], column 2 secondary, column 3 tertiary) so the
    reader can track a single position/decomposition down the figure.
    Every panel's median and n are reported in an in-axes overlay in
    the upper-left corner — titles stay minimal.
    """
    setup_style()

    fig = plt.figure(figsize=(FIG_WIDTH, 3 * FIG_HEIGHT_ROW * 0.72))
    gs = gridspec.GridSpec(3, 1, figure=fig,
                           hspace=0.55,
                           left=0.06, right=0.96, top=0.95, bottom=0.06)

    all_axes = []
    # One colour per column, reused across all three rows. Columns
    # correspond to: AA / Pos 1 / per-phage Pos 1 (col 1), Full codon /
    # Pos 2 / per-phage Pos 2 (col 2), RSCU / Pos 3 / per-phage Pos 3
    # (col 3).
    col_colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    def _overlay(ax, lines):
        """Draw a multi-line stat overlay in the top-left of the panel."""
        ax.text(0.03, 0.96, '\n'.join(lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=FONT_SIZES['legend'],
                color=COLORS['text'])

    # ---- Row 1: Biological decomposition (AA -> Full -> RSCU) ----
    gs_r1 = gs[0].subgridspec(1, 3, wspace=0.25)
    decomp_specs = [
        ('corr_aa',   'Amino acid usage',   'aa'),
        ('corr_full', 'Full codon profile', 'full'),
        ('corr_rscu', 'RSCU',               'rscu'),
    ]
    for di, (col, title, key) in enumerate(decomp_specs):
        ax = fig.add_subplot(gs_r1[0, di])
        real = genus_df[col].dropna().values
        ax.hist(real, bins=30, density=True, alpha=0.7,
                color=col_colors[di], label='Matched',
                edgecolor='white', linewidth=0.5, zorder=2)
        if shuffled_decomp and key in shuffled_decomp:
            ax.hist(shuffled_decomp[key], bins=50, density=True, alpha=0.35,
                    color=COLORS['grid'], label='Shuffled',
                    edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Spearman correlation')
        if di == 0:
            ax.set_ylabel('Density')
        ax.legend(frameon=False, loc='center left')
        ax.set_title(title)
        _overlay(ax, [f'median = {np.nanmedian(real):.3f}',
                      f'n = {len(real)}'])
        all_axes.append(ax)

    # ---- Row 2: Position decomposition, genus-averaged ----
    gs_r2 = gs[1].subgridspec(1, 3, wspace=0.25)
    pos_labels = ['Position 1', 'Position 2', 'Position 3']
    for pos in range(3):
        ax = fig.add_subplot(gs_r2[0, pos])
        real = genus_df[f'corr_pos{pos}'].dropna().values
        ax.hist(real, bins=30, density=True, alpha=0.7,
                color=col_colors[pos], label='Matched',
                edgecolor='white', linewidth=0.5, zorder=2)
        ax.hist(shuffled_by_pos[pos], bins=50, density=True, alpha=0.35,
                color=COLORS['grid'], label='Shuffled',
                edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Profile correlation (Spearman)')
        if pos == 0:
            ax.set_ylabel('Density')
        ax.legend(frameon=False, loc='center left')
        ax.set_title(pos_labels[pos])
        _overlay(ax, [f'median = {np.nanmedian(real):.3f}',
                      f'n = {len(real)}'])
        all_axes.append(ax)

    # ---- Row 3: Position decomposition, per-phage ----
    gs_r3 = gs[2].subgridspec(1, 3, wspace=0.25)

    needed_cols = ['phage_profile_pos0', 'phage_profile_pos1', 'phage_profile']
    have_per_phage = (per_phage_df is not None and len(per_phage_df) > 0
                      and all(c in per_phage_df.columns for c in needed_cols))

    if have_per_phage:
        logger.info("  Computing per-phage per-position correlations "
                    "(genus-pooled hosts) ...")

        # Map genus -> host profile at each codon position
        host_by_genus = {}
        for pos in range(3):
            hcol = 'host_profile' if pos == 2 else f'host_profile_pos{pos}'
            if hcol not in genus_df.columns:
                logger.warning(f"  genus_df missing {hcol}")
                host_by_genus[pos] = {}
                continue
            host_by_genus[pos] = dict(zip(genus_df['group'],
                                          genus_df[hcol]))

        # Pre-filter per-phage rows to those whose parent_genus is pooled
        pp_df = per_phage_df[
            per_phage_df['parent_genus'].isin(genus_df['group'])
        ].reset_index(drop=True)
        n_dropped = len(per_phage_df) - len(pp_df)
        if n_dropped > 0:
            logger.info(f"    Dropped {n_dropped} per-phage rows whose "
                        f"parent genus is absent from genus_df")

        available_genera = list(genus_df['group'])
        rng = np.random.RandomState(42)

        for pos in range(3):
            ax = fig.add_subplot(gs_r3[0, pos])
            pcol = 'phage_profile' if pos == 2 else f'phage_profile_pos{pos}'

            matched_corrs = []
            null_corrs = []
            host_map = host_by_genus[pos]
            for _, row in pp_df.iterrows():
                phage_p = row[pcol]
                genus = row['parent_genus']
                if genus in host_map:
                    _, s = _profile_corr(phage_p, host_map[genus])
                    if np.isfinite(s):
                        matched_corrs.append(s)
                # Random-genus null (sampled independently per phage)
                rand_g = available_genera[rng.randint(len(available_genera))]
                if rand_g in host_map:
                    _, s = _profile_corr(phage_p, host_map[rand_g])
                    if np.isfinite(s):
                        null_corrs.append(s)
            matched_corrs = np.array(matched_corrs)
            null_corrs = np.array(null_corrs)

            bins_pp = np.linspace(-1, 1, 60)
            ax.hist(matched_corrs, bins=bins_pp, density=True, alpha=0.7,
                    color=col_colors[pos], label='Matched host',
                    edgecolor='white', linewidth=0.5, zorder=2)
            ax.hist(null_corrs, bins=bins_pp, density=True, alpha=0.35,
                    color=COLORS['grid'], label='Random host',
                    edgecolor='white', linewidth=0.5, zorder=3)
            ax.set_xlim(-1, 1)
            ax.set_xlabel('Profile correlation (Spearman)')
            if pos == 0:
                ax.set_ylabel('Density')
            ax.legend(frameon=False, loc='center left')
            ax.set_title(f'{pos_labels[pos]}, per-phage')
            _overlay(ax, [
                f'matched median = {np.nanmedian(matched_corrs):.3f}',
                f'null median = {np.nanmedian(null_corrs):.3f}',
                f'n = {len(matched_corrs)}',
            ])
            all_axes.append(ax)
    else:
        for i in range(3):
            ax = fig.add_subplot(gs_r3[0, i])
            ax.text(0.5, 0.5,
                    'No per-phage per-position data\n(regenerate cache)',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_axis_off()
            all_axes.append(ax)

    letters = 'ABCDEFGHI'
    for ax, letter in zip(all_axes, letters):
        ax.text(-0.12, 1.08, letter, transform=ax.transAxes,
                fontsize=FONT_SIZES['panel_letter'], fontweight='bold',
                color=COLORS['text'], va='top')

    _suptitle(fig, 'Phage\u2013host codon usage correlation (genus level)',
              y=1.0)
    _save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phage-host codon usage analysis at the genus level. '
                    'Species is the atomic unit; genus is derived by pooling.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset directory (train.fna.gz, test.fna.gz, '
                             'phages_hosts.csv)')
    parser.add_argument('--host_genome_dir', type=str, required=True,
                        help='Host genome directory '
                             '(host_genome_manifest.tsv)')
    parser.add_argument('--output', '-o', type=str,
                        default='codon_stats.png',
                        help='Output filename base')
    parser.add_argument('--count_only', action='store_true',
                        help='Print data landscape summary and exit '
                             '(no gene calling)')
    parser.add_argument('--min_phages_per_species', type=int, default=3,
                        help='Min phages a species needs to qualify')
    parser.add_argument('--min_species_per_genus', type=int, default=3,
                        help='Min qualifying species a genus needs')
    parser.add_argument('--max_phages_per_species', type=int, default=50,
                        help='Cap phage sequences per species')
    parser.add_argument('--top_genera', type=int, default=None,
                        help='After genus-first filtering, keep only the '
                             'N genera with most total phages (None = all)')
    parser.add_argument('--min_codons', type=int, default=500,
                        help='Min total codons per side per species '
                             '(applied after gene calling)')
    parser.add_argument('--n_shuffles', type=int, default=1000,
                        help='Number of shuffled pairings for null')
    parser.add_argument('--cache', type=str, default=None,
                        help='Path to cache species stats as pickle')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--presentation', action='store_true',
                        help='Increase font sizes for presentations')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # ---- Count-only mode ----
    if args.count_only:
        count_only_summary(args.dataset_dir, args.host_genome_dir)
        return

    if args.presentation:
        enable_presentation_mode()

    # ==================================================================
    # Step 1: Load or compute species-level stats
    # ==================================================================
    if args.cache and os.path.exists(args.cache):
        logger.info(f"Loading cached stats from {args.cache}")
        import pickle
        with open(args.cache, 'rb') as f:
            cached = pickle.load(f)
        species_df = cached['df']
        per_phage_df = cached.get('per_phage_df', pd.DataFrame())
        # Migrate legacy 'genus' column to 'group'
        if 'genus' in species_df.columns and 'group' not in species_df.columns:
            species_df = species_df.rename(columns={'genus': 'group'})
        if ('genus' in per_phage_df.columns
                and 'group' not in per_phage_df.columns):
            per_phage_df = per_phage_df.rename(columns={'genus': 'group'})
        logger.info(f"  {len(species_df)} species, "
                    f"{len(per_phage_df)} per-phage profiles")
    else:
        # ---- Load all phages (train + test) ----
        logger.info("Loading phage sequences (train+test) ...")
        phage_records = load_phage_sequences_with_ids(
            args.dataset_dir, include_test=True)
        logger.info(f"  {len(phage_records)} phage sequences")

        # ---- Map phages to species ----
        logger.info("Loading phage-host mapping (species) ...")
        idx_to_species = load_phage_host_map(
            args.dataset_dir, rank='species', include_test=True)
        logger.info(f"  {len(idx_to_species)} phages mapped to "
                    f"{len(set(idx_to_species.values()))} species")

        # ---- Genus-first selection ----
        species_counts = defaultdict(int)
        species_to_genus = {}
        for sp in idx_to_species.values():
            species_counts[sp] += 1
            species_to_genus[sp] = sp.split()[0]

        # Filter species by min phages
        qualifying_species = {
            sp for sp, cnt in species_counts.items()
            if cnt >= args.min_phages_per_species
        }
        logger.info(f"  {len(qualifying_species)} species with "
                    f">= {args.min_phages_per_species} phages")

        # Group by genus, filter genera by min species
        genus_qual = defaultdict(set)
        for sp in qualifying_species:
            genus_qual[species_to_genus[sp]].add(sp)

        passing_genera = {
            g: sps for g, sps in genus_qual.items()
            if len(sps) >= args.min_species_per_genus
        }

        # Optionally limit to top N genera by total phage count
        if args.top_genera and args.top_genera < len(passing_genera):
            genus_total_phages = {
                g: sum(species_counts[sp] for sp in sps)
                for g, sps in passing_genera.items()
            }
            top_genera = sorted(genus_total_phages,
                                key=genus_total_phages.get,
                                reverse=True)[:args.top_genera]
            passing_genera = {g: passing_genera[g] for g in top_genera}
            logger.info(f"  Limited to top {args.top_genera} genera "
                        f"(min {genus_total_phages[top_genera[-1]]} "
                        f"phages/genus)")

        selected_species = set()
        for sps in passing_genera.values():
            selected_species |= sps

        logger.info(f"  {len(passing_genera)} genera with "
                    f">= {args.min_species_per_genus} qualifying species "
                    f"-> {len(selected_species)} species selected")

        # Filter idx_to_species to only selected species
        idx_to_species = {idx: sp for idx, sp in idx_to_species.items()
                          if sp in selected_species}

        # ---- Load host genomes (1 per species) ----
        logger.info("Loading host genomes ...")
        host_genomes = load_host_genomes(
            args.host_genome_dir,
            taxa_needed=selected_species,
            max_per_taxon=1,
            rank='species')
        logger.info(f"  {len(host_genomes)} species with genomes")

        # ---- Compute species-level stats ----
        logger.info("Computing per-species wobble statistics ...")
        species_df, per_phage_df = compute_species_stats(
            phage_records, idx_to_species, host_genomes,
            min_codons=args.min_codons,
            max_phages_per_species=args.max_phages_per_species)

        if args.cache:
            import pickle
            with open(args.cache, 'wb') as f:
                pickle.dump({'df': species_df,
                             'per_phage_df': per_phage_df}, f)
            logger.info(f"  Cached stats to {args.cache}")

    # ==================================================================
    # Step 2: Derive genus-level stats by pooling
    # ==================================================================
    logger.info("Pooling species into genus-level profiles ...")
    genus_df = pool_to_genus(species_df)

    # ==================================================================
    # Step 3: Compute shuffled nulls at the genus level (per position)
    # ==================================================================
    logger.info("Computing shuffled nulls (genus) ...")
    gen_shuffled = {}
    for pos in range(3):
        gen_shuffled[pos] = compute_shuffled_correlations(
            genus_df, position=pos, n_shuffles=args.n_shuffles)

    # Signal summary
    pos_names = ['Position 1', 'Position 2', 'Position 3 (wobble)']
    logger.info("  genus signal summary:")
    for pos in range(3):
        matched = genus_df[f'corr_pos{pos}'].dropna().values
        null = gen_shuffled[pos]
        m_mean, n_mean = np.nanmean(matched), np.nanmean(null)
        m_std, n_std = np.nanstd(matched), np.nanstd(null)
        pooled_std = np.sqrt((m_std ** 2 + n_std ** 2) / 2)
        d = (m_mean - n_mean) / pooled_std if pooled_std > 0 else np.inf
        U, _ = sp_stats.mannwhitneyu(matched, null, alternative='greater')
        auroc = U / (len(matched) * len(null))
        logger.info(f"    {pos_names[pos]:22s}  "
                    f"med matched={np.nanmedian(matched):.3f}  "
                    f"null={np.nanmedian(null):.3f}  "
                    f"d={d:.2f}  AUROC={auroc:.4f}")

    # ==================================================================
    # Step 4: Decomposition nulls (AA / RSCU / full codon, genus level)
    # ==================================================================
    logger.info("Computing decomposition nulls (genus) ...")
    gen_decomp = {}
    n = len(genus_df)
    rng = np.random.RandomState(99)
    for key, pc, hc in [
        ('aa',   'phage_aa_profile',   'host_aa_profile'),
        ('rscu', 'phage_rscu_profile', 'host_rscu_profile'),
        ('full', 'phage_full_profile', 'host_full_profile'),
    ]:
        if pc not in genus_df.columns:
            continue
        pp = np.stack(genus_df[pc].values)
        hp = np.stack(genus_df[hc].values)
        corrs = []
        for _ in range(args.n_shuffles):
            perm = rng.permutation(n)
            for i in range(n):
                _, s = _profile_corr(pp[i], hp[perm[i]])
                if np.isfinite(s):
                    corrs.append(s)
        gen_decomp[key] = np.array(corrs)

    # ==================================================================
    # Step 5: Generate the combined figure (genus level)
    # ==================================================================
    logger.info("Generating combined figure ...")
    make_figure_combined(
        genus_df, per_phage_df, gen_shuffled, gen_decomp,
        args.output, dpi=args.dpi)

    logger.info("Done.")


if __name__ == '__main__':
    main()
