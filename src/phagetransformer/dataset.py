import csv
import gzip
import logging
import os
import random
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def read_fasta_gz(fasta_path: str) -> List[str]:
    seqs = []
    with gzip.open(fasta_path, 'rt') as fh:
        for rec in SeqIO.parse(fh, 'fasta'):
            seqs.append(str(rec.seq))
    return seqs


def load_phage_host_merged(path_to_dataset):
    train_seqs = read_fasta_gz(os.path.join(path_to_dataset, "train.fna.gz"))
    test_seqs = read_fasta_gz(os.path.join(path_to_dataset, "test.fna.gz"))
    df = pd.read_csv(os.path.join(path_to_dataset, "phages_hosts.csv"),
                     delimiter=',', dtype=str)
    test_idx = np.array([b == "1" for b in df["in_testset"]])
    train_idx = ~test_idx
    genus_index = {x: i for i, x in enumerate(
        sorted({h for hs in df["host_genus_lineage"] for h in hs.split("|")}))}
    labels = np.zeros((len(df), len(genus_index)), dtype=np.float32)
    for i, hosts in enumerate(df["host_genus_lineage"]):
        for h in hosts.split("|"):
            labels[i, genus_index[h]] = 1
    hosts = np.array([g[0] for g in sorted(genus_index.items(), key=lambda x: x[1])])
    return train_seqs, labels[train_idx], test_seqs, labels[test_idx], hosts


def load_phage_host_test(path_to_dataset, hosts):
    test_seqs = read_fasta_gz(os.path.join(path_to_dataset, "combined.fna.gz"))
    df = pd.read_csv(os.path.join(path_to_dataset, "combined_lineage.csv"),
                     delimiter=',', dtype=str)
    genus_index = {x: i for i, x in enumerate(hosts)}
    labels = np.zeros((len(df), len(genus_index)), dtype=np.float32)
    for i, hs in enumerate(df["host_genus_lineage"]):
        for h in hs.split("|"):
            h_trunc = ';'.join(h.split(';')[1:])
            if h_trunc in genus_index:
                labels[i, genus_index[h_trunc]] = 1
    keep = df["dataset"] != "refseq"
    return [s for i, s in enumerate(test_seqs) if keep[i]], labels[keep]


def _read_fasta_raw(path: str) -> str:
    """Read a (possibly gzipped) FASTA and return concatenated sequence.

    Uses raw line parsing instead of BioPython for ~5-10x faster loading
    on gzipped files.
    """
    opener = gzip.open if path.endswith('.gz') else open
    parts = []
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                continue
            parts.append(line.strip().upper())
    return ''.join(parts)


class BacterialGenomeStore:
    """Holds bacterial genomes in memory for efficient chunk sampling.

    Loads all genomes once (in parallel) from the host genome manifest,
    splits species into train/val partitions, and provides fast
    chunk resampling for each training epoch.

    Parameters
    ----------
    host_genome_dir : str
        Directory containing ``host_genome_manifest.tsv``.
    val_frac : float
        Fraction of each genome reserved for validation (random contiguous region).
    num_workers : int
        Threads for parallel FASTA loading.
    seed : int
        Random seed for reproducible val region placement.
    """

    def __init__(self, host_genome_dir: str,
                 val_frac: float = 0.2,
                 num_workers: int = 8,
                 seed: int = 42,
                 one_per_genus: bool = False):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        manifest = os.path.join(host_genome_dir, 'host_genome_manifest.tsv')
        if not os.path.exists(manifest):
            raise FileNotFoundError(
                f"No host_genome_manifest.tsv in {host_genome_dir}. "
                f"Run the download script first.")

        # Parse manifest — genome_path is resolved relative to host_genome_dir
        entries = []  # [(species, path), ...]
        with open(manifest) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                genome_path = row['genome_path']
                if not os.path.isabs(genome_path):
                    genome_path = os.path.join(host_genome_dir, genome_path)
                entries.append((row['species'], genome_path))

        # Optionally keep only one species per genus
        if one_per_genus:
            seen_genera = set()
            filtered = []
            for species, path in entries:
                genus = species.split()[0]
                if genus not in seen_genera:
                    seen_genera.add(genus)
                    filtered.append((species, path))
            logger.info(f"  --one_genome_per_genus: {len(entries)} -> "
                        f"{len(filtered)} entries ({len(seen_genera)} genera)")
            entries = filtered

        # Load genomes in parallel
        logger.info(f"  Loading {len(entries)} bacterial genomes "
                    f"({num_workers} threads) …")
        t0 = time.time()
        genomes = {}   # species -> sequence
        n_skipped = 0

        def _load_one(species_path):
            species, path = species_path
            if not os.path.exists(path):
                return species, None, f"not found: {path}"
            try:
                seq = _read_fasta_raw(path)
                return species, seq, None
            except Exception as e:
                return species, None, str(e)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_load_one, e) for e in entries]
            for fut in as_completed(futures):
                species, seq, err = fut.result()
                if err:
                    logger.warning(f"    {species}: {err}")
                    n_skipped += 1
                else:
                    genomes[species] = seq

        elapsed = time.time() - t0
        logger.info(f"  Loaded {len(genomes)} genomes in {elapsed:.1f}s "
                    f"({n_skipped} skipped)")

        self.genomes = genomes

        # Per-genome split: cut out a random contiguous val_frac region,
        # leaving two flanking train regions.
        rng = np.random.RandomState(seed)
        self.species_list = sorted(genomes.keys())
        self.splits = {}
        for sp in self.species_list:
            seq_len = len(genomes[sp])
            val_len = int(seq_len * val_frac)
            max_start = seq_len - val_len
            val_start = rng.randint(0, max(max_start, 1))
            val_end = val_start + val_len
            self.splits[sp] = {
                'val': (val_start, val_end),
                'train': [(0, val_start), (val_end, seq_len)],
            }
        min_val = min(s['val'][1] - s['val'][0] for s in self.splits.values())
        logger.info(f"  Per-genome region split: ~{1.0-val_frac:.0%} train / "
                    f"~{val_frac:.0%} val (random cutout, seed={seed}, "
                    f"smallest val region: {min_val:,} nt)")

    def write_species_log(self, path: str):
        """Write a TSV listing all loaded species with genome lengths and split coords."""
        with open(path, 'w') as f:
            f.write('species\tgenus\tgenome_len\t'
                    'train1_start\ttrain1_end\tval_start\tval_end\t'
                    'train2_start\ttrain2_end\n')
            for sp in self.species_list:
                genus = sp.split()[0]
                seq_len = len(self.genomes[sp])
                (t1s, t1e), (t2s, t2e) = self.splits[sp]['train']
                vs, ve = self.splits[sp]['val']
                f.write(f'{sp}\t{genus}\t{seq_len}\t'
                        f'{t1s}\t{t1e}\t{vs}\t{ve}\t{t2s}\t{t2e}\n')
        logger.info(f"  Bacterial species log: {path} ({len(self.species_list)} species)")

    def load_splits(self, path: str):
        """Override per-genome train/val splits from a previously saved TSV.

        This ensures evaluation uses the exact same splits as training.
        Species in the TSV that are not loaded are silently skipped;
        loaded species not in the TSV keep their random splits.
        """
        import pandas as _pd
        df = _pd.read_csv(path, sep='\t')
        n_applied = 0
        for _, row in df.iterrows():
            sp = row['species']
            if sp not in self.genomes:
                continue
            self.splits[sp] = {
                'val': (int(row['val_start']), int(row['val_end'])),
                'train': [(int(row['train1_start']), int(row['train1_end'])),
                          (int(row['train2_start']), int(row['train2_end']))],
            }
            n_applied += 1
        logger.info(f"  Loaded splits from {path}: {n_applied} species "
                    f"(of {len(df)} in file, {len(self.species_list)} loaded)")

    def sample_subseq(self, split: str, subseq_len: int) -> tuple:
        """Sample a random subsequence from the given split ('train' or 'val').

        For 'val', samples from the single contiguous val region.
        For 'train', picks one of the two flanking regions (weighted by
        length), then samples a start position within it.  If the chosen
        region is shorter than subseq_len, returns the full region
        (trimmed to a multiple of 3).

        Returns (subsequence, genus) where genus is the first word of
        the species name (e.g. 'Bacillus' from 'Bacillus subtilis').
        """
        sp = self.species_list[np.random.randint(len(self.species_list))]
        seq = self.genomes[sp]
        genus = sp.split()[0]

        if split == 'val':
            region_start, region_end = self.splits[sp]['val']
        else:
            regions = self.splits[sp]['train']
            lengths = [max(0, e - s) for s, e in regions]
            total = sum(lengths)
            if total == 0:
                # Fallback: use whole genome
                region_start, region_end = 0, len(seq)
            else:
                # Pick region weighted by length
                if np.random.random() < lengths[0] / total:
                    region_start, region_end = regions[0]
                else:
                    region_start, region_end = regions[1]

        region_len = region_end - region_start
        if region_len <= subseq_len:
            chunk = seq[region_start:region_end]
        else:
            start = region_start + np.random.randint(0, region_len - subseq_len)
            chunk = seq[start:start + subseq_len]
        return chunk, genus

    def sample_train_subseq(self, subseq_len: int) -> tuple:
        return self.sample_subseq('train', subseq_len)

    def sample_val_subseq(self, subseq_len: int) -> tuple:
        return self.sample_subseq('val', subseq_len)


def _compute_patches_per_seq(labels, sequences, min_patches, max_patches, patch_len):
    """Per-sequence patch count from inverse label rarity.

    Rare-class sequences get ``max_patches`` random patches per epoch,
    common-class sequences get ``min_patches``.
    """
    class_counts = labels.sum(axis=0).clip(min=1)
    max_count = class_counts.max()
    counts = []
    for lab, seq in zip(labels, sequences):
        active = np.where(lab > 0)[0]
        if len(active) == 0:
            counts.append(min_patches)
            continue
        # ratio → 0 for rare, 1 for common
        ratio = class_counts[active].min() / max_count
        n = int(len(seq) / patch_len) + 1
        f = max_patches - ratio * (max_patches - min_patches)
        counts.append(int(n if n <= 2 else f * n))
    return counts


class RandomPatchDataset(Dataset):
    """Encoder-phase training dataset: randomly samples patch positions each access.

    Each item is one (6, codon_L) tokenised patch + its parent sequence label.
    Rare-class sequences contribute more patches per epoch (curriculum).
    Patch start positions are uniformly random — different every epoch.
    """

    def __init__(self, sequences, labels, tokenizer, patch_nt_len=4096,
                 min_patches_per_seq=1, max_patches_per_seq=5, scramble_rate=0.1):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.patch_nt_len = patch_nt_len
        self.scramble_rate = scramble_rate
        self.zero_label = torch.from_numpy(np.zeros(len(labels[0]), dtype=np.float32))
        
        # Build flat index: repeat seq_idx based on its patch count
        pps = _compute_patches_per_seq(labels, sequences, min_patches_per_seq,
                                       max_patches_per_seq, patch_nt_len)
        self.index = []
        for si, n in enumerate(pps):
            self.index.extend([si] * n)
        self._pps = pps

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si = self.index[idx]
        seq = self.sequences[si]
        slen = len(seq)
        plen = self.patch_nt_len

        # Random start position (uniform)
        if slen <= plen:
            start = 0
        else:
            start = torch.randint(0, slen - plen + 1, (1,)).item()

        patch = seq[start:start + plen]
        
        if np.random.rand() <= self.scramble_rate:
            chars = list(patch)
            random.shuffle(chars)
            patch = ''.join(chars)
            label = self.zero_label 
        else:
            label = torch.from_numpy(self.labels[si])
            
        tokens = self.tokenizer.tokenize(patch)    
        return tokens, label


class EvalPatchDataset(Dataset):
    """Encoder-phase eval dataset: deterministic stride-based tiling."""

    def __init__(self, sequences, labels, tokenizer, patch_nt_len=4096,
                 stride=None):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.patch_nt_len = patch_nt_len
        stride = stride or patch_nt_len // 2

        self.index = []
        for si, seq in enumerate(sequences):
            slen = len(seq)
            added = False
            for start in range(0, max(1, slen - patch_nt_len + 1), stride):
                self.index.append((si, start))
                added = True
            if slen > patch_nt_len:
                last = slen - patch_nt_len
                if not self.index or self.index[-1] != (si, last):
                    self.index.append((si, last))
            elif not added:
                self.index.append((si, 0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, start = self.index[idx]
        patch = self.sequences[si][start:start + self.patch_nt_len]
        tokens = self.tokenizer.tokenize(patch)
        label = torch.from_numpy(self.labels[si])
        return tokens, label


def patch_collate_fn(batch):
    tokens_list, labels = zip(*batch)
    max_cl = max(t.size(1) for t in tokens_list)
    B = len(batch)
    out = torch.zeros(B, 6, max_cl, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        out[i, :, :t.size(1)] = t
    return out, torch.stack(labels)


class BacterialPatchDataset(Dataset):
    """Bacterial patch dataset for encoder-phase training/evaluation.

    Each item is a single patch sampled from a random bacterial genome,
    labeled with a multi-hot vector: the ``bacterial_fragment`` class
    (last column) is always set, and the corresponding genus column is
    set when the genus appears in the host list.  Scrambled samples
    receive a zero label.

    Used alongside :class:`RandomPatchDataset` (via ``ConcatDataset``)
    so the encoder learns to distinguish phage from bacterial DNA.
    """

    def __init__(self, genome_store, tokenizer, num_classes: int,
                 genus_to_idx: dict,
                 n_samples: int, patch_nt_len: int = 3074,
                 is_train: bool = True, scramble_rate: float = 0.0):
        self.genome_store = genome_store
        self.tokenizer = tokenizer
        self.patch_nt_len = patch_nt_len
        self.split = 'train' if is_train else 'val'
        self.n_samples = n_samples
        self.scramble_rate = scramble_rate
        self.num_classes = num_classes
        self.genus_to_idx = genus_to_idx
        self.bact_idx = num_classes - 1  # bacterial_fragment is last class

    def __len__(self):
        return self.n_samples

    def _make_label(self, genus: str) -> np.ndarray:
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[self.bact_idx] = 1.0
        idx = self.genus_to_idx.get(genus)
        if idx is not None:
            label[idx] = 1.0
        return label

    def __getitem__(self, idx):
        seq, genus = self.genome_store.sample_subseq(
            self.split, self.patch_nt_len)

        if self.scramble_rate > 0 and np.random.rand() < self.scramble_rate:
            chars = list(seq)
            random.shuffle(chars)
            seq = ''.join(chars)
            label = np.zeros(self.num_classes, dtype=np.float32)
        else:
            label = self._make_label(genus)

        tokens = self.tokenizer.tokenize(seq)
        return tokens, torch.from_numpy(label)


class PatchSequenceDataset(Dataset):
    """Sequence-level dataset for the aggregator phase.

    Training:  Randomly truncates a fraction of the sequence (0 to
               ``seq_drop_rate``) at a random position, then tiles with
               stride = patch_nt_len / coverage + random offset.
               After tiling, randomly drops patches (0 to ``patch_drop_rate``
               fraction, keeping at least 1).
    Eval:      Deterministic stride-based tiling on the full sequence.
    """

    def __init__(self, sequences, labels, tokenizer, patch_nt_len=4096,
                 max_patches=512, is_train=True, eval_stride=None,
                 coverage=2.0, seq_drop_rate=0.0, patch_drop_rate=0.0,
                 scramble_rate=0.1):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.patch_nt_len = patch_nt_len
        self.max_patches = max_patches
        self.is_train = is_train
        self.coverage = coverage
        self.seq_drop_rate = seq_drop_rate
        self.patch_drop_rate = patch_drop_rate
        self.scramble_rate = scramble_rate
        self.zero_label = np.zeros(len(labels[0]), dtype=np.float32)
        
        # Compute strides
        self.train_stride = max(1, int(patch_nt_len / coverage))
        self.eval_stride = eval_stride or patch_nt_len // 2

    def __len__(self):
        return len(self.sequences)

    def _tile(self, seq, stride):
        plen = self.patch_nt_len
        patches = []
        for start in range(0, max(1, len(seq) - plen + 1), stride):
            patches.append(seq[start:start + plen])
            if len(patches) >= self.max_patches:
                break
        if len(seq) > plen:
            last = seq[len(seq) - plen:]
            if not patches or last != patches[-1]:
                patches.append(last)
        if not patches:
            patches.append(seq)
        return patches[:self.max_patches]

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        if self.is_train:
            
            if np.random.rand() <= self.scramble_rate:
                chars = list(seq)
                random.shuffle(chars)
                seq = ''.join(chars)
                label = self.zero_label
                                
            # Cut away a contiguous chunk at a random position, keep the rest
            if self.seq_drop_rate > 0 and len(seq) > self.patch_nt_len:
                drop_frac = random.random() * self.seq_drop_rate
                drop_nt = int(len(seq) * drop_frac)
                if drop_nt > 0 and len(seq) - drop_nt >= self.patch_nt_len:
                    cut_start = random.randint(0, len(seq) - drop_nt)
                    seq = seq[:cut_start] + seq[cut_start + drop_nt:]

            # Random offset shifts the tiling grid each epoch
            max_offset = self.patch_nt_len // 2
            if len(seq) > self.patch_nt_len + max_offset:
                offset = random.randint(0, max_offset)
                seq = seq[offset:]

            patches = self._tile(seq, self.train_stride)

            # Randomly drop patches (keep at least 1)
            if self.patch_drop_rate > 0 and len(patches) > 1:
                drop_chance = random.random() * self.patch_drop_rate
                keep = [p for p in patches if random.random() >= drop_chance]
                if not keep:
                    keep = [random.choice(patches)]
                patches = keep
        else:
            patches = self._tile(seq, self.eval_stride)

        toks = [self.tokenizer.tokenize(p) for p in patches]
        max_cl = max(t.size(1) for t in toks)
        padded = torch.zeros(len(toks), 6, max_cl, dtype=torch.long)
        for i, t in enumerate(toks):
            padded[i, :, :t.size(1)] = t
        return padded, len(toks), torch.from_numpy(label)


class BacterialSequenceDataset(Dataset):
    """Sequence-level bacterial genome dataset for aggregator training.

    Samples random bacterial subsequences with lengths drawn from the
    phage genome length distribution, tiles them into patches, and
    assigns a multi-hot label: the ``bacterial_fragment`` column plus
    the genus column when the genus appears in the host list.
    """

    def __init__(self, genome_store, tokenizer, phage_lengths,
                 n_samples, agg_num_classes, genus_to_idx: dict,
                 patch_nt_len=3072,
                 max_patches=512, coverage=2.0, is_train=True,
                 eval_stride=None):
        self.genome_store = genome_store
        self.tokenizer = tokenizer
        self.phage_lengths = phage_lengths
        self.n_samples = n_samples
        self.patch_nt_len = patch_nt_len
        self.max_patches = max_patches
        self.is_train = is_train
        self.split = 'train' if is_train else 'val'
        self.train_stride = max(1, int(patch_nt_len / coverage))
        self.eval_stride = eval_stride or patch_nt_len // 2
        self.num_classes = agg_num_classes
        self.genus_to_idx = genus_to_idx
        self.bact_idx = agg_num_classes - 1

    def __len__(self):
        return self.n_samples

    def _make_label(self, genus: str) -> np.ndarray:
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[self.bact_idx] = 1.0
        idx = self.genus_to_idx.get(genus)
        if idx is not None:
            label[idx] = 1.0
        return label

    def _tile(self, seq, stride):
        plen = self.patch_nt_len
        patches = []
        for start in range(0, max(1, len(seq) - plen + 1), stride):
            patches.append(seq[start:start + plen])
            if len(patches) >= self.max_patches:
                break
        if len(seq) > plen:
            last = seq[len(seq) - plen:]
            if not patches or last != patches[-1]:
                patches.append(last)
        if not patches:
            patches.append(seq)
        return patches[:self.max_patches]

    def __getitem__(self, idx):
        target_len = self.phage_lengths[
            np.random.randint(len(self.phage_lengths))]
        seq, genus = self.genome_store.sample_subseq(
            self.split, target_len)

        stride = self.train_stride if self.is_train else self.eval_stride
        patches = self._tile(seq, stride)

        label = self._make_label(genus)

        toks = [self.tokenizer.tokenize(p) for p in patches]
        max_cl = max(t.size(1) for t in toks)
        padded = torch.zeros(len(toks), 6, max_cl, dtype=torch.long)
        for i, t in enumerate(toks):
            padded[i, :, :t.size(1)] = t
        return padded, len(toks), torch.from_numpy(label)


def sequence_collate_fn(batch):
    patches_list, counts, labels = zip(*batch)
    max_n = max(counts)
    max_cl = max(p.size(2) for p in patches_list)
    B = len(batch)
    out = torch.zeros(B, max_n, 6, max_cl, dtype=torch.long)
    for i, p in enumerate(patches_list):
        out[i, :p.size(0), :, :p.size(2)] = p
    return out, torch.tensor(counts, dtype=torch.long), torch.stack(labels)
