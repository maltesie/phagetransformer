# PhageTransformer

A hierarchical DNA classifier for predicting the bacterial hosts of bacteriophages from genomic sequences. PhageTransformer processes raw nucleotide sequences into trinucleotides across all six reading frames, using a codon-level tokenizer, a feature extraction CNN, cross-frame attention, and a multi-level transformer architecture to produce multi-label predictions for 1064 host genera.

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

Create and activate a fresh environment (conda or venv), then install:

```bash
# Using conda
conda create -n phagetransformer python=3.11
conda activate phagetransformer

# Or using venv
python -m venv ptenv && source ptenv/bin/activate

# Install the package
pip install git+https://github.com/maltesie/phagetransformer.git
```

### GPU support

PhageTransformer uses PyTorch and will automatically use CUDA if available. For GPU support, ensure you have the appropriate PyTorch version installed for your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions.

## Quickstart

```bash
# Download pre-trained model weights (~100 MB)
phagetransformer init

# Or download to a specific directory
phagetransformer init --model_dir ./models/PT

# Predict hosts for phage genomes
phagetransformer predict --input phages.fasta --model_dir ./models/PT

# Train a new model
phagetransformer train --dataset_dir ./data --host_genome_dir ./genomes
```

All commands accept `--help` for full option details:

```bash
phagetransformer predict --help
```

## Commands

### init

Download pre-trained model weights and calibration files. Run this once after installation.

```bash
phagetransformer init --model_dir ./models/PT
```

| Parameter | Default | Description |
| --- | --- | --- |
| `--model_dir` | ~/.local/share/phagetransformer/default | Directory to store model files |
| `--force` | off | Re-download even if files already exist |

### predict

Predict bacterial hosts from phage genome sequences. Reads FASTA input (plain or gzipped), runs the full model, and outputs a TSV with host predictions and confidence scores.

```bash
phagetransformer predict --input phages.fasta --model_dir ./models/PT
```

#### Output format

The output TSV contains one row per prediction (a sequence can have multiple host predictions above threshold):

| Column | Description |
| --- | --- |
| `sequence_id` | FASTA record ID |
| `genus` | Predicted host genus name |
| `lineage` | Full taxonomic lineage (Phylum;Class;Order;Family;Genus) |
| `score` | Calibrated prediction score for this genus |
| `bacterial_score` | Score for the bacterial_fragment class (empty if model was trained without bacterial genomes) |
| `above_host_threshold` | `yes` if score ≥ host threshold |
| `above_bacterial_threshold` | `yes` if bacterial_score ≥ bacterial threshold |

If no prediction exceeds the threshold for a sequence, the single best prediction is returned with `above_host_threshold=no`.

#### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--input`, `-i` | *required* | Input FASTA file (plain or .gz) |
| `--model_dir` | *required* | Model directory containing calibration.json and checkpoints/ |
| `--output`, `-o` | stdout | Output TSV file |
| `--threshold` | — | Fixed score threshold (overrides --fdr) |
| `--fdr` | 0.1 | FDR level for threshold from calibration (e.g. `0.1` for 10% FDR, `0.2` for 20%) |
| `--bacterial_threshold` | 0.5 | Score threshold for the bacterial_fragment class |
| `--filter_output` | off | Only report predictions above the host threshold and below the bacterial threshold |
| `--top_k` | 0 | Max predictions per sequence (0 = all above threshold) |
| `--batch_size` | 1 | Sequences per batch (1 saves memory) |
| `--device` | cuda | Device (cuda or cpu) |

#### Threshold

By default, predictions are thresholded at the FDR 10% level from `calibration.json` (`--fdr 0.1`). Use `--fdr 0.2` for a more permissive 20% FDR, or `--threshold 0.3` to set an exact score cutoff (overrides `--fdr`).

The tiling stride used during prediction is read from `calibration.json` (saved during training) to ensure consistency between training evaluation and inference.

#### Filtering

Use `--filter_output` to get a clean list of confident phage-host predictions. This discards sequences that are either below the host threshold or flagged as bacterial:

```bash
phagetransformer predict --input phages.fasta --model_dir ./models/PT \
    --filter_output --bacterial_threshold 0.4
```

### train

Train a new model from scratch. Training runs in two phases: an **encoder phase** that trains the patch-level encoder on individual DNA windows, then an **aggregator phase** that trains the sequence-level aggregator on full genomes with the encoder frozen.

#### Preparing the dataset

Create a dataset directory with three splits — train, validation, and test. Each split is a gzipped FASTA of phage genomes plus a CSV that links every sequence to its host label(s):

```
data/
├── train.fna.gz
├── train.csv
├── val.fna.gz
├── val.csv
├── test.fna.gz
└── test.csv
```

**FASTA files** hold the phage nucleotide sequences. Each record's ID — the first whitespace-delimited token of the header line — is what matches it to a CSV row.

**CSV files** must contain exactly these two columns:

| Column | Description |
| --- | --- |
| `seq_id` | Matches a FASTA record ID in the same split |
| `host_labels` | One or more host labels. Separate multiple hosts with `\|`. |

Sequences and rows are matched by `seq_id`, not by file order, and only IDs present in both the FASTA and the CSV of a split are used. The set of host classes is built from `train.csv`; any label in `val.csv` or `test.csv` that does not appear in training is ignored — so all three files must use the **same label format**.

Example `train.csv`:

```
seq_id,host_labels
k141_111919,Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;Catonella
k141_179903,Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Prevotella
```

#### Downloading host genomes (optional)

Providing bacterial host genomes lets the model learn to distinguish phage DNA from bacterial DNA (the `bacterial_fragment` class). `download_host_genomes.py` from the scripts folder fetches GTDB species representatives for every genus in your training labels:

```bash
python download_host_genomes.py data/train.csv
```

This writes the genome FASTA files and a `host_genome_manifest.tsv` into `data/host_genomes/`, ready to pass to training. Skip this step to train on phage sequences only.

#### Running training

With the dataset directory (and optionally the host genomes) in place, start training with default settings:

```bash
phagetransformer train \
    --dataset_dir ./data \
    --host_genome_dir ./data/host_genomes \
    --run_name my_model \
    --output_folder ./models
```

Omit `--host_genome_dir` to train without bacterial genomes. Run `phagetransformer train --help` for the full list of architecture, schedule, and augmentation options.

#### Model directory structure

After training, the model directory contains everything needed for inference:

```
{output_folder}/{run_name}/
├── calibration.json       # Temperature, thresholds, eval_stride, model config, host list
├── checkpoints/
│   ├── best_encoder.pt
│   ├── best_aggregator.pt
│   └── ...
└── logs/
    ├── metrics.csv
    ├── train.log
    └── bacterial_species.tsv
```

The `calibration.json` file stores all parameters needed for reproducible inference: the temperature scaling factor, FDR-calibrated thresholds, the evaluation tiling stride, blocked classes, the host list with full lineage strings, and the model architecture config.

## Citation

*Paper forthcoming.*

## License

MIT
