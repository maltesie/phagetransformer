# PhageTransformer

A hierarchical DNA classifier for predicting the bacterial hosts of bacteriophages from genomic sequences. PhageTransformer processes raw nucleotide sequences into trinucleotides across all six reading frames, using a codon-level tokenizer, per-frame CNNs, cross-frame attention, and a multi-level transformer architecture to produce multi-label host genus predictions.

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
| `--threshold` | — | Fixed score threshold (highest priority) |
| `--fdr` | — | Use FDR-calibrated threshold (e.g. `0.1` for 10% FDR) |
| `--bacterial_threshold` | 0.5 | Score threshold for the bacterial_fragment class |
| `--filter_output` | off | Only report predictions above the host threshold and below the bacterial threshold |
| `--top_k` | 0 | Max predictions per sequence (0 = all above threshold) |
| `--batch_size` | 1 | Sequences per batch (1 saves memory) |
| `--device` | cuda | Device (cuda or cpu) |

#### Threshold priority

The host prediction threshold is resolved in this order (highest to lowest priority):

1. `--threshold` — explicit fixed value
2. `--fdr` — FDR level from calibration (e.g. `--fdr 0.2` for 20% FDR)
3. FDR 10% threshold from `calibration.json` (default if neither flag is given)
4. Generic `threshold` field from `calibration.json` (fallback if no FDR thresholds were computed)

The tiling stride used during prediction is read from `calibration.json` (saved during training) to ensure consistency between training evaluation and inference.

#### Filtering

Use `--filter_output` to get a clean list of confident phage-host predictions. This discards sequences that are either below the host threshold or flagged as bacterial:

```bash
phagetransformer predict --input phages.fasta --model_dir ./models/PT \
    --filter_output --bacterial_threshold 0.4
```

### train

Train a new model from scratch. Training proceeds in two phases: an **encoder phase** that trains the patch-level encoder on individual DNA windows, followed by an **aggregator phase** that trains the sequence-level aggregator on full genomes with the encoder frozen.

```bash
phagetransformer train \
    --dataset_dir ./data \
    --host_genome_dir ./genomes \
    --run_name my_model \
    --output_folder ./models
```

#### Data requirements

The training data directory (`--dataset_dir`) should contain `train.fna.gz`, `test.fna.gz`, and `phages_hosts.csv`. The host genome directory (`--host_genome_dir`) should contain `host_genome_manifest.tsv` pointing to bacterial genome FASTA files.

#### Key training parameters

**Architecture:**

| Parameter | Default | Description |
| --- | --- | --- |
| `--transformer_dim` | 512 | Transformer embedding dimension |
| `--num_transformer_layers` | 4 | Encoder transformer depth |
| `--num_heads` | 8 | Attention heads |
| `--cnn_kernels` | 9 9 7 7 5 5 | CNN kernel sizes (determines compression factor) |
| `--frame_stats_channels` | 64 | Frame-stats branch hidden dimension |
| `--agg_layers` | 3 | Aggregator transformer depth |

**Training schedule:**

| Parameter | Default | Description |
| --- | --- | --- |
| `--encoder_epochs` | 15 | Patch encoder pre-training epochs |
| `--aggregator_epochs` | 12 | Aggregator training epochs |
| `--learning_rate` | 5e-5 | Base learning rate |
| `--aggregator_lr_factor` | 0.5 | LR multiplier for aggregator phase |
| `--encoder_bacteria_ratio` | 0.2 | Bacterial patches as fraction of phage patches in encoder training |
| `--aggregator_bacteria_ratio` | 0.1 | Bacterial sequences as fraction of phage in aggregator training |

**Patching and tiling:**

| Parameter | Default | Description |
| --- | --- | --- |
| `--patch_nt_len` | 3074 | Nucleotides per patch (3k+2 avoids frame padding) |
| `--max_patches` | 512 | Maximum patches per sequence |
| `--eval_stride` | 2400 | Tiling stride for evaluation (saved to calibration.json for use by predict and evaluation scripts) |
| `--train_coverage` | 1.3 | Aggregator training stride = patch_nt_len / coverage |

**Augmentation:**

| Parameter | Default | Description |
| --- | --- | --- |
| `--encoder_scramble_rate` | 0.1 | Fraction of encoder patches with shuffled nucleotides (labels zeroed) |
| `--seq_scramble_rate` | 0.1 | Fraction of aggregator sequences with shuffled nucleotides — applied to both phage and bacterial sequences (labels zeroed) |
| `--seq_drop_rate` | 0.6 | Max fraction of sequence to cut during aggregator training |
| `--patch_drop_rate` | 0.1 | Max fraction of patches to drop during aggregator training |

**Bacterial genome sampling:**

| Parameter | Default | Description |
| --- | --- | --- |
| `--host_genome_dir` | — | Directory with `host_genome_manifest.tsv` and genome FASTA files |
| `--one_genome_per_genus` | off | Keep only one genome per genus (reduces redundancy but loses diversity) |
| `--genus_alpha` | 0.25 | Power-law exponent for genus sampling balance. Controls how much weight diverse genera get: 0 = all genera equally likely, 1 = proportional to number of species (old behavior). At 0.25, a genus with 600 species is ~5× more likely to be sampled than a single-species genus (down from 600×) |

**Loss:**

| Parameter | Default | Description |
| --- | --- | --- |
| `--focal_gamma` | 1.8 | Focal loss gamma (0 = plain BCE) |
| `--pos_weight_cap` | 3.0 | Maximum positive class weight |

**Runtime:**

| Parameter | Default | Description |
| --- | --- | --- |
| `--device` | cuda | Device |
| `--bf16` | off | Use bfloat16 mixed precision |
| `--gradient_checkpointing` | off | Trade compute for memory |
| `--num_workers` | 4 | Data loading workers |

**Checkpoints and special modes:**

| Parameter | Description |
| --- | --- |
| `--encoder_checkpoint` | Load pre-trained encoder weights |
| `--aggregator_checkpoint` | Load full model checkpoint |
| `--merge_val` | Merge train+val for final production run. Skips evaluation and calibration. Bacterial genomes use the full sequence (no val region held out) |
| `--calibrate_only` | Skip training, just run temperature calibration on validation set |

## Model directory structure

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

## Evaluation scripts

The `scripts/` directory contains standalone evaluation and comparison scripts:

| Script | Description |
| --- | --- |
| `evaluate_phages.py` | Evaluate phage host prediction on validation and test sets. Generates training curves, calibration plots, taxonomic breakdown, embedding analysis, and PR curves |
| `evaluate_bacteria.py` | Evaluate bacterial detection and classification on held-out bacterial genome regions |
| `compare.py` | Compare PhageTransformer predictions against iPHoP and CHERRY on external test datasets |
| `attention.py` | Extract and visualize hierarchical importance weights as genome annotation heatmaps |
| `scan.py` | Scan bacterial genomes with a sliding window to detect candidate prophage regions |

All evaluation scripts share a common plot styling system (`eval_utils.py`) for consistent publication-quality figures. They read the evaluation tiling stride from `calibration.json` to match training conditions, with `--eval_stride` available as an override.

## Citation

*Paper forthcoming.*

## License

MIT
