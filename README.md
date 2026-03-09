# PhageTransformer

THIS IS STILL WORK IN PROGRESS. WILL REMOVE THIS HINT WHEN EVERYTHING WORKS.

A hierarchical DNA classifier for predicting the bacterial hosts of bacteriophages from genomic sequences. PhageTransformer processes raw nucleotide sequences into trinucleotides across all six reading frames, using a codon-level tokenizer, per-frame CNNs, cross-frame attention, and a multi-level transformer architecture to produce multi-label host genus predictions.

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

Create and activate a fresh environment (conda or venv), then install:

```bash
# Using conda
conda create -n phagetransformer python=3.11
conda activate phagetransformer

# Or using venv
python -m venv .venv && source .venv/bin/activate

# Install the package
pip install git+https://github.com/yourname/phagetransformer.git
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
phagetransformer predict --input phages.fasta --model_dir ~/.local/share/phagetransformer/default

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
# Download to default location (~/.local/share/phagetransformer/default)
phagetransformer init

# Download to a custom directory
phagetransformer init --model_dir ./models/PT

# Re-download even if files exist
phagetransformer init --force
```

| Parameter | Default | Description |
|---|---|---|
| `--model_dir` | ~/.local/share/phagetransformer/default | Directory to store model files |
| `--force` | off | Re-download even if files already exist |

The downloaded directory can then be passed as `--model_dir` to all other commands.

### predict

Predict bacterial hosts from phage genome sequences. Reads FASTA input (plain or gzipped), runs the full model, and outputs a TSV with host predictions and confidence scores.

```bash
phagetransformer predict --input phages.fasta --model_dir ./models/PT
```

| Parameter | Default | Description |
|---|---|---|
| `--input`, `-i` | *required* | Input FASTA file (plain or .gz) |
| `--model_dir` | *required* | Model directory containing calibration.json and checkpoints/ |
| `--output`, `-o` | stdout | Output TSV file |
| `--threshold` | from calibration | Score threshold for reporting predictions |
| `--fdr` | — | Use FDR-calibrated threshold (e.g. `0.1` for 10% FDR) |
| `--top_k` | 0 | Max predictions per sequence (0 = all above threshold) |
| `--batch_size` | 1 | Sequences per batch (1 saves memory) |
| `--device` | cuda | Device (cuda or cpu) |

The `--fdr` option uses thresholds computed during training calibration. Available FDR levels depend on the training run (typically 10% and 20%).

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
|---|---|---|
| `--transformer_dim` | 512 | Transformer embedding dimension |
| `--num_transformer_layers` | 4 | Encoder transformer depth |
| `--num_heads` | 8 | Attention heads |
| `--cnn_kernels` | 9 9 7 7 5 5 | CNN kernel sizes (determines compression factor) |
| `--frame_stats_channels` | 64 | Frame-stats branch hidden dimension |
| `--agg_layers` | 3 | Aggregator transformer depth |

**Training schedule:**

| Parameter | Default | Description |
|---|---|---|
| `--encoder_epochs` | 15 | Patch encoder pre-training epochs |
| `--aggregator_epochs` | 12 | Aggregator training epochs |
| `--learning_rate` | 5e-5 | Base learning rate |
| `--aggregator_lr_factor` | 0.5 | LR multiplier for aggregator phase |
| `--encoder_bacteria_ratio` | 0.2 | Bacterial patches as fraction of phage patches in encoder training |
| `--aggregator_bacteria_ratio` | 0.1 | Bacterial sequences as fraction of phage in aggregator training |

**Patching:**

| Parameter | Default | Description |
|---|---|---|
| `--patch_nt_len` | 3074 | Nucleotides per patch (3k+2 avoids frame padding) |
| `--max_patches` | 512 | Maximum patches per sequence |
| `--encoder_scramble_rate` | 0.1 | Fraction of encoder patches with shuffled nucleotides (regularization) |
| `--seq_scramble_rate` | 0.1 | Fraction of aggregator sequences with shuffled nucleotides |

**Loss:**

| Parameter | Default | Description |
|---|---|---|
| `--focal_gamma` | 1.8 | Focal loss gamma (0 = plain BCE) |
| `--pos_weight_cap` | 3.0 | Maximum positive class weight |

**Runtime:**

| Parameter | Default | Description |
|---|---|---|
| `--device` | cuda | Device |
| `--bf16` | off | Use bfloat16 mixed precision |
| `--gradient_checkpointing` | off | Trade compute for memory |
| `--num_workers` | 4 | Data loading workers |

**Checkpoints and resuming:**

| Parameter | Description |
|---|---|
| `--encoder_checkpoint` | Load pre-trained encoder weights |
| `--aggregator_checkpoint` | Load full model checkpoint |
| `--merge_val` | Merge train+val for final production run (skips eval and calibration) |
| `--calibrate_only` | Skip training, just run temperature calibration on validation set |

## Model directory structure

After training, the model directory (`--model_dir`) contains everything needed for inference:

```
models/PT/
├── calibration.json       # Temperature, thresholds, model config, host list
├── checkpoints/
│   ├── best_encoder.pt
│   ├── best_aggregator.pt
│   └── ...
└── logs/
    ├── metrics.csv
    └── bacterial_species.tsv
```

## Citation

*Paper forthcoming.*

## License

MIT
