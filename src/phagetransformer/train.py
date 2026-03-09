#!/usr/bin/env python3
"""Hierarchical DNA classifier — two-phase staged training.

Encoder phase:     Patch encoder (with integrated frame-stats branch)
                   + temporary classifier head on individual patches.
                   When --host_genome_dir is provided, bacterial patches
                   are mixed in with a bacterial_fragment class.
Aggregator phase:  Freeze encoder, train aggregator on full sequences.

Focal BCE loss with capped pos_weight throughout.  Patch-level curriculum
gives rare-class sequences denser tiling (more patches per epoch).
"""

import argparse
import gc
import json
import logging
import os
import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from .model import (
    HierarchicalDNAClassifier, PatchClassifier,
    CodonTokenizer,
)
from .dataset import (
    load_phage_host_merged, load_phage_host_test,
    BacterialGenomeStore, BacterialPatchDataset, BacterialSequenceDataset,
    RandomPatchDataset, EvalPatchDataset, patch_collate_fn,
    PatchSequenceDataset, sequence_collate_fn,
)
from .utils import (
    compute_metrics, get_cosine_schedule_with_warmup,
    save_checkpoint, load_component, load_best_or_last,
    CSVLogger, run_calibration,
)

logger = logging.getLogger(__name__)
        
# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ((1 - p_t) ** self.gamma) * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss


# ---------------------------------------------------------------------------
# Batch unpacking helpers (keeps train/eval loops generic)
# ---------------------------------------------------------------------------

def _unpack_patch_batch(model, batch, device):
    """Encoder / frame-stats phases: batch = (tokens, labels)."""
    tokens, labels = batch
    tokens = tokens.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return model(tokens), labels


def _unpack_sequence_batch(model, batch, device):
    """Aggregator phase: batch = (patches, counts, labels)."""
    patches, counts, labels = batch
    patches = patches.to(device, non_blocking=True)
    counts = counts.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return model(patches, counts), labels

# ---------------------------------------------------------------------------
# Generic train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    scaler, grad_accum, max_grad_norm, log_every, epoch,
                    unpack_fn):
    model.train()
    # If patch encoder is frozen, keep it in eval mode (preserves BatchNorm)
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model  # torch.compile
    if hasattr(raw, 'patch_encoder') and not any(
            p.requires_grad for p in raw.patch_encoder.parameters()):
        raw.patch_encoder.eval()
    total_loss, n_steps = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for step, batch in enumerate(loader):
        with torch.amp.autocast('cuda', enabled=scaler is not None,
                                dtype=torch.bfloat16):
            logits, labels = unpack_fn(model, batch, device)
            loss = criterion(logits, labels) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * grad_accum
        n_steps += 1

        if log_every and (step + 1) % log_every == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"  epoch {epoch} | step {step+1}/{len(loader)} | "
                        f"loss {total_loss/n_steps:.4f} | lr {lr:.2e} | "
                        f"{time.time()-t0:.1f}s")

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold, tag, unpack_fn,
             n_extra_classes=0):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in loader:
        logits, labels = unpack_fn(model, batch, device)
        total_loss += criterion(logits, labels).item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    if n_extra_classes > 0:
        core_logits = all_logits[:, :-n_extra_classes]
        core_labels = all_labels[:, :-n_extra_classes]

        # Split samples: bacterial_fragment is the last column
        bact_flag = all_labels[:, -1].bool()
        phage_mask = ~bact_flag
        bact_mask = bact_flag

        # Phage-only genus metrics (drives checkpoint selection)
        n_phage = phage_mask.sum().item()
        if n_phage > 0:
            m = compute_metrics(core_logits[phage_mask],
                                core_labels[phage_mask], threshold)
        else:
            m = compute_metrics(core_logits, core_labels, threshold)
        logger.info(f"  [{tag}:phage] n={n_phage}  "
                    f"micro-F1={m['micro_f1']:.4f}  "
                    f"macro-F1={m['macro_f1']:.4f}  "
                    f"micro-P={m['micro_p']:.4f}  "
                    f"micro-R={m['micro_r']:.4f}")

        # Bacterial genus metrics (informational)
        n_bact = bact_mask.sum().item()
        if n_bact > 0:
            m_bact_genus = compute_metrics(core_logits[bact_mask],
                                           core_labels[bact_mask], threshold)
            logger.info(f"  [{tag}:bact_genus] n={n_bact}  "
                        f"micro-F1={m_bact_genus['micro_f1']:.4f}  "
                        f"micro-P={m_bact_genus['micro_p']:.4f}  "
                        f"micro-R={m_bact_genus['micro_r']:.4f}")

        # Bacterial fragment detection (binary, all samples)
        extra_logits = all_logits[:, -n_extra_classes:]
        extra_labels = all_labels[:, -n_extra_classes:]
        m_extra = compute_metrics(extra_logits, extra_labels, threshold)
        logger.info(f"  [{tag}:bact_detect]  "
                    f"micro-F1={m_extra['micro_f1']:.4f}  "
                    f"micro-P={m_extra['micro_p']:.4f}  "
                    f"micro-R={m_extra['micro_r']:.4f}")
    else:
        m = compute_metrics(all_logits, all_labels, threshold)

    m['loss'] = total_loss / max(len(loader), 1)

    logger.info(f"  [{tag}] loss={m['loss']:.4f}  "
                f"micro-F1={m['micro_f1']:.4f}  macro-F1={m['macro_f1']:.4f}  "
                f"micro-P={m['micro_p']:.4f}  micro-R={m['micro_r']:.4f}  "
                f"classes={m['n_classes_with_support']}")
    return m, all_logits, all_labels





# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(
    phase_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    unpack_fn: Callable,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_fraction: float,
    min_lr_ratio: float,
    grad_accum: int,
    max_grad_norm: float,
    log_every: int,
    eval_threshold: float,
    ckpt_dir: str,
    csv_log: CSVLogger,
    use_bf16: bool,
    epoch_offset: int = 0,
    best_val_f1: float = 0.0,
    checkpoint_model: Optional[nn.Module] = None,
    save_component: Optional[str] = None,
    n_extra_classes: int = 0,
) -> Tuple[float, int]:
    """Run one training phase. Returns (best_val_f1, last_global_epoch).

    If ``checkpoint_model`` is given, checkpoints reference it instead of
    ``model``.  If ``save_component`` is also given (e.g. 'patch_encoder'),
    only that submodule's state_dict is saved — allowing component-level
    checkpoint loading between phases.
    """
    save_model = checkpoint_model or model

    logger.info(f"\n{'='*60}")
    logger.info(f"  {phase_name}  ({num_epochs} epochs, lr={learning_rate:.1e})")
    logger.info(f"{'='*60}")

    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    logger.info(f"  trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(params, lr=learning_rate,
                                  weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs // grad_accum
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio)

    scaler = torch.amp.GradScaler('cuda') if use_bf16 else None

    logger.info(f"  steps/epoch={len(train_loader)}  total_steps={total_steps}  "
                f"warmup={warmup_steps}")

    for ep in range(num_epochs):
        global_ep = epoch_offset + ep + 1
        t0 = time.time()
        logger.info(f"--- {phase_name} epoch {ep+1}/{num_epochs}  "
                     f"(global {global_ep}) ---")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            scaler, grad_accum, max_grad_norm, log_every, global_ep,
            unpack_fn,
        )

        val_m = {}
        if val_loader is not None:
            val_m, _, _ = evaluate(model, val_loader, criterion, device,
                             eval_threshold, 'val', unpack_fn,
                             n_extra_classes)
        test_m = {}
        if test_loader is not None:
            test_m, _, _ = evaluate(model, test_loader, criterion, device,
                              eval_threshold, 'test', unpack_fn,
                              n_extra_classes)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        csv_log.log({
            'phase': phase_name, 'epoch': global_ep,
            'train_loss': f"{train_loss:.5f}",
            'val_loss': f"{val_m['loss']:.5f}" if val_m else '',
            'val_micro_f1': f"{val_m['micro_f1']:.5f}" if val_m else '',
            'val_macro_f1': f"{val_m['macro_f1']:.5f}" if val_m else '',
            'val_micro_p': f"{val_m['micro_p']:.5f}" if val_m else '',
            'val_micro_r': f"{val_m['micro_r']:.5f}" if val_m else '',
            'test_loss': f"{test_m['loss']:.5f}" if test_m else '',
            'test_micro_f1': f"{test_m['micro_f1']:.5f}" if test_m else '',
            'test_macro_f1': f"{test_m['macro_f1']:.5f}" if test_m else '',
            'test_micro_p': f"{test_m['micro_p']:.5f}" if test_m else '',
            'test_micro_r': f"{test_m['micro_r']:.5f}" if test_m else '',
            'lr': f"{lr_now:.2e}", 'elapsed_s': f"{elapsed:.1f}",
        })

        # Checkpoint
        if save_component:
            state_dict = getattr(save_model, save_component).state_dict()
        else:
            state_dict = save_model.state_dict()
        ckpt_state = {
            'phase': phase_name, 'epoch': global_ep,
            'model_state_dict': state_dict,
            'best_val_f1': best_val_f1,
        }

        if val_m:
            is_best = val_m['micro_f1'] > best_val_f1
            if is_best:
                best_val_f1 = val_m['micro_f1']
                ckpt_state['best_val_f1'] = best_val_f1
                save_checkpoint(ckpt_state,
                                os.path.join(ckpt_dir, f'best_{phase_name}.pt'))
                logger.info(f"  ★ new best val micro-F1: {best_val_f1:.5f}")

        # Always save last epoch (needed for merge_val mode)
        save_checkpoint(ckpt_state,
                        os.path.join(ckpt_dir, f'last_{phase_name}.pt'))

    return best_val_f1, epoch_offset + num_epochs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train hierarchical DNA classifier (3-phase)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- model: patch encoder -----------------------------------------------
    g = parser.add_argument_group('Patch encoder')
    g.add_argument('--cnn_embed_dim', type=int, default=128)
    g.add_argument('--cnn_hidden_dim', type=int, default=500)
    g.add_argument('--cnn_kernels', type=int, nargs='+', default=[9, 9, 7, 7, 5])
    g.add_argument('--transformer_dim', type=int, default=300)
    g.add_argument('--num_transformer_layers', type=int, default=4)
    g.add_argument('--num_heads', type=int, default=10)

    # -- model: aggregator --------------------------------------------------
    g = parser.add_argument_group('Sequence aggregator')
    g.add_argument('--agg_layers', type=int, default=4)
    g.add_argument('--agg_heads', type=int, default=10)

    # -- model: frame stats branch ------------------------------------------
    g = parser.add_argument_group('Frame stats branch')
    g.add_argument('--frame_stats_channels', type=int, default=64,
                   help='Hidden channels in FrameStatsBranch conv layers')
    g.add_argument('--frame_stats_kernel_size', type=int, default=3,
                   help='Kernel size for FrameStatsBranch conv layers')

    # -- patching / curriculum ----------------------------------------------
    g = parser.add_argument_group('Patching')
    g.add_argument('--encoder_scramble_rate', type=float, default=0.1)
    g.add_argument('--seq_scramble_rate', type=float, default=0.1)
    g.add_argument('--patch_nt_len', type=int, default=3074)
    g.add_argument('--min_patches_per_seq', type=float, default=1.0,
                   help='Encoder phase: patches per epoch for common-class sequences')
    g.add_argument('--max_patches_per_seq', type=float, default=2.0,
                   help='Encoder phase: patches per epoch for rarest-class sequences')
    g.add_argument('--eval_stride', type=int, default=2400,
                   help='Stride for eval tiling (default: patch_nt_len // 2)')
    g.add_argument('--train_coverage', type=float, default=1.3,
                   help='Aggregator phase: stride = patch_nt_len / coverage')
    g.add_argument('--seq_drop_rate', type=float, default=0.6,
                   help='Aggregator phase train: max fraction of sequence to truncate')
    g.add_argument('--patch_drop_rate', type=float, default=0.1,
                   help='Aggregator phase train: max fraction of patches to drop after tiling')
    g.add_argument('--patches_per_forward', type=int, default=32)
    g.add_argument('--max_patches', type=int, default=512,
                   help='Hard cap on patches per sequence (memory guard)')
    
    # -- loss ---------------------------------------------------------------
    g = parser.add_argument_group('Loss')
    g.add_argument('--focal_gamma', type=float, default=1.8)
    g.add_argument('--pos_weight_cap', type=float, default=3.0)
    g.add_argument('--pos_weight_scaling_factor', type=float, default=2.0)
    
    # -- phase epochs -------------------------------------------------------
    g = parser.add_argument_group('Training phases')
    g.add_argument('--encoder_epochs', type=int, default=15,
                   help='Patch encoder pre-training epochs')
    g.add_argument('--encoder_bacteria_ratio', type=float, default=0.2,
                   help='Bacterial patches as fraction of phage patch count '
                        'in encoder training (0 = disable)')
    g.add_argument('--aggregator_epochs', type=int, default=12,
                   help='Aggregator training epochs (encoder + branch frozen)')
    g.add_argument('--aggregator_lr_factor', type=float, default=0.5,
                   help='LR multiplier for aggregator phase')
    g.add_argument('--aggregator_batch_size', type=int, default=8,
                   help='Batch size for aggregator phase')
    g.add_argument('--aggregator_bacteria_ratio', type=float, default=0.1,
                   help='Bacterial sequences as fraction of phage count '
                        'in aggregator training (0 = disable)')

    # -- training -----------------------------------------------------------
    g = parser.add_argument_group('Training')
    g.add_argument('--encoder_batch_size', type=int, default=32,
                   help='Batch size for patch-level encoder phase')
    g.add_argument('--learning_rate', type=float, default=5e-5)
    g.add_argument('--weight_decay', type=float, default=0.01)
    g.add_argument('--warmup_fraction', type=float, default=0.05,
                   help='Fraction of phase steps used for LR warmup')
    g.add_argument('--min_lr_ratio', type=float, default=0.1)
    g.add_argument('--gradient_accumulation_steps', type=int, default=1)
    g.add_argument('--max_grad_norm', type=float, default=1.0)
    g.add_argument('--dropout', type=float, default=0.05)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--eval_threshold', type=float, default=0.5)
    g.add_argument('--min_val_precision', type=float, default=0.6,
                   help='Block classes with val precision below this (0 = disable)')
    g.add_argument('--min_val_support', type=int, default=1,
                   help='Min positive val samples to evaluate a class for blocking')

    # -- runtime ------------------------------------------------------------
    g = parser.add_argument_group('Runtime')
    g.add_argument('--device', type=str, default='cuda')
    g.add_argument('--num_workers', type=int, default=4)
    g.add_argument('--bf16', action='store_true')
    g.add_argument('--compile', action='store_true')
    g.add_argument('--gradient_checkpointing', action='store_true')
    g.add_argument('--log_every_n_steps', type=int, default=5000)

    # -- paths --------------------------------------------------------------
    g = parser.add_argument_group('Paths')
    g.add_argument('--run_name', type=str, default='HierDNA')
    g.add_argument('--output_folder', type=str, default='../../../models')
    g.add_argument('--dataset_dir', type=str, default=None)
    g.add_argument('--testset_dir', type=str, default=None)
    g.add_argument('--host_genome_dir', type=str, default=None,
                   help='Directory with host_genome_manifest.tsv '
                        '(required for bacterial_fragment class in encoder '
                        'and aggregator phases)')
    g.add_argument('--one_genome_per_genus', action='store_true',
                   help='Keep only one genome per genus from the manifest')
    g.add_argument('--encoder_checkpoint', type=str, default=None,
                   help='Pre-trained encoder checkpoint to load before '
                        'encoder phase (or to skip it with --encoder_epochs 0)')
    g.add_argument('--aggregator_checkpoint', type=str, default=None,
                   help='Full model checkpoint (aggregator phase output). '
                        'Also used for --calibrate_only.')
    g.add_argument('--merge_val', action='store_true',
                   help='Merge train+val for final production run. '
                        'Disables eval, saves last-epoch checkpoint only, '
                        'skips calibration.')
    g.add_argument('--calibrate_only', action='store_true',
                   help='Load checkpoint and only run temperature calibration')

    args = parser.parse_args()

    # ---- dirs ------------------------------------------------------------
    run_dir = os.path.join(args.output_folder, args.run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---- logging ---------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(log_dir, 'train.log'))],
    )
    logger.info(f"args: {json.dumps(vars(args), indent=2)}")

    # ---- seed ------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- data ------------------------------------------------------------
    here = os.path.dirname(os.path.realpath(__file__))
    ds_path = args.dataset_dir or os.path.join(here, '..', '..', '..', 'datasets', 'PhageTransformer')
    ts_path = args.testset_dir or os.path.join(here, '..', '..', '..', 'data', 'testsets')

    logger.info("Loading data …")
    train_seqs, train_labels, val_seqs, val_labels, hosts = \
        load_phage_host_merged(ds_path)
    test_seqs, test_labels = load_phage_host_test(ts_path, hosts)
    num_classes = len(hosts)

    if args.merge_val:
        logger.info("  --merge_val: merging train + val into one training set")
        train_seqs = train_seqs + val_seqs
        train_labels = np.concatenate([train_labels, val_labels], axis=0)
        val_seqs, val_labels = [], np.zeros((0, num_classes), dtype=np.float32)

    logger.info(f"  train={len(train_seqs)}  val={len(val_seqs)}  "
                f"test={len(test_seqs)}  classes={num_classes}")

    # ---- bacterial genomes ------------------------------------------------
    genome_store = None
    if args.host_genome_dir:
        logger.info("Loading bacterial genomes …")
        genome_store = BacterialGenomeStore(
            args.host_genome_dir,
            num_workers=args.num_workers,
            seed=args.seed,
            one_per_genus=args.one_genome_per_genus,
        )
        genome_store.write_species_log(
            os.path.join(run_dir, 'logs', 'bacterial_species.tsv'))

    # Determine final number of output classes (genus + optional bacterial)
    use_bacteria = (genome_store is not None
                    and (args.encoder_bacteria_ratio > 0
                         or args.aggregator_bacteria_ratio > 0))
    n_extra_classes = 1 if use_bacteria else 0
    agg_num_classes = num_classes + n_extra_classes

    # Build genus → label index mapping for bacterial genome labeling
    genus_to_idx = {}
    if use_bacteria:
        for i, h in enumerate(hosts):
            genus = h.split(';')[-1] if ';' in h else h
            genus_to_idx[genus] = i
        n_matched = sum(1 for sp in genome_store.species_list
                        if sp.split()[0] in genus_to_idx)
        logger.info(f"  Genus mapping: {len(genus_to_idx)} host genera, "
                    f"{n_matched}/{len(genome_store.species_list)} bacterial "
                    f"species have a matching host genus")

    # ---- model config dict (saved with calibration) ----------------------
    model_config = dict(
        num_classes=agg_num_classes, cnn_embed_dim=args.cnn_embed_dim,
        cnn_hidden_dim=args.cnn_hidden_dim, transformer_dim=args.transformer_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads, agg_layers=args.agg_layers,
        agg_heads=args.agg_heads,
        patches_per_forward=args.patches_per_forward,
        frame_stats_channels=args.frame_stats_channels,
        frame_stats_kernel_size=args.frame_stats_kernel_size,
        dropout=args.dropout, cnn_kernel_sizes=args.cnn_kernels,
        patch_nt_len=args.patch_nt_len,
    )

    device = torch.device(args.device)
    tokenizer = CodonTokenizer()

    # ---- model -----------------------------------------------------------
    model = HierarchicalDNAClassifier(
        **{k: v for k, v in model_config.items()},
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f"  total params: {total_p:,}")
    logger.info(f"  CNN: {len(args.cnn_kernels)} layers, kernels={args.cnn_kernels}, "
                f"compression={model.patch_encoder.frame_cnn.compression_factor}x")

    # ---- common loader kwargs --------------------------------------------
    ldr_kw = dict(num_workers=args.num_workers, pin_memory=True,
                  persistent_workers=args.num_workers > 0)

    # =====================================================================
    # CALIBRATE-ONLY MODE
    # =====================================================================
    if args.calibrate_only:
        if args.merge_val:
            logger.error("  --calibrate_only and --merge_val are incompatible "
                         "(no val set to calibrate on)")
            return
        ckpt_path = args.aggregator_checkpoint or os.path.join(
            ckpt_dir, 'best_aggregator.pt')
        logger.info(f"  Calibrate-only mode: loading {ckpt_path}")
        load_component(ckpt_path, model)

        # Pad labels with bacterial_fragment column if model was trained with it
        calib_val_labels = val_labels
        calib_hosts = hosts
        if use_bacteria:
            zero_col = np.zeros((len(val_labels), 1), dtype=np.float32)
            calib_val_labels = np.concatenate([val_labels, zero_col], axis=1)
            calib_hosts = np.append(hosts, 'bacterial_fragment')

        # Build val loader for calibration
        val_seq_ds = PatchSequenceDataset(
            val_seqs, calib_val_labels, tokenizer,
            patch_nt_len=args.patch_nt_len, max_patches=args.max_patches,
            is_train=False, eval_stride=args.eval_stride,
        )
        val_loader = DataLoader(
            val_seq_ds, batch_size=args.aggregator_batch_size, shuffle=False,
            collate_fn=sequence_collate_fn, **ldr_kw)

        run_calibration(model, val_loader, device, _unpack_sequence_batch,
                        run_dir, calib_hosts, model_config, args.eval_threshold,
                        args.min_val_precision, args.min_val_support)
        logger.info("Calibration complete.")
        return

    # ---- pos_weight ------------------------------------------------------
    cc = train_labels.sum(axis=0).clip(min=1)
    reference_count = 100.0
    pw = np.clip(reference_count / cc, a_min=1.0, a_max=args.pos_weight_cap) * args.pos_weight_scaling_factor

    # Extend with bacterial_fragment class if using bacteria
    if use_bacteria:
        pw = np.append(pw, 1.0)

    pos_weight = torch.tensor(pw, dtype=torch.float32)

    rare = (cc < 10).sum()
    med = ((cc >= 10) & (cc < 100)).sum()
    common = (cc >= 100).sum()
    logger.info(f"  class dist: {rare} <10, {med} 10-99, {common} >=100")
    logger.info(f"  class counts: min={cc.min():.0f}  median={np.median(cc):.0f}  "
                f"max={cc.max():.0f}")
    logger.info(f"  pos_weight: ref={reference_count:.0f}  "
                f"range=[{pw.min():.2f}, {pw.max():.2f}]  "
                f"(cap {args.pos_weight_cap})")

    criterion = FocalBCEWithLogitsLoss(
        gamma=args.focal_gamma, pos_weight=pos_weight.to(device))

    if args.compile:
        logger.info("  compiling model …")
        model = torch.compile(model, mode='reduce-overhead')

    # ---- CSV log ---------------------------------------------------------
    csv_fields = [
        'phase', 'epoch', 'train_loss',
        'val_loss', 'val_micro_f1', 'val_macro_f1', 'val_micro_p', 'val_micro_r',
        'test_loss', 'test_micro_f1', 'test_macro_f1', 'test_micro_p', 'test_micro_r',
        'lr', 'elapsed_s',
    ]
    csv_log = CSVLogger(os.path.join(log_dir, 'metrics.csv'), csv_fields)

    best_f1 = 0.0
    epoch_offset = 0

    # ---- Load CLI-specified component checkpoints -----------------------
    if args.encoder_checkpoint:
        load_component(args.encoder_checkpoint, model, 'patch_encoder')
    if args.aggregator_checkpoint:
        load_component(args.aggregator_checkpoint, model)

    # =====================================================================
    # Patch-level datasets (encoder phase)
    # =====================================================================
    # Extend labels with bacterial_fragment column (zeros for phage)
    if use_bacteria:
        enc_train_labels = np.concatenate(
            [train_labels, np.zeros((len(train_labels), 1), dtype=np.float32)],
            axis=1)
        enc_val_labels = np.concatenate(
            [val_labels, np.zeros((len(val_labels), 1), dtype=np.float32)],
            axis=1)
    else:
        enc_train_labels = train_labels
        enc_val_labels = val_labels

    train_patch_ds = RandomPatchDataset(
        train_seqs, enc_train_labels, tokenizer,
        patch_nt_len=args.patch_nt_len,
        min_patches_per_seq=args.min_patches_per_seq,
        max_patches_per_seq=args.max_patches_per_seq,
        scramble_rate=args.encoder_scramble_rate,
    )

    # Add bacterial patches to encoder training
    enc_use_bacteria = (use_bacteria and args.encoder_bacteria_ratio > 0)
    if enc_use_bacteria:
        n_bact_patches = max(1, int(len(train_patch_ds)
                                     * args.encoder_bacteria_ratio))
        bact_train_patch_ds = BacterialPatchDataset(
            genome_store, tokenizer, num_classes=agg_num_classes,
            genus_to_idx=genus_to_idx,
            n_samples=n_bact_patches,
            patch_nt_len=args.patch_nt_len, is_train=True,
            scramble_rate=args.encoder_scramble_rate,
        )
        combined_train_patch_ds = ConcatDataset(
            [train_patch_ds, bact_train_patch_ds])
        logger.info(f"  Encoder patches: {len(train_patch_ds)} phage + "
                    f"{n_bact_patches} bacterial "
                    f"= {len(combined_train_patch_ds)} total")
    else:
        combined_train_patch_ds = train_patch_ds
        pps = train_patch_ds._pps
        logger.info(f"  Encoder patches: train={len(train_patch_ds)}  "
                     f"patches/seq: min={min(pps)} median={int(np.median(pps))} "
                     f"max={max(pps)}")

    enc_train_loader = DataLoader(
        combined_train_patch_ds, batch_size=args.encoder_batch_size,
        shuffle=True, collate_fn=patch_collate_fn, drop_last=True, **ldr_kw)

    enc_val_loader = None
    if not args.merge_val:
        val_patch_ds = EvalPatchDataset(
            val_seqs, enc_val_labels, tokenizer,
            patch_nt_len=args.patch_nt_len, stride=args.eval_stride,
        )
        if enc_use_bacteria:
            n_bact_val = max(1, int(len(val_patch_ds)
                                     * args.encoder_bacteria_ratio))
            bact_val_patch_ds = BacterialPatchDataset(
                genome_store, tokenizer, num_classes=agg_num_classes,
                genus_to_idx=genus_to_idx,
                n_samples=n_bact_val,
                patch_nt_len=args.patch_nt_len, is_train=False,
            )
            combined_val_patch_ds = ConcatDataset(
                [val_patch_ds, bact_val_patch_ds])
        else:
            combined_val_patch_ds = val_patch_ds
        enc_val_loader = DataLoader(
            combined_val_patch_ds, batch_size=args.encoder_batch_size,
            shuffle=False, collate_fn=patch_collate_fn, **ldr_kw)

    # =====================================================================
    # ENCODER PHASE — Patch encoder pre-training
    # =====================================================================
    if args.encoder_epochs > 0:
        patch_clf = PatchClassifier(
            model.patch_encoder, agg_num_classes, args.dropout).to(device)

        best_f1, epoch_offset = run_phase(
            phase_name='encoder',
            model=patch_clf,
            train_loader=enc_train_loader,
            val_loader=enc_val_loader,
            test_loader=None,
            unpack_fn=_unpack_patch_batch,
            criterion=criterion, device=device,
            num_epochs=args.encoder_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_fraction=args.warmup_fraction,
            min_lr_ratio=args.min_lr_ratio,
            grad_accum=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every_n_steps,
            eval_threshold=args.eval_threshold,
            ckpt_dir=ckpt_dir, csv_log=csv_log, use_bf16=args.bf16,
            epoch_offset=epoch_offset, best_val_f1=best_f1,
            checkpoint_model=model,
            save_component='patch_encoder',
            n_extra_classes=n_extra_classes,
        )
        logger.info("  Encoder phase done.")
        del patch_clf
        torch.cuda.empty_cache()
        load_best_or_last(ckpt_dir, 'encoder', model,
                          component='patch_encoder')

    # Free patch-level datasets
    del train_patch_ds, enc_train_loader
    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # Sequence-level datasets (aggregator phase)
    # =====================================================================
    if use_bacteria:
        # Extend labels with bacterial_fragment column (zeros for phage)
        zero_col = np.zeros((len(train_labels), 1), dtype=np.float32)
        agg_train_labels = np.concatenate([train_labels, zero_col], axis=1)
        agg_val_labels = np.concatenate(
            [val_labels, np.zeros((len(val_labels), 1), dtype=np.float32)],
            axis=1)
        agg_test_labels = np.concatenate(
            [test_labels, np.zeros((len(test_labels), 1), dtype=np.float32)],
            axis=1)

        phage_lengths = [len(s) for s in train_seqs]
        n_bacteria_train = max(1, int(len(train_seqs)
                                      * args.aggregator_bacteria_ratio))
        logger.info(f"  Aggregator: {len(train_seqs)} phage + "
                    f"{n_bacteria_train} bacterial sequences/epoch "
                    f"({agg_num_classes} classes)")
    else:
        agg_train_labels = train_labels
        agg_val_labels = val_labels
        agg_test_labels = test_labels

    train_seq_ds = PatchSequenceDataset(
        train_seqs, agg_train_labels, tokenizer,
        patch_nt_len=args.patch_nt_len, max_patches=args.max_patches,
        is_train=True, coverage=args.train_coverage,
        seq_drop_rate=args.seq_drop_rate,
        patch_drop_rate=args.patch_drop_rate,
        scramble_rate=args.seq_scramble_rate,
    )
    if use_bacteria:
        bact_train_ds = BacterialSequenceDataset(
            genome_store, tokenizer, phage_lengths,
            n_samples=n_bacteria_train,
            agg_num_classes=agg_num_classes,
            genus_to_idx=genus_to_idx,
            patch_nt_len=args.patch_nt_len,
            max_patches=args.max_patches,
            coverage=args.train_coverage, is_train=True,
        )
        combined_train_ds = ConcatDataset([train_seq_ds, bact_train_ds])
    else:
        combined_train_ds = train_seq_ds

    seq_train_loader = DataLoader(
        combined_train_ds, batch_size=args.aggregator_batch_size, shuffle=True,
        collate_fn=sequence_collate_fn, drop_last=True, **ldr_kw)

    seq_val_loader = None
    seq_test_loader = None
    if not args.merge_val:
        val_seq_ds = PatchSequenceDataset(
            val_seqs, agg_val_labels, tokenizer,
            patch_nt_len=args.patch_nt_len, max_patches=args.max_patches,
            is_train=False, eval_stride=args.eval_stride,
        )
        if use_bacteria:
            n_bacteria_val = max(1, int(len(val_seqs)
                                        * args.aggregator_bacteria_ratio))
            bact_val_ds = BacterialSequenceDataset(
                genome_store, tokenizer, phage_lengths,
                n_samples=n_bacteria_val,
                agg_num_classes=agg_num_classes,
                genus_to_idx=genus_to_idx,
                patch_nt_len=args.patch_nt_len,
                max_patches=args.max_patches, is_train=False,
                eval_stride=args.eval_stride,
            )
            combined_val_ds = ConcatDataset([val_seq_ds, bact_val_ds])
        else:
            combined_val_ds = val_seq_ds
        seq_val_loader = DataLoader(
            combined_val_ds, batch_size=args.aggregator_batch_size, shuffle=False,
            collate_fn=sequence_collate_fn, **ldr_kw)

    test_seq_ds = PatchSequenceDataset(
        test_seqs, agg_test_labels, tokenizer,
        patch_nt_len=args.patch_nt_len, max_patches=args.max_patches,
        is_train=False, eval_stride=args.eval_stride,
    )
    if use_bacteria:
        n_bacteria_test = max(1, int(len(test_seqs)
                                     * args.aggregator_bacteria_ratio))
        bact_test_ds = BacterialSequenceDataset(
            genome_store, tokenizer, phage_lengths,
            n_samples=n_bacteria_test,
            agg_num_classes=agg_num_classes,
            genus_to_idx=genus_to_idx,
            patch_nt_len=args.patch_nt_len,
            max_patches=args.max_patches, is_train=False,
            eval_stride=args.eval_stride,
        )
        combined_test_ds = ConcatDataset([test_seq_ds, bact_test_ds])
    else:
        combined_test_ds = test_seq_ds
    seq_test_loader = DataLoader(
        combined_test_ds, batch_size=args.aggregator_batch_size, shuffle=False,
        collate_fn=sequence_collate_fn, **ldr_kw)

    # =====================================================================
    # AGGREGATOR PHASE — (encoder frozen)
    # =====================================================================
    if args.aggregator_epochs > 0:
        # Ensure best encoder weights are loaded
        load_best_or_last(ckpt_dir, 'encoder', model,
                          component='patch_encoder')
        model.freeze_patch_encoder()
        agg_lr = args.learning_rate * args.aggregator_lr_factor
        best_f1 = 0.0

        best_f1, epoch_offset = run_phase(
            phase_name='aggregator',
            model=model,
            train_loader=seq_train_loader,
            val_loader=seq_val_loader,
            test_loader=seq_test_loader,
            unpack_fn=_unpack_sequence_batch,
            criterion=criterion, device=device,
            num_epochs=args.aggregator_epochs,
            learning_rate=agg_lr,
            weight_decay=args.weight_decay,
            warmup_fraction=args.warmup_fraction,
            min_lr_ratio=args.min_lr_ratio,
            grad_accum=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every_n_steps,
            eval_threshold=args.eval_threshold,
            ckpt_dir=ckpt_dir, csv_log=csv_log, use_bf16=args.bf16,
            epoch_offset=epoch_offset, best_val_f1=best_f1,
            n_extra_classes=n_extra_classes,
        )
        logger.info("  Aggregator phase done.")
        load_best_or_last(ckpt_dir, 'aggregator', model)

    csv_log.close()

    # =====================================================================
    # Post-training temperature calibration (skip in merge_val mode)
    # =====================================================================
    if use_bacteria:
        hosts = np.append(hosts, 'bacterial_fragment')

    if not args.merge_val:
        logger.info("\n--- Temperature calibration on validation set ---")
        if not load_best_or_last(ckpt_dir, 'aggregator', model):
            logger.warning("  No sequence-level checkpoint found, "
                           "calibrating with current weights")

        run_calibration(model, seq_val_loader, device, _unpack_sequence_batch,
                        run_dir, hosts, model_config, args.eval_threshold,
                        args.min_val_precision, args.min_val_support)
    else:
        logger.info("\n--- Skipping calibration (--merge_val mode) ---")
        logger.info("  Copy calibration.json from your tuning run into this "
                    "run directory to use with predict.py")

    logger.info(f"\nTraining complete.")
    logger.info(f"Outputs: {os.path.abspath(run_dir)}")


if __name__ == '__main__':
    main()
