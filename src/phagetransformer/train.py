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
import gzip
import json
import logging
import os
import random
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
    load_phage_host,
    BacterialGenomeStore, BacterialPatchDataset, BacterialSequenceDataset,
    RandomPatchDataset, EvalPatchDataset, patch_collate_fn,
    PatchSequenceDataset, sequence_collate_fn,
)
from .utils import (
    compute_metrics, get_cosine_schedule_with_warmup,
    save_checkpoint, load_component, load_best_or_last,
    CSVLogger, run_calibration, run_ood_only_calibration,
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
# Optimizer parameter grouping
# ---------------------------------------------------------------------------

def build_param_groups(model: nn.Module, weight_decay: float):
    """Split trainable params into weight-decay and no-decay groups.

    Matrix-like parameters (Linear/Conv weights) are decayed; everything else
    is excluded from weight decay.  The no-decay set covers biases,
    LayerNorm/BatchNorm affine params, the learnable attention query
    parameters, and embedding tables.

    Two subtleties the routing handles explicitly:

    * The query parameters are declared with shape ``(1, 1, dim)``, so a plain
      ``ndim`` test would misclassify them as matrices.  We route on
      *effective* dimensionality — axes with size > 1 — so a ``(1, 1, dim)``
      query counts as a vector and is excluded from decay.
    * Embedding tables are 2-D ``(vocab, dim)`` and are shape-indistinguishable
      from Linear weights, so they are identified by module type and forced
      into the no-decay group.
    """
    # Parameter ids belonging to nn.Embedding modules — excluded from decay.
    embedding_param_ids = {
        id(p)
        for m in model.modules() if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        effective_ndim = sum(1 for s in p.shape if s > 1)
        if effective_ndim >= 2 and id(p) not in embedding_param_ids:
            decay.append(p)
        else:
            no_decay.append(p)

    n_decay = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    logger.info(f"  param groups: decay={len(decay)} tensors "
                f"({n_decay:,} params), "
                f"no_decay={len(no_decay)} tensors ({n_no_decay:,} params)")

    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


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
    resample_dataset=None,
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  trainable parameters: {trainable:,}")

    param_groups = build_param_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate,
                                  weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs // grad_accum
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio)

    scaler = torch.amp.GradScaler('cuda') if use_bf16 else None

    logger.info(f"  steps/epoch={len(train_loader)}  total_steps={total_steps}  "
                f"warmup={warmup_steps}")

    for ep in range(num_epochs):
        if resample_dataset is not None and hasattr(resample_dataset, 'resample_index'):
            resample_dataset.resample_index()
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
    g.add_argument('--bio_codon_init', action='store_true',
                   help='Initialize codon embeddings from biochemical '
                        'properties (default: random normal init)')

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
    g.add_argument('--stride', type=int, default=2400,
                   help='Tiling stride (nt) between patch starts, used '
                        'identically for aggregator training and for '
                        'eval/inference. Default patch_nt_len // 2 if unset.')
    g.add_argument('--seq_drop_rate', type=float, default=0.6,
                   help='Aggregator phase train: max fraction of sequence to truncate')
    g.add_argument('--patch_drop_rate', type=float, default=0.1,
                   help='Aggregator phase train: max fraction of patches to drop after tiling')
    g.add_argument('--min_seq_repeats', type=float, default=1.0,
                   help='Aggregator phase: sequence repeats per epoch for common-class sequences')
    g.add_argument('--max_seq_repeats', type=float, default=3.0,
                   help='Aggregator phase: sequence repeats per epoch for rarest-class sequences')
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
                   help='DEPRECATED (no longer drives blocking; kept for CLI '
                        'compatibility). Class blocking has been replaced by '
                        'the OOD rejection detector — see --fit_ood.')
    g.add_argument('--min_val_support', type=int, default=1,
                   help='DEPRECATED (was: min predictions for block '
                        'eligibility). Unused now that blocking is replaced '
                        'by OOD rejection.')
    g.add_argument('--block_max_fdr', type=float, default=0.5,
                   help='DEPRECATED (was: block classes whose empirical FDR '
                        'exceeded this). Unused now that blocking is replaced '
                        'by OOD rejection.')
    # -- OOD detector (class-conditional Mahalanobis) ----------------------
    g.add_argument('--fit_ood', dest='fit_ood', action='store_true',
                   default=True,
                   help='Fit the class-conditional Mahalanobis OOD detector at '
                        'calibration time (replaces class blocking). Default on.')
    g.add_argument('--no_fit_ood', dest='fit_ood', action='store_false',
                   help='Disable fitting the OOD detector.')
    g.add_argument('--ood_target_reject_rate', type=float, default=0.05,
                   help='Target in-distribution rejection rate used to set the '
                        'OOD reject threshold on validation phages.')
    g.add_argument('--ood_shrinkage_lambda', type=float, default=200.0,
                   help='Hierarchical mean-shrinkage strength (in equivalent '
                        'patches) pulling rare-genus means toward the family '
                        'centroid, and families toward the grand mean.')
    g.add_argument('--ood_score_agg', type=str, default='mean',
                   choices=['mean', 'max'],
                   help='Aggregate per-patch Mahalanobis distance into a '
                        'sequence score via mean (stable) or max (sensitive).')
    g.add_argument('--ood_tau', type=float, default=0.95,
                   help='Per-patch typicality threshold (ID-CDF quantile) above '
                        'which a patch counts toward the OOD-fraction metric.')
    g.add_argument('--ood_n0', type=float, default=0.0,
                   help='Patch-count floor constant. 0 (default) disables the '
                        'floor so short genomes (e.g. microviruses) are not '
                        'penalised for low patch count; the log1p(n) logistic '
                        'feature still captures any real count effect. Set >0 '
                        'to multiply reliability by n/(n+n0).')
    g.add_argument('--reliability_features', type=str, default='full',
                   choices=['full', 'ood_count', 'ood_only'],
                   help="Which features feed the reliability logistic. 'full': "
                        "OOD + host confidence (top1, margin). 'ood_only': OOD "
                        "signals alone (no prediction-derived features), so "
                        "reliability reflects out-of-distribution-ness only. "
                        "'ood_count': OOD + patch count, no host confidence.")
    g.add_argument('--ood_normalize', action='store_true',
                   help='L2-normalize embeddings before fitting and scoring the '
                        'OOD detectors (whitened distance on the unit sphere / '
                        'directional). Applied consistently at fit and '
                        'inference via the stored sidecar. Off by default '
                        '(magnitude-sensitive Mahalanobis).')
    g.add_argument('--merge_ood_from', type=str, default=None,
                   help='Path to a validation-model calibration.json. When set, '
                        'the OOD/reliability block is fit here (on the discard '
                        'set) and the logit-space calibration (temperature, FDR '
                        'thresholds) is spliced in from that file instead of '
                        'being recomputed. Hard-fails if hosts/model_config/'
                        'stride differ. Works with --merge_val and '
                        '--calibrate_only.')
    g.add_argument('--validation_tani_column', type=str, default=None,
                   help='Column in val.csv with each phage\'s max tANI to the '
                        'training set (fraction 0-1). If set, temperature and '
                        'FDR thresholds are calibrated on the distant subset.')
    g.add_argument('--tani_cutoff', type=float, default=0.5,
                   help='Calibrate on val phages with tANI <= this '
                        '(requires --validation_tani_column)')

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
    g.add_argument('--host_genome_dir', type=str, default=None,
                   help='Directory with host_genome_manifest.tsv '
                        '(required for bacterial_fragment class in encoder '
                        'and aggregator phases)')
    g.add_argument('--bacterial_mask_regions', type=str, default=None,
                   help='Path to phage_hit_regions.tsv (written by '
                        'compute_phage_hit_regions.py). When set, each '
                        'bacterial genome has its phage-matched regions '
                        'excised before train/val splitting — preventing '
                        'leakage from phage-derived sequence content into '
                        'the bacterial_fragment class. Excision (not '
                        'N-masking) avoids introducing a learnable N-stretch '
                        'artifact.')
    g.add_argument('--one_genome_per_genus', action='store_true',
                   help='Keep only one genome per genus from the manifest')
    g.add_argument('--genus_alpha', type=float, default=0.25,
                   help='Power-law exponent for genus sampling balance. '
                        '0=genus-uniform, 1=species-proportional, '
                        'default 0.25 gives mild diversity bonus')
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

    logger.info("Loading data …")
    (train_seqs, train_labels, val_seqs, val_labels,
     test_seqs, test_labels, hosts, val_tani) = load_phage_host(
        ds_path, val_tani_col=args.validation_tani_column)
    num_classes = len(hosts)

    if args.merge_val:
        logger.info("  --merge_val: merging train + val into one training set")
        train_seqs = train_seqs + val_seqs
        train_labels = np.concatenate([train_labels, val_labels], axis=0)
        val_seqs, val_labels = [], np.zeros((0, num_classes), dtype=np.float32)
        val_tani = None

    logger.info(f"  train={len(train_seqs)}  val={len(val_seqs)}  "
                f"test={len(test_seqs)}  classes={num_classes}")

    # ---- bacterial genomes ------------------------------------------------
    genome_store = None
    if args.host_genome_dir:
        logger.info("Loading bacterial genomes …")
        genome_store = BacterialGenomeStore(
            args.host_genome_dir,
            val_frac=0.0 if args.merge_val else 0.2,
            seed=args.seed,
            one_per_genus=args.one_genome_per_genus,
            genus_alpha=args.genus_alpha,
            mask_regions_tsv=args.bacterial_mask_regions,
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
        # Warn about host labels (genera) with no bacterial genome data.
        available_genera = {sp.split()[0] for sp in genome_store.species_list}
        missing = sorted(g for g in genus_to_idx if g not in available_genera)
        if missing:
            shown = ', '.join(missing[:10])
            logger.warning(
                f"  {len(missing)}/{len(genus_to_idx)} host genera have no "
                f"bacterial genome data: {shown}"
                f"{' …' if len(missing) > 10 else ''}")

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
        bio_codon_init=args.bio_codon_init,
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

    # OOD fit loader: training phages only, deterministic eval tiling (full
    # coverage — no subsampling). Host-only labels (width == n_host) so the
    # gatherer treats every row as phage. Used to fit the ID manifold.
    def build_ood_fit_loader():
        fit_ds = PatchSequenceDataset(
            train_seqs, train_labels, tokenizer,
            patch_nt_len=args.patch_nt_len, max_patches=args.max_patches,
            is_train=False, stride=args.stride)
        return DataLoader(
            fit_ds, batch_size=args.aggregator_batch_size, shuffle=False,
            collate_fn=sequence_collate_fn, **ldr_kw)

    def _read_gz_fasta(path):
        seqs, pid, buf = {}, None, []
        with gzip.open(path, 'rt') as f:
            for line in f:
                if line.startswith('>'):
                    if pid is not None:
                        seqs[pid] = ''.join(buf)
                    pid = line[1:].split()[0]
                    buf = []
                else:
                    buf.append(line.strip())
            if pid is not None:
                seqs[pid] = ''.join(buf)
        return seqs

    def build_discard_calibration(label_width):
        """Assemble the production-model OOD calibration set from the discard
        pools in the dataset folder: discard_id + same-family discard_ood form
        the ID-calibration set (CDF/threshold/logistic); distant discard_ood is
        the OOD evaluation set. Returns None if the files aren't present.

        Correctness for the logistic is family-level, so each calib row carries
        the set of TRUE host families; weights are sampling_weight for
        discard_id and 1.0 for same-family. Labels are zeros (unused: the
        manifold is fit on the train loader, and correctness uses true_families).
        """
        d = ds_path
        paths = {k: os.path.join(d, v) for k, v in dict(
            id_csv='discard_id.csv', ood_csv='discard_ood.csv',
            id_fna='discard_id.fna.gz', ood_fna='discard_ood.fna.gz').items()}
        if not all(os.path.exists(p) for p in paths.values()):
            logger.warning("  Discard pools not found in dataset dir; "
                           "OOD calibration falls back to the val set.")
            return None
        import pandas as pd
        id_meta = pd.read_csv(paths['id_csv'])
        ood_meta = pd.read_csv(paths['ood_csv'])
        id_seq = _read_gz_fasta(paths['id_fna'])
        ood_seq = _read_gz_fasta(paths['ood_fna'])

        def fam_set(field):
            return {';'.join(str(g).split(';')[:4])
                    for g in str(field).split('|') if g and g != 'nan'}

        calib_seqs, calib_fams, calib_w = [], [], []
        for _, r in id_meta.iterrows():
            s = id_seq.get(r['phage_id'])
            if s:
                calib_seqs.append(s)
                calib_fams.append(fam_set(r['trained_hosts']))
                calib_w.append(float(r.get('sampling_weight', 1.0)))
        div = ood_meta['divergence'] if 'divergence' in ood_meta else None
        sf = ood_meta[div == 'same_family'] if div is not None else ood_meta.iloc[0:0]
        for _, r in sf.iterrows():
            s = ood_seq.get(r['phage_id'])
            if s:
                calib_seqs.append(s)
                calib_fams.append(fam_set(r['novel_hosts']))
                calib_w.append(1.0)
        dist = ood_meta[div == 'distant'] if div is not None else ood_meta.iloc[0:0]
        eval_seqs = [ood_seq[r['phage_id']] for _, r in dist.iterrows()
                     if ood_seq.get(r['phage_id'])]

        if not calib_seqs:
            logger.warning("  No usable discard_id/same-family sequences; "
                           "falling back to val set for OOD calibration.")
            return None

        def _loader(seqs):
            labels = np.zeros((len(seqs), label_width), dtype=np.float32)
            ds = PatchSequenceDataset(
                seqs, labels, tokenizer, patch_nt_len=args.patch_nt_len,
                max_patches=args.max_patches, is_train=False, stride=args.stride)
            return DataLoader(ds, batch_size=args.aggregator_batch_size,
                              shuffle=False, collate_fn=sequence_collate_fn,
                              **ldr_kw)

        logger.info(f"  OOD calibration set: {len(id_meta)} discard_id + "
                    f"{len(sf)} same-family = {len(calib_seqs)} ID-calib rows; "
                    f"{len(eval_seqs)} distant for eval.")
        return dict(loader=_loader(calib_seqs),
                    true_families=calib_fams,
                    weights=np.asarray(calib_w, dtype=float),
                    eval_loader=_loader(eval_seqs) if eval_seqs else None)

    # Scrambled val phages: synthetic OOD floor for the diagnostic (structure
    # destroyed, composition preserved). Deterministic given --seed.
    def build_ood_scramble_loader():
        rng = random.Random(args.seed)
        scrambled = []
        for s in val_seqs:
            cl = list(s)
            rng.shuffle(cl)
            scrambled.append(''.join(cl))
        sc_ds = PatchSequenceDataset(
            scrambled, val_labels, tokenizer,
            patch_nt_len=args.patch_nt_len, max_patches=args.max_patches,
            is_train=False, stride=args.stride)
        return DataLoader(
            sc_ds, batch_size=args.aggregator_batch_size, shuffle=False,
            collate_fn=sequence_collate_fn, **ldr_kw)

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
            is_train=False, stride=args.stride,
        )
        if use_bacteria:
            phage_lengths = [len(s) for s in train_seqs]
            n_bacteria_val = max(1, int(len(val_seqs)
                                        * args.aggregator_bacteria_ratio))
            bact_val_ds = BacterialSequenceDataset(
                genome_store, tokenizer, phage_lengths,
                n_samples=n_bacteria_val,
                agg_num_classes=agg_num_classes,
                genus_to_idx=genus_to_idx,
                patch_nt_len=args.patch_nt_len,
                max_patches=args.max_patches, is_train=False,
                stride=args.stride,
            )
            combined_val_ds = ConcatDataset([val_seq_ds, bact_val_ds])
        else:
            combined_val_ds = val_seq_ds
        val_loader = DataLoader(
            combined_val_ds, batch_size=args.aggregator_batch_size,
            shuffle=False, collate_fn=sequence_collate_fn, **ldr_kw)

        _disc = build_discard_calibration(calib_val_labels.shape[1])
        if args.merge_ood_from is not None:
            if _disc is None:
                logger.error("--merge_ood_from requires the discard pools in "
                             "the dataset dir; none found.")
                return
            run_ood_only_calibration(
                model, build_ood_fit_loader(), _disc['loader'], device,
                _unpack_sequence_batch, run_dir, calib_hosts, model_config,
                args.eval_threshold, n_extra_classes=n_extra_classes,
                stride=args.stride, merge_ood_from=args.merge_ood_from,
                ood_calib_true_families=_disc['true_families'],
                ood_calib_weights=_disc['weights'],
                ood_eval_loader=_disc['eval_loader'],
                ood_target_reject_rate=args.ood_target_reject_rate,
                ood_shrinkage_lambda=args.ood_shrinkage_lambda,
                ood_score_agg=args.ood_score_agg,
                scramble_loader=build_ood_scramble_loader(),
                ood_tau=args.ood_tau, ood_n0=args.ood_n0,
                ood_normalize=args.ood_normalize)
            logger.info("Calibration complete (OOD fit + merged logit "
                        "calibration).")
            return
        run_calibration(model, val_loader, device, _unpack_sequence_batch,
                        run_dir, calib_hosts, model_config, args.eval_threshold,
                        args.min_val_precision, args.min_val_support,
                        stride=args.stride,
                        n_extra_classes=n_extra_classes,
                        tani=val_tani, tani_cutoff=args.tani_cutoff,
                        block_max_fdr=args.block_max_fdr,
                        fit_loader=build_ood_fit_loader(), fit_ood=args.fit_ood,
                        ood_target_reject_rate=args.ood_target_reject_rate,
                        ood_shrinkage_lambda=args.ood_shrinkage_lambda,
                        ood_score_agg=args.ood_score_agg,
                        scramble_loader=build_ood_scramble_loader(),
                        ood_tau=args.ood_tau, ood_n0=args.ood_n0,
                        reliability_features=args.reliability_features,
                        ood_normalize=args.ood_normalize,
                        ood_calib_loader=(_disc['loader'] if _disc else None),
                        ood_calib_true_families=(_disc['true_families'] if _disc else None),
                        ood_calib_weights=(_disc['weights'] if _disc else None),
                        ood_eval_loader=(_disc['eval_loader'] if _disc else None))
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
            patch_nt_len=args.patch_nt_len, stride=args.stride,
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
        is_train=True, stride=args.stride,
        seq_drop_rate=args.seq_drop_rate,
        patch_drop_rate=args.patch_drop_rate,
        scramble_rate=args.seq_scramble_rate,
        min_seq_repeats=args.min_seq_repeats,
        max_seq_repeats=args.max_seq_repeats,
    )
    if use_bacteria:
        bact_train_ds = BacterialSequenceDataset(
            genome_store, tokenizer, phage_lengths,
            n_samples=n_bacteria_train,
            agg_num_classes=agg_num_classes,
            genus_to_idx=genus_to_idx,
            patch_nt_len=args.patch_nt_len,
            max_patches=args.max_patches,
            is_train=True, stride=args.stride,
            scramble_rate=args.seq_scramble_rate,
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
            is_train=False, stride=args.stride,
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
                stride=args.stride,
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
        is_train=False, stride=args.stride,
    )
    if use_bacteria and not args.merge_val:
        n_bacteria_test = max(1, int(len(test_seqs)
                                     * args.aggregator_bacteria_ratio))
        bact_test_ds = BacterialSequenceDataset(
            genome_store, tokenizer, phage_lengths,
            n_samples=n_bacteria_test,
            agg_num_classes=agg_num_classes,
            genus_to_idx=genus_to_idx,
            patch_nt_len=args.patch_nt_len,
            max_patches=args.max_patches, is_train=False,
            stride=args.stride,
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
            resample_dataset=train_seq_ds,
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

        _disc = build_discard_calibration(agg_val_labels.shape[1])
        run_calibration(model, seq_val_loader, device, _unpack_sequence_batch,
                        run_dir, hosts, model_config, args.eval_threshold,
                        args.min_val_precision, args.min_val_support,
                        stride=args.stride,
                        n_extra_classes=n_extra_classes,
                        tani=val_tani, tani_cutoff=args.tani_cutoff,
                        block_max_fdr=args.block_max_fdr,
                        fit_loader=build_ood_fit_loader(), fit_ood=args.fit_ood,
                        ood_target_reject_rate=args.ood_target_reject_rate,
                        ood_shrinkage_lambda=args.ood_shrinkage_lambda,
                        ood_score_agg=args.ood_score_agg,
                        scramble_loader=build_ood_scramble_loader(),
                        ood_tau=args.ood_tau, ood_n0=args.ood_n0,
                        reliability_features=args.reliability_features,
                        ood_normalize=args.ood_normalize,
                        ood_calib_loader=(_disc['loader'] if _disc else None),
                        ood_calib_true_families=(_disc['true_families'] if _disc else None),
                        ood_calib_weights=(_disc['weights'] if _disc else None),
                        ood_eval_loader=(_disc['eval_loader'] if _disc else None))
    else:
        logger.info("\n--- merge_val mode: fitting OOD calibration on discards ---")
        _disc = build_discard_calibration(agg_train_labels.shape[1])
        if _disc is None:
            logger.warning("  No discard pools found; skipping calibration. "
                           "Copy calibration.json from your validation run "
                           "into this run directory to use with predict.py")
        else:
            if args.merge_ood_from is None:
                logger.warning("  --merge_ood_from not set: the merged "
                               "calibration will have no temperature/FDR "
                               "(OOD-only). Pass a validation calibration.json "
                               "to splice those in.")
            run_ood_only_calibration(
                model, build_ood_fit_loader(), _disc['loader'], device,
                _unpack_sequence_batch, run_dir, hosts, model_config,
                args.eval_threshold, n_extra_classes=n_extra_classes,
                stride=args.stride, merge_ood_from=args.merge_ood_from,
                ood_calib_true_families=_disc['true_families'],
                ood_calib_weights=_disc['weights'],
                ood_eval_loader=_disc['eval_loader'],
                ood_target_reject_rate=args.ood_target_reject_rate,
                ood_shrinkage_lambda=args.ood_shrinkage_lambda,
                ood_score_agg=args.ood_score_agg,
                scramble_loader=build_ood_scramble_loader(),
                ood_tau=args.ood_tau, ood_n0=args.ood_n0,
                ood_normalize=args.ood_normalize)

    logger.info(f"\nTraining complete.")
    logger.info(f"Outputs: {os.path.abspath(run_dir)}")


if __name__ == '__main__':
    main()
