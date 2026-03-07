from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_util
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_length_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    """1.0 for valid, 0.0 for padding.  Shape (B, max_len)."""
    return (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_NUC_TO_IDX = np.full(128, 255, dtype=np.uint8)
_NUC_TO_IDX[ord('A')] = 0
_NUC_TO_IDX[ord('C')] = 1
_NUC_TO_IDX[ord('G')] = 2
_NUC_TO_IDX[ord('T')] = 3
_COMP_MAP = np.array([3, 2, 1, 0, 255], dtype=np.uint8)


def _codons_from_numeric(nuc: np.ndarray, frame: int) -> np.ndarray:
    s = nuc[frame:]
    n = (len(s) // 3) * 3
    if n == 0:
        return np.array([], dtype=np.int64)
    s = s[:n].reshape(-1, 3)
    valid = (s < 4).all(axis=1)
    ids = s[:, 0].astype(np.int64) * 16 + s[:, 1].astype(np.int64) * 4 + s[:, 2].astype(np.int64) + 2
    ids[~valid] = 1
    return ids


class CodonTokenizer:
    PAD = 0; UNK = 1; VOCAB_SIZE = 66

    def __init__(self):
        nucs = ['A', 'C', 'G', 'T']
        self.codons = [a + b + c for a in nucs for b in nucs for c in nucs]
        self.pad_token, self.unk_token = '<PAD>', '<UNK>'
        self.vocab = {self.pad_token: 0, self.unk_token: 1}
        for i, c in enumerate(self.codons, 2):
            self.vocab[c] = i
        self.vocab_size = self.VOCAB_SIZE
        self.pad_token_id = self.PAD
        self.unk_token_id = self.UNK

    def tokenize(self, seq: str) -> torch.Tensor:
        raw = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
        nuc = _NUC_TO_IDX[np.minimum(raw, 127)]
        rev = _COMP_MAP[np.minimum(nuc, 4)][::-1].copy()
        frames = [_codons_from_numeric(nuc, o) for o in range(3)] + \
                 [_codons_from_numeric(rev, o) for o in range(3)]
        ml = max(len(f) for f in frames)
        out = np.zeros((6, ml), dtype=np.int64)
        for i, f in enumerate(frames):
            out[i, :len(f)] = f
        return torch.from_numpy(out)

    def batch_tokenize(self, sequences: list) -> Tuple[torch.Tensor, torch.Tensor]:
        toks = [self.tokenize(s) for s in sequences]
        ml = max(t.size(1) for t in toks)
        bt, bm = [], []
        for t in toks:
            if t.size(1) < ml:
                t = F.pad(t, (0, ml - t.size(1)), value=0)
            bm.append((t == 0).all(dim=0).float())
            bt.append(t)
        return torch.stack(bt), torch.stack(bm)


# ---------------------------------------------------------------------------
# CNN backbone  (no padding awareness — patches are always dense)
# ---------------------------------------------------------------------------

class PerFrameCNN(nn.Module):
    """Conv1d tower with 2× downsampling per layer.

    The first layer uses stride=1 followed by MaxPool1d(2) to preserve
    fine-grained features before downsampling.  Subsequent layers use
    stride=2 convolutions.

    Expects dense (unpadded) input — no length tracking or masking.

    Parameters
    ----------
    vocab_size, embedding_dim, hidden_dim, output_dim : int
    kernel_sizes : list[int]   default ``[9, 9, 7, 7, 5, 5]`` (64x compression)
    dropout : float
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 10,
                 hidden_dim: int = 128, output_dim: int = 265,
                 kernel_sizes: Optional[list] = None, dropout: float = 0.1):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 9, 7, 7, 5, 5]
        self.kernel_sizes = list(kernel_sizes)
        n_layers = len(self.kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        convs, norms = [], []
        for i, k in enumerate(self.kernel_sizes):
            ic = embedding_dim if i == 0 else hidden_dim
            oc = output_dim if i == n_layers - 1 else hidden_dim
            stride = 1 if i == 0 else 2
            convs.append(nn.Conv1d(ic, oc, kernel_size=k, stride=stride,
                                   padding=k // 2))
            norms.append(nn.BatchNorm1d(oc))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.pool0 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    @property
    def compression_factor(self) -> int:
        f = 1
        for _ in self.kernel_sizes:
            f *= 2
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L) token-id tensor  (dense, no padding)
        Returns : (B, D, L') feature maps
        """
        x = self.embedding(x).transpose(1, 2)              # (B, E, L)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = self.dropout(F.gelu(norm(conv(x))))
            if i == 0:
                x = self.pool0(x)
        return x


# ---------------------------------------------------------------------------
# Cross-frame attention
# ---------------------------------------------------------------------------

class CrossFrameAttention(nn.Module):
    """Attend across the 6 reading frames per position, mean-pool to one seq."""

    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                return_weights: bool = False):
        """x: (B, 6, L', D) -> (B, L', D) or ((B, L', D), (B, L', 6))"""
        B, F_, L, D = x.shape
        xf = x.permute(0, 2, 1, 3).reshape(B * L, F_, D)
        q = self.q_proj(xf).reshape(B * L, F_, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(xf).reshape(B * L, F_, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(xf).reshape(B * L, F_, self.num_heads, self.head_dim).transpose(1, 2)
        a = self.dropout(F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1))
        out = torch.matmul(a, v).transpose(1, 2).reshape(B * L, F_, D)
        merged = self.norm(self.out_proj(out).mean(dim=1).reshape(B, L, D))
        if return_weights:
            w = a.mean(dim=1).mean(dim=1).reshape(B, L, F_)
            return merged, w
        return merged

# ---------------------------------------------------------------------------
# Flash-attention transformer (no positional bias)
# ---------------------------------------------------------------------------

class FlashAttention(nn.Module):
    """Multi-head SDPA attention. mask is optional (None = no padding)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : (B, L, D)
        mask : (B, L) bool, True = padded.  None = no masking.
        """
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_mask = None
        if mask is not None:
            attn_mask = torch.where(
                mask[:, None, None, :],
                torch.tensor(float('-inf'), device=x.device, dtype=q.dtype),
                torch.tensor(0.0, device=x.device, dtype=q.dtype),
            )

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=False,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        h = 256 * ((int(dim * 8 / 3) + 255) // 256)
        self.w1 = nn.Linear(dim, h, bias=False)
        self.w2 = nn.Linear(h, dim, bias=False)
        self.w3 = nn.Linear(dim, h, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm block: LayerNorm -> FlashAttention -> SwiGLU."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout)
        self.ffn = SwiGLU(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks, optional gradient checkpointing."""

    def __init__(self, d_model, n_heads, n_layers, dropout=0.1,
                 gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, src, src_key_padding_mask=None):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                src = checkpoint_util.checkpoint(
                    layer, src, src_key_padding_mask, use_reentrant=False,
                )
            else:
                src = layer(src, src_key_padding_mask)
        return src


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class QueryAttentionPooling(nn.Module):
    """Learnable query attends over the sequence -> single pooled vector."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, padding_mask=None):
        B = h.size(0)
        K, V = self.key_proj(h), self.value_proj(h)
        s = torch.matmul(self.query.expand(B, -1, -1), K.transpose(-2, -1)).squeeze(1) * self.scale
        if padding_mask is not None:
            s = s.masked_fill(padding_mask, -float('inf'))
        w = self.dropout(F.softmax(s, dim=-1))              # (B, L)
        pooled = self.norm(torch.matmul(w.unsqueeze(1), V).squeeze(1))
        return pooled, w


# ---------------------------------------------------------------------------
# Stage 1 — Patch encoder
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """(B, 6, L) tokenised patch  ->  (B, D) embedding.

    Patches are dense (no padding), so the CNN, transformer, and pooler
    are all called without masks.  Reverse-complement frames (3-5) are
    flipped along the sequence axis after CNN to align with forward frames.

    A parallel :class:`FrameStatsBranch` operates on the same CNN feature
    maps through a lightweight independent pipeline.  Its output is added
    to the main-path embedding so downstream components receive a single
    fused (B, D) vector.
    """

    def __init__(self, vocab_size=66, cnn_embed_dim=10, cnn_hidden_dim=128,
                 transformer_dim=512, num_transformer_layers=4, num_heads=8,
                 dropout=0.1, gradient_checkpointing=False,
                 cnn_kernel_sizes=None,
                 frame_stats_channels=64, frame_stats_kernel_size=3):
        super().__init__()
        self.frame_cnn = PerFrameCNN(
            vocab_size, cnn_embed_dim, cnn_hidden_dim, transformer_dim,
            kernel_sizes=cnn_kernel_sizes, dropout=dropout,
        )
        self.cross_frame_attn = CrossFrameAttention(transformer_dim, num_heads, dropout)
        self.transformer = TransformerEncoder(
            transformer_dim, num_heads, num_transformer_layers,
            dropout=dropout, gradient_checkpointing=gradient_checkpointing,
        )
        self.norm = nn.LayerNorm(transformer_dim)
        self.pooler = QueryAttentionPooling(transformer_dim, dropout)

        self.frame_stats_branch = FrameStatsBranch(
            input_dim=transformer_dim,
            output_dim=transformer_dim,
            n_channels=frame_stats_channels,
            kernel_size=frame_stats_kernel_size,
            dropout=dropout,
        )

    @property
    def output_dim(self):
        return self.norm.normalized_shape[0]

    def forward(self, frame_inputs: torch.Tensor,
                return_weights: bool = False):
        """Encode tokenised patches.

        Parameters
        ----------
        frame_inputs : (B, 6, L)
        return_weights : bool
            If True, return ``(embedding, weights_dict)`` with all
            attention weight layers.  If False, return just the
            embedding tensor.

        Returns
        -------
        embedding : (B, D)
            Always returned.
        weights : dict  (only when ``return_weights=True``)
            frame_w         : (B, L', 6)  — main cross-frame attention
            pool_w          : (B, L')     — main position pooling
            branch_frame_w  : (B, L', 6)  — branch cross-frame attention
            branch_pool_w   : (B, L')     — branch position pooling
        """
        B, NF, L = frame_inputs.shape
        flat = frame_inputs.reshape(B * NF, L)
        feats = self.frame_cnn(flat)
        D, Lp = feats.size(1), feats.size(2)
        feats = feats.reshape(B, NF, D, Lp)
        feats[:, 3] = torch.flip(feats[:, 3], dims=[2])
        feats[:, 4] = torch.flip(feats[:, 4], dims=[2])
        feats[:, 5] = torch.flip(feats[:, 5], dims=[2])

        feats_perm = feats.permute(0, 1, 3, 2)           # (B, 6, L', D)

        if return_weights:
            x, frame_w = self.cross_frame_attn(
                feats_perm, return_weights=True)
        else:
            x = self.cross_frame_attn(feats_perm)

        x = self.norm(self.transformer(x))
        pooled, pool_w = self.pooler(x)                    # (B, D), (B, L')

        if return_weights:
            branch_emb, branch_frame_w, branch_pool_w = \
                self.frame_stats_branch(feats, return_weights=True)
            pooled = pooled + branch_emb
            return pooled, {
                'frame_w': frame_w,
                'pool_w': pool_w,
                'branch_frame_w': branch_frame_w,
                'branch_pool_w': branch_pool_w,
            }
        else:
            pooled = pooled + self.frame_stats_branch(feats)
            return pooled


class PatchClassifier(nn.Module):
    """Wraps PatchEncoder + temporary classification head for encoder-phase training.

    The head is discarded after the encoder phase — only the encoder weights carry over.
    """

    def __init__(self, patch_encoder: PatchEncoder, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_encoder = patch_encoder
        dim = patch_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frame_inputs: torch.Tensor) -> torch.Tensor:
        """frame_inputs: (B, 6, L) -> (B, num_classes) logits"""
        return self.classifier(self.patch_encoder(frame_inputs))


class FrameStatsBranch(nn.Module):
    """Parallel branch on per-frame CNN features → embedding.

    Shared Conv1d per frame → CrossFrameAttention → QueryAttentionPooling
    → linear projection.  Frame-symmetric by construction.

    Operates on the same CNN feature maps as the main encoder path but
    through a lightweight, independent pipeline that can specialise on
    coding-structure signals.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 n_channels: int = 64, kernel_size: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.frame_cnn = nn.Sequential(
            nn.Conv1d(input_dim, n_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
            nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
            nn.Conv1d(n_channels, n_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
        )
        self.cross_frame_attn = CrossFrameAttention(
            n_channels, n_heads, dropout)
        self.norm = nn.LayerNorm(n_channels)
        self.pooler = QueryAttentionPooling(n_channels, dropout)
        self.proj = nn.Linear(n_channels, output_dim)

    def forward(self, frame_feats: torch.Tensor,
                return_weights: bool = False):
        """frame_feats: (B, 6, D, L') → (B, output_dim)
        or (embedding, frame_w, pool_w) when return_weights=True."""
        B, F_, D, L = frame_feats.shape
        x = frame_feats.reshape(B * F_, D, L)              # (B*6, D, L')
        x = self.frame_cnn(x)                               # (B*6, C, L')
        C, Lp = x.size(1), x.size(2)
        x = x.reshape(B, F_, C, Lp).permute(0, 1, 3, 2)   # (B, 6, L', C)

        x, frame_w = self.cross_frame_attn(
            x, return_weights=True)                         # (B, L', C), (B, L', 6)
        x = self.norm(x)
        pooled, pool_w = self.pooler(x)                     # (B, C), (B, L')
        emb = self.proj(pooled)                              # (B, D)

        if return_weights:
            return emb, frame_w, pool_w
        return emb


class SequenceAggregator(nn.Module):
    """(B, N_patches, D) patch embeddings  ->  (B, num_classes) logits.

    This level *does* use padding masks because different sequences produce
    different numbers of patches.
    """

    def __init__(self, d_model, num_classes, n_heads=8, n_layers=3,
                 dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.transformer = TransformerEncoder(
            d_model, n_heads, n_layers, dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pooler = QueryAttentionPooling(d_model, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, embs, patch_counts):
        B, N, D = embs.shape
        device = embs.device
        x = embs
        pad = ~(_make_length_mask(patch_counts.float(), N, device).bool())
        x = x * (~pad).float().unsqueeze(-1)
        x = self.norm(self.transformer(x, src_key_padding_mask=pad))
        pooled, _ = self.pooler(x, padding_mask=pad)
        return self.classifier(pooled)

    @torch.no_grad()
    def get_pooling_weights(self, embs, patch_counts):
        """Run aggregator and return pooling attention weights.

        Returns : (B, N) — importance of each patch for the prediction.
        """
        B, N, D = embs.shape
        device = embs.device
        pad = ~(_make_length_mask(patch_counts.float(), N, device).bool())
        x = embs * (~pad).float().unsqueeze(-1)
        x = self.norm(self.transformer(x, src_key_padding_mask=pad))
        _, w = self.pooler(x, padding_mask=pad)
        return w                                             # (B, N)


# ---------------------------------------------------------------------------
# Full hierarchical model
# ---------------------------------------------------------------------------

class HierarchicalDNAClassifier(nn.Module):
    def __init__(self, num_classes: int, vocab_size: int = 66,
                 cnn_embed_dim: int = 10, cnn_hidden_dim: int = 256,
                 transformer_dim: int = 512, num_transformer_layers: int = 4,
                 num_heads: int = 8, agg_layers: int = 3, agg_heads: int = 8,
                 patches_per_forward: int = 48, frame_stats_channels: int = 64,
                 frame_stats_kernel_size: int = 3, patch_nt_len: int = 3072,
                 dropout: float = 0.1, gradient_checkpointing: bool = False,
                 cnn_kernel_sizes: Optional[list] = None):
        super().__init__()
        self.patches_per_forward = patches_per_forward
        self.patch_encoder = PatchEncoder(
            vocab_size=vocab_size, cnn_embed_dim=cnn_embed_dim,
            cnn_hidden_dim=cnn_hidden_dim, transformer_dim=transformer_dim,
            num_transformer_layers=num_transformer_layers, num_heads=num_heads,
            dropout=dropout, gradient_checkpointing=gradient_checkpointing,
            cnn_kernel_sizes=cnn_kernel_sizes,
            frame_stats_channels=frame_stats_channels,
            frame_stats_kernel_size=frame_stats_kernel_size,
        )
        self.aggregator = SequenceAggregator(
            d_model=transformer_dim, num_classes=num_classes,
            n_heads=agg_heads, n_layers=agg_layers, dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, patches: torch.Tensor, patch_counts: torch.Tensor) -> torch.Tensor:
        """
        patches      : (B, max_N, 6, codon_L)
        patch_counts : (B,)
        Returns      : (B, num_classes) logits
        """
        B, maxN, F_, cL = patches.shape
        device = patches.device
        D = self.patch_encoder.output_dim

        idx = torch.arange(maxN, device=device).unsqueeze(0)
        valid = idx < patch_counts.unsqueeze(1)
        valid_flat = valid.reshape(-1)
        flat = patches.reshape(B * maxN, F_, cL)
        valid_patches = flat[valid_flat]

        encoder_frozen = not any(p.requires_grad
                                 for p in self.patch_encoder.parameters())
        ctx = torch.no_grad() if encoder_frozen else nullcontext()

        chunk = self.patches_per_forward
        parts = []
        for i in range(0, valid_patches.size(0), chunk):
            blk = valid_patches[i:i + chunk]
            with ctx:
                if (self.patch_encoder.transformer.gradient_checkpointing
                        and self.training and not encoder_frozen):
                    emb = checkpoint_util.checkpoint(
                        self.patch_encoder, blk, use_reentrant=False)
                else:
                    emb = self.patch_encoder(blk)
            parts.append(emb)
        valid_embs = torch.cat(parts, 0) if parts else torch.zeros(0, D, device=device)

        per_seq = torch.split(valid_embs, patch_counts.tolist())
        embs = torch.nn.utils.rnn.pad_sequence(per_seq, batch_first=True)

        return self.aggregator(embs, patch_counts)

    @torch.no_grad()
    def annotate(self, patches: torch.Tensor,
                 patch_counts: torch.Tensor) -> list:
        """Extract all attention weight layers for annotation.

        Returns list of dicts (one per sequence), each containing:
            frame_w          : (n_patches, L', 6)  — main cross-frame attention
            pool_w           : (n_patches, L')     — main position pooling
            branch_frame_w   : (n_patches, L', 6)  — branch cross-frame attention
            branch_pool_w    : (n_patches, L')     — branch position pooling
            agg_w            : (n_patches,)        — aggregator patch-level pooling
        """
        B, maxN, F_, cL = patches.shape
        device = patches.device

        valid = torch.arange(maxN, device=device).unsqueeze(0) < patch_counts.unsqueeze(1)
        flat = patches.reshape(B * maxN, F_, cL)
        valid_patches = flat[valid.reshape(-1)]

        chunk = self.patches_per_forward
        emb_parts = []
        weight_parts = {}
        for i in range(0, valid_patches.size(0), chunk):
            blk = valid_patches[i:i + chunk]
            emb, weights = self.patch_encoder(blk, return_weights=True)
            emb_parts.append(emb)
            for key, val in weights.items():
                weight_parts.setdefault(key, []).append(val)

        if not emb_parts:
            empty = {'frame_w': np.zeros((0, 6)),
                     'pool_w': np.zeros(0),
                     'branch_frame_w': np.zeros((0, 6)),
                     'branch_pool_w': np.zeros(0),
                     'agg_w': np.zeros(0)}
            return [empty for _ in range(B)]

        all_embs = torch.cat(emb_parts, 0)
        merged_weights = {k: torch.cat(vs, 0) for k, vs in weight_parts.items()}

        # Aggregator weights from embeddings
        per_seq_embs = torch.split(all_embs, patch_counts.tolist())
        embs_padded = torch.nn.utils.rnn.pad_sequence(
            per_seq_embs, batch_first=True)
        agg_w = self.aggregator.get_pooling_weights(
            embs_padded, patch_counts)

        # Split all weight tensors per sequence
        per_seq = {k: torch.split(v, patch_counts.tolist())
                   for k, v in merged_weights.items()}

        results = []
        for i in range(B):
            n = patch_counts[i].item()
            seq_weights = {k: per_seq[k][i].cpu().numpy()
                           for k in merged_weights}
            seq_weights['agg_w'] = agg_w[i, :n].cpu().numpy()
            results.append(seq_weights)

        return results

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.patch_encoder.frame_cnn.embedding.weight.numel()
        return n

    def freeze_patch_encoder(self):
        for p in self.patch_encoder.parameters():
            p.requires_grad = False
        self.patch_encoder.eval()

    def unfreeze_patch_encoder(self):
        for p in self.patch_encoder.parameters():
            p.requires_grad = True
        self.patch_encoder.train()
