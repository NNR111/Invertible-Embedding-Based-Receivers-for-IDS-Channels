
from __future__ import annotations

import torch
import torch.nn as nn

from ids_receiver.config import (
    PAD_VALUE,
    EMBED_DIM,
    GRU_HIDDEN,
    GRU_LAYERS,
    MSG_LEN,
    CC_K,
    DROPOUT,
)


CODE_BITS = 2 * (MSG_LEN + CC_K - 1)


class SymbolFrontEnd(nn.Module):
    def __init__(self, vocab_size: int = 5, emb_dim: int = EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_VALUE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class BiGRUBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.proj = nn.Linear(in_dim, hidden * 2) if in_dim != hidden * 2 else nn.Identity()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        y = self.drop(y)
        return self.norm(y + self.proj(x))


class SequenceEncoder(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM, hidden: int = GRU_HIDDEN, layers: int = GRU_LAYERS):
        super().__init__()
        blocks = []
        cur = in_dim
        for _ in range(layers):
            blocks.append(BiGRUBlock(cur, hidden))
            cur = hidden * 2
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class IndexedAttentionHead(nn.Module):
    def __init__(self, num_queries: int, feat_dim: int, use_parity: bool = True):
        super().__init__()
        self.num_queries = num_queries
        self.use_parity = use_parity
        self.index_emb = nn.Embedding(num_queries, feat_dim)
        self.parity_emb = nn.Embedding(2, feat_dim) if use_parity else None
        self.query_norm = nn.LayerNorm(feat_dim)
        self.out = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(feat_dim, 1),
        )

    def _make_queries(self, device: torch.device) -> torch.Tensor:
        idx = torch.arange(self.num_queries, device=device)
        q = self.index_emb(idx)
        if self.parity_emb is not None:
            q = q + self.parity_emb(idx % 2)
        return self.query_norm(q)

    def forward(self, feats: torch.Tensor, lens: torch.Tensor):
        bsz, time_steps, dim = feats.shape
        mask = torch.arange(time_steps, device=feats.device).unsqueeze(0) < lens.unsqueeze(1)
        q = self._make_queries(feats.device).unsqueeze(0).expand(bsz, -1, -1)

        scores = torch.matmul(q, feats.transpose(1, 2)) / (dim ** 0.5)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, feats)
        logits = self.out(torch.cat([ctx, q], dim=-1)).squeeze(-1)
        return logits, ctx, attn


class NBMHeadNoEmbed(nn.Module):
    def __init__(self, feat_dim: int = GRU_HIDDEN * 2, code_bits: int = CODE_BITS):
        super().__init__()
        self.head = IndexedAttentionHead(num_queries=code_bits, feat_dim=feat_dim, use_parity=True)

    def forward(self, feats: torch.Tensor, lens: torch.Tensor):
        return self.head(feats, lens)


class BiGRUDecoder(nn.Module):
    def __init__(self, code_ctx_dim: int = GRU_HIDDEN * 2, msg_len: int = MSG_LEN):
        super().__init__()
        self.code_logit_proj = nn.Sequential(
            nn.Linear(1, code_ctx_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.fuse = nn.Sequential(
            nn.Linear(code_ctx_dim * 2, code_ctx_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.seq_encoder = SequenceEncoder(
            in_dim=code_ctx_dim,
            hidden=GRU_HIDDEN,
            layers=max(2, GRU_LAYERS),
        )
        self.msg_head = IndexedAttentionHead(
            num_queries=msg_len,
            feat_dim=GRU_HIDDEN * 2,
            use_parity=False,
        )

    def forward(self, code_logits: torch.Tensor, code_ctx: torch.Tensor):
        logit_feat = self.code_logit_proj(code_logits.unsqueeze(-1))
        x = self.fuse(torch.cat([code_ctx, logit_feat], dim=-1))
        h = self.seq_encoder(x)
        lens = torch.full(
            (h.size(0),),
            h.size(1),
            dtype=torch.long,
            device=h.device,
        )
        msg_logits, msg_ctx, msg_attn = self.msg_head(h, lens)
        return msg_logits, msg_ctx, msg_attn


class FullNoEmbedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = SymbolFrontEnd()
        self.backbone = SequenceEncoder(
            in_dim=EMBED_DIM,
            hidden=GRU_HIDDEN,
            layers=max(2, GRU_LAYERS),
        )
        self.nbm = NBMHeadNoEmbed(feat_dim=GRU_HIDDEN * 2, code_bits=CODE_BITS)
        self.decoder = BiGRUDecoder(code_ctx_dim=GRU_HIDDEN * 2, msg_len=MSG_LEN)

    def encode(self, noisy_syms: torch.Tensor) -> torch.Tensor:
        feats = self.front(noisy_syms)
        feats = self.backbone(feats)
        return feats

    def forward_nbm(self, noisy_syms: torch.Tensor, noisy_lens: torch.Tensor):
        feats = self.encode(noisy_syms)
        code_logits, code_ctx, code_attn = self.nbm(feats, noisy_lens)
        return code_logits, code_ctx, code_attn

    def forward_decoder(self, noisy_syms: torch.Tensor, noisy_lens: torch.Tensor):
        code_logits, code_ctx, code_attn = self.forward_nbm(noisy_syms, noisy_lens)
        msg_logits, msg_ctx, msg_attn = self.decoder(code_logits, code_ctx)
        aux = {
            "code_attn": code_attn,
            "msg_attn": msg_attn,
            "msg_ctx": msg_ctx,
        }
        return code_logits, msg_logits, code_ctx, aux
