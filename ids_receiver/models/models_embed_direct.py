from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ids_receiver.config import PAD_VALUE, EMBED_DIM, GRU_HIDDEN, GRU_LAYERS, PROJ_DIM, MSG_LEN, CC_K, DROPOUT


class SiameseBiGRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 5,
        emb_dim: int = EMBED_DIM,
        hidden: int = GRU_HIDDEN,
        layers: int = GRU_LAYERS,
        proj_dim: int = PROJ_DIM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_VALUE)
        self.gru = nn.GRU(
            emb_dim,
            hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=layers,
            dropout=DROPOUT if layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        emb = self.embedding(x)
        feat, _ = self.gru(emb)

        mask = (
            torch.arange(feat.size(1), device=feat.device).unsqueeze(0)
            < lens.unsqueeze(1)
        ).float().unsqueeze(-1)

        pooled = (feat * mask).sum(1) / lens.clamp_min(1).unsqueeze(1)
        z = F.normalize(self.proj(pooled), dim=-1)
        return z, feat


class LocalBitHead(nn.Module):
    """
    Dipertahankan hanya agar compatible dengan checkpoint embedding lama.
    Tidak dipakai saat direct decoder training.
    """
    def __init__(
        self,
        in_dim: int = GRU_HIDDEN * 2,
        code_bits: int = 2 * (MSG_LEN + CC_K - 1),
        hidden: int = GRU_HIDDEN,
    ):
        super().__init__()
        self.code_bits = code_bits
        self.pre = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.index_emb = nn.Embedding(code_bits, hidden * 2)
        self.parity_emb = nn.Embedding(2, hidden * 2)
        self.role_emb = nn.Embedding(2, hidden * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, 1),
        )

    def make_queries(self, device: torch.device):
        idx = torch.arange(self.code_bits, device=device)
        parity = idx % 2
        role = idx % 2
        return self.index_emb(idx) + self.parity_emb(parity) + self.role_emb(role)

    def forward(self, feats: torch.Tensor, lens: torch.Tensor):
        H, _ = self.pre(feats)
        B, T, D = H.shape
        mask = torch.arange(T, device=H.device).unsqueeze(0) < lens.unsqueeze(1)
        q = self.make_queries(H.device).unsqueeze(0).expand(B, -1, -1)
        scores = torch.matmul(q, H.transpose(1, 2)) / (D ** 0.5)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, H)
        logits = self.out(torch.cat([ctx, q], dim=-1)).squeeze(-1)
        return logits, ctx


class DirectBiGRUDecoder(nn.Module):
    """
    Decoder langsung dari encoder feature sequence, tanpa NBM.
    """
    def __init__(
        self,
        in_dim: int = GRU_HIDDEN * 2,
        hidden: int = GRU_HIDDEN,
        layers: int = GRU_LAYERS,
        msg_len: int = MSG_LEN,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.gru = nn.GRU(
            hidden,
            hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=layers,
            dropout=DROPOUT if layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden * 2, msg_len),
        )

    def forward(self, feats: torch.Tensor, lens: torch.Tensor):
        x = self.in_proj(feats)
        H, _ = self.gru(x)

        T = H.size(1)
        mask = torch.arange(T, device=H.device).unsqueeze(0) < lens.unsqueeze(1)

        scores = self.attn(H).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        a = torch.softmax(scores, dim=1)

        pooled = torch.sum(H * a.unsqueeze(-1), dim=1)
        maxp = H.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values

        return self.mlp(torch.cat([pooled, maxp], dim=-1))


class FullDirectEmbedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SiameseBiGRUEncoder()
        self.local_head = LocalBitHead()
        self.decoder = DirectBiGRUDecoder()

    def encode(self, syms: torch.Tensor, lens: torch.Tensor):
        return self.encoder(syms, lens)

    def forward_decoder(self, noisy_syms: torch.Tensor, noisy_lens: torch.Tensor):
        z, feats = self.encoder(noisy_syms, noisy_lens)
        msg_logits = self.decoder(feats, noisy_lens)
        return msg_logits, z, feats
