"""Embedding 関連の実装。"""

import math

import torch
from torch import Tensor, nn


class Embeddings(nn.Module):
    """token id を d_model 次元の埋め込みベクトルへ変換する。"""

    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        # 論文実装に合わせて、埋め込みを sqrt(d_model) 倍する。
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """埋め込みに系列位置の情報を加える。"""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        # pe[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # batch 次元を足して、(1, max_len, d_model) として保持する。
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # token embedding に位置 encoding を足してから dropout をかける。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
