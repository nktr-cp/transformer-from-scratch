"""Attention 関連の実装。"""

import math

import torch
from torch import Tensor, nn

from transformer_from_scratch.layers import clones


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """Scaled Dot-Product Attention を計算する。"""
    d_k = query.size(-1)

    # Q K^T
    # 各 query と各 key の内積を一括計算し、
    # query ごとの生スコアを作る。
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Q K^T / sqrt(d_k)
    # d_k が大きいと内積も大きくなりやすく、
    # softmax が飽和して学習しづらくなるためスケーリングする。
    scores /= math.sqrt(d_k)

    if mask is not None:
        # mask が 0 の位置は見てはいけない位置なので、
        # softmax 後の重みがほぼ 0 になるよう非常に小さい値で埋める。
        scores = scores.masked_fill(mask == 0, -1e9)

    # softmax(Q K^T / sqrt(d_k))
    # 各 query が各 key にどれだけ注意を向けるかを確率化する。
    attention_weights = scores.softmax(dim=-1)

    if dropout is not None:
        # attention の重みに dropout をかけて正則化する。
        attention_weights = dropout(attention_weights)

    # softmax(Q K^T / sqrt(d_k)) V
    # attention の重みで value を重み付き和し、出力ベクトルを得る。
    return torch.matmul(attention_weights, value), attention_weights


class MultiHeadedAttention(nn.Module):
    """複数の attention head を並列に計算する。"""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % h != 0:
            msg = "d_model must be divisible by h"
            raise ValueError(msg)

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention_weights: Tensor | None = None
        self.dropout = nn.Dropout(p=dropout)

    def _project(
        self,
        x: Tensor,
        linear: nn.Linear,
        batch_size: int,
    ) -> Tensor:
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        projected = linear(x)
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        projected = projected.view(batch_size, -1, self.h, self.d_k)
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        return projected.transpose(1, 2)

    def _concat_heads(self, x: Tensor, batch_size: int) -> Tensor:
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k)
        x = x.transpose(1, 2).contiguous()
        # (batch, seq_len, h, d_k) -> (batch, seq_len, h * d_k)
        return x.view(batch_size, -1, self.h * self.d_k)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if mask is not None:
            # mask に head 軸を 1 つ足し、
            # 同じ mask を全 head に適用できるようにする。
            # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # Q = query W_Q
        # (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        projected_query = self._project(query, self.linears[0], batch_size)
        # K = key W_K
        # (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        projected_key = self._project(key, self.linears[1], batch_size)
        # V = value W_V
        # (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        projected_value = self._project(value, self.linears[2], batch_size)

        x, self.attention_weights = attention(
            projected_query,
            projected_key,
            projected_value,
            mask=mask,
            dropout=self.dropout,
        )

        # Concat(head_1, ..., head_h)
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h * d_k)
        x = self._concat_heads(x, batch_size)

        # Concat した結果に最終線形変換 W_O をかける。
        return self.linears[-1](x)
