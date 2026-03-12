"""Attention 関連の実装。"""

import math

import torch
from torch import Tensor, nn


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
