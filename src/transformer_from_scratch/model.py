"""モデル定義。"""

import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """同一構造のモジュールを n 個複製する。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class EncoderDecoder(nn.Module):
    """標準的な Encoder-Decoder アーキテクチャ全体を束ねる。"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module,
    ) -> None:
        super().__init__()
        # encoder: 入力系列を文脈化する本体
        self.encoder = encoder
        # decoder: encoder の出力を参照しつつ出力側表現を作る本体
        self.decoder = decoder
        # src_embed: 入力 token id をモデル内部のベクトルへ変換する層
        self.src_embed = src_embed
        # tgt_embed: 出力 token id をモデル内部のベクトルへ変換する層
        self.tgt_embed = tgt_embed
        # generator: decoder の出力を語彙 logits へ変換する最終射影
        self.generator = generator

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """入力と出力系列を受け取り、decoder 出力を返す。"""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: Tensor,
        src_mask: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """decoder の出力を語彙方向の対数確率へ変換する。"""

    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        # proj: 各位置の隠れ状態を語彙数ぶんの logits へ写す
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return torch.log_softmax(self.proj(x), dim=-1)


class LayerNorm(nn.Module):
    """系列の最後の次元に対して Layer Normalization を適用する。"""

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        # normalized = (x - mean) / (std + eps)
        # output = scale * normalized + bias
        # scale と bias は最後の次元ごとに学習されるパラメータ。
        self.scale = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.scale * normalized + self.bias


class Encoder(nn.Module):
    """EncoderLayer を縦に積んだ encoder 本体。"""

    def __init__(self, layer: nn.Module, n: int) -> None:
        super().__init__()
        # layers: 同じ構造の EncoderLayer を n 段積む
        self.layers = clones(layer, n)
        # norm: 最終段のあとにかける LayerNorm
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """残差接続で sublayer を包む。"""

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        # pre-norm を採用する。
        # 論文本文の LayerNorm(x + Sublayer(x)) ではなく、
        # x + Dropout(Sublayer(LayerNorm(x))) の順序にする。
        # Pre-LN は学習安定性の面で有利とされる
        # (Xiong et al., 2020, https://arxiv.org/abs/2002.04745)。
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        sublayer: Callable[[Tensor], Tensor],
    ) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """self-attention と feed-forward からなる encoder の 1 層。"""

    def __init__(
        self,
        size: int,
        self_attention: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()
        # self_attention: 一般形 attention(query, key, value, mask) の
        # self-attention 版を受け持つ。encoder では query=key=value=x。
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.sublayers[0](
            x,
            lambda normalized_x: self.self_attention(
                normalized_x,
                normalized_x,
                normalized_x,
                mask,
            ),
        )
        return self.sublayers[1](x, self.feed_forward)


class Decoder(nn.Module):
    """DecoderLayer を縦に積んだ decoder 本体。"""

    def __init__(self, layer: nn.Module, n: int) -> None:
        super().__init__()
        # layers: 同じ構造の DecoderLayer を n 段積む
        self.layers = clones(layer, n)
        # norm: 最終段のあとにかける LayerNorm
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """self-attention と cross-attention を含む decoder の 1 層。"""

    def __init__(
        self,
        size: int,
        self_attention: nn.Module,
        src_attention: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        # self_attention: target 系列どうしを見る attention
        self.self_attention = self_attention
        # src_attention: encoder の出力を見る cross-attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        x = self.sublayers[0](
            x,
            lambda normalized_x: self.self_attention(
                normalized_x,
                normalized_x,
                normalized_x,
                # この時点では先のトークンを見ないように mask する。
                tgt_mask,
            ),
        )
        x = self.sublayers[1](
            x,
            lambda normalized_x: self.src_attention(
                normalized_x,
                # encoder の出力との cross-attention。
                memory,
                memory,
                src_mask,
            ),
        )
        return self.sublayers[2](x, self.feed_forward)


def subsequent_mask(size: int) -> Tensor:
    """未来の位置を見ないための causal mask を作る。"""
    attention_shape = (1, size, size)
    mask = torch.triu(torch.ones(attention_shape), diagonal=1).type(torch.uint8)
    return mask == 0
