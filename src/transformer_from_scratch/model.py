"""モデル全体の組み立て。"""

import torch
from torch import Tensor, nn

from transformer_from_scratch.attention import MultiHeadedAttention
from transformer_from_scratch.embeddings import Embeddings, PositionalEncoding
from transformer_from_scratch.layers import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    PositionwiseFeedForward,
)


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


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    n: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """ハイパーパラメータから Transformer を組み立てる。"""
    encoder_layer = EncoderLayer(
        d_model,
        MultiHeadedAttention(h, d_model, dropout),
        PositionwiseFeedForward(d_model, d_ff, dropout),
        dropout,
    )
    decoder_layer = DecoderLayer(
        d_model,
        MultiHeadedAttention(h, d_model, dropout),
        MultiHeadedAttention(h, d_model, dropout),
        PositionwiseFeedForward(d_model, d_ff, dropout),
        dropout,
    )
    src_embedding = nn.Sequential(
        Embeddings(d_model, src_vocab),
        PositionalEncoding(d_model, dropout),
    )
    tgt_embedding = nn.Sequential(
        Embeddings(d_model, tgt_vocab),
        PositionalEncoding(d_model, dropout),
    )

    model = EncoderDecoder(
        Encoder(encoder_layer, n),
        Decoder(decoder_layer, n),
        src_embedding,
        tgt_embedding,
        Generator(d_model, tgt_vocab),
    )

    # 重み行列は論文実装に合わせて Xavier で初期化する。
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return model
