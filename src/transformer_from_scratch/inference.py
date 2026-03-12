"""推論まわりの配管。"""

import torch
from torch import Tensor

from transformer_from_scratch.layers import subsequent_mask
from transformer_from_scratch.model import EncoderDecoder


def greedy_decode(
    model: EncoderDecoder,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int | None = None,
) -> Tensor:
    """greedy decoding で 1 token ずつ出力系列を伸ばす。"""
    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), start_symbol, dtype=src.dtype, device=src.device)

    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src_mask)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=1, keepdim=True)
        ys = torch.cat([ys, next_word], dim=1)
        if end_symbol is not None and torch.all(next_word == end_symbol):
            break

    return ys
