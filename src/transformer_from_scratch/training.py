"""学習まわりの配管。"""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformer_from_scratch.layers import subsequent_mask
from transformer_from_scratch.model import EncoderDecoder


class LossCompute(Protocol):
    """run_epoch から呼ばれる loss 計算インターフェース。"""

    def __call__(self, x: Tensor, y: Tensor, norm: int) -> tuple[float, Tensor]: ...


class Batch:
    """学習用バッチと、それに対応する mask をまとめて持つ。"""

    def __init__(self, src: Tensor, tgt: Tensor | None = None, pad: int = 2) -> None:
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is None:
            self.tgt = None
            self.tgt_y = None
            self.tgt_mask = None
            self.ntokens = 0
            return

        # decoder 入力は末尾を落とし、教師信号は先頭を落とす。
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = self.make_std_mask(self.tgt, pad)
        self.ntokens = int((self.tgt_y != pad).sum().item())

    @staticmethod
    def make_std_mask(tgt: Tensor, pad: int) -> Tensor:
        """padding と future token の両方を隠す mask を作る。"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


@dataclass
class TrainState:
    """学習中の step 数や token 数を追跡する。"""

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter: Iterable[Batch],
    model: EncoderDecoder,
    loss_compute: LossCompute,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: TrainState | None = None,
) -> tuple[float, TrainState]:
    """1 epoch 分の train / eval を回す。"""
    if train_state is None:
        train_state = TrainState()

    is_training = mode in {"train", "train+log"}
    model.train(is_training)

    start = time.time()
    total_tokens = 0
    total_loss = 0.0
    tokens = 0
    n_accum = 0

    for step_index, batch in enumerate(data_iter):
        if batch.tgt is None or batch.tgt_y is None or batch.tgt_mask is None:
            msg = "training batches must include tgt, tgt_y, and tgt_mask"
            raise ValueError(msg)

        with torch.set_grad_enabled(is_training):
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if is_training:
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if optimizer is not None and (step_index + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1

            if scheduler is not None:
                scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if step_index % 40 == 1 and mode == "train+log" and optimizer is not None:
            learning_rate = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                "Epoch Step: "
                f"{step_index:6d} | "
                f"Accumulation Step: {n_accum:3d} | "
                f"Loss: {loss / batch.ntokens:6.2f} | "
                f"Tokens / Sec: {tokens / elapsed:7.1f} | "
                f"Learning Rate: {learning_rate:6.1e}"
            )
            start = time.time()
            tokens = 0

    if is_training and optimizer is not None and train_state.step % accum_iter != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_state.accum_step += 1

    if total_tokens == 0:
        msg = "total_tokens must be positive"
        raise ValueError(msg)

    return total_loss / total_tokens, train_state


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """Transformer 論文の learning rate schedule。"""
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def make_scheduler(
    optimizer: Optimizer,
    model_size: int,
    factor: float = 1.0,
    warmup: int = 4000,
) -> LambdaLR:
    """論文準拠の warmup + inverse square root decay を返す。"""
    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size, factor, warmup),
    )


def make_optimizer(model: EncoderDecoder) -> Adam:
    """論文準拠の Adam 設定を返す。"""
    return Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)


class LabelSmoothing(nn.Module):
    """KLDivLoss を使った label smoothing。"""

    def __init__(
        self,
        size: int,
        padding_idx: int,
        smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist: Tensor | None = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if x.size(1) != self.size:
            msg = "x.size(1) must match vocabulary size"
            raise ValueError(msg)

        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        padding_mask = torch.nonzero(target == self.padding_idx, as_tuple=False)
        if padding_mask.dim() > 0 and padding_mask.numel() > 0:
            true_dist.index_fill_(0, padding_mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.detach())


class SimpleLossCompute:
    """generator と criterion を束ねた最小の loss 計算 helper。"""

    def __init__(self, generator: nn.Module, criterion: nn.Module) -> None:
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: Tensor, y: Tensor, norm: int) -> tuple[float, Tensor]:
        logits = self.generator(x)
        loss = self.criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            y.contiguous().view(-1),
        )
        normalized_loss = loss / norm
        return float(loss.detach().item()), normalized_loss
