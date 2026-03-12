"""Multi30k を使った最小の翻訳学習入口。"""

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from transformer_from_scratch.data import (
    create_dataloaders,
    decode_tokens,
    encode_text,
    load_tokenizers,
    load_vocab,
)
from transformer_from_scratch.inference import greedy_decode
from transformer_from_scratch.model import EncoderDecoder, make_model
from transformer_from_scratch.training import (
    Batch,
    LabelSmoothing,
    SimpleLossCompute,
    TrainState,
    make_optimizer,
    make_scheduler,
    run_epoch,
)


@dataclass
class TrainConfig:
    """翻訳学習に使う最小構成。"""

    batch_size: int = 32
    num_epochs: int = 8
    accum_iter: int = 1
    max_padding: int = 72
    warmup: int = 3000
    model_path_prefix: str = "multi30k_model_"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    distributed: bool = torch.cuda.device_count() > 1


def _setup_device(config: TrainConfig) -> tuple[torch.device, bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = config.distributed and world_size > 1

    if is_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), True, local_rank, world_size

    return torch.device(config.device), False, local_rank, world_size


def _cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def train_model(config: TrainConfig | None = None) -> EncoderDecoder:
    """Multi30k で Transformer を学習する。"""
    if config is None:
        config = TrainConfig()

    device, is_distributed, local_rank, world_size = _setup_device(config)
    is_main_process = not is_distributed or local_rank == 0
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    pad_idx = vocab_tgt["<blank>"]

    model = make_model(len(vocab_src), len(vocab_tgt))
    model.to(device)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    module = model.module if is_distributed else model
    criterion = LabelSmoothing(
        size=len(vocab_tgt),
        padding_idx=pad_idx,
        smoothing=0.1,
    ).to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config.batch_size // world_size,
        max_padding=config.max_padding,
        is_distributed=is_distributed,
    )

    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, model_size=512, warmup=config.warmup)
    train_state = TrainState()

    try:
        for epoch in range(config.num_epochs):
            if is_distributed:
                train_sampler = train_dataloader.sampler
                valid_sampler = valid_dataloader.sampler
                if isinstance(train_sampler, torch.utils.data.distributed.DistributedSampler):
                    train_sampler.set_epoch(epoch)
                if isinstance(valid_sampler, torch.utils.data.distributed.DistributedSampler):
                    valid_sampler.set_epoch(epoch)

            model.train()
            _, train_state = run_epoch(
                (Batch(src, tgt, pad_idx) for src, tgt in train_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                optimizer,
                scheduler,
                mode="train+log" if is_main_process else "train",
                accum_iter=config.accum_iter,
                train_state=train_state,
            )

            model.eval()
            validation_loss, _ = run_epoch(
                (Batch(src, tgt, pad_idx) for src, tgt in valid_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                mode="eval",
            )
            if is_main_process:
                print(f"Epoch {epoch} validation loss: {validation_loss:.4f}")
                checkpoint_path = Path(f"{config.model_path_prefix}{epoch:02d}.pt")
                torch.save(module.state_dict(), checkpoint_path)

        if is_main_process:
            final_model_path = Path(f"{config.model_path_prefix}final.pt")
            torch.save(module.state_dict(), final_model_path)
        return module
    finally:
        _cleanup_distributed(is_distributed)


def load_trained_model(
    vocab_src: object,
    vocab_tgt: object,
    model_path: str | Path = "multi30k_model_final.pt",
    device: str | torch.device | None = None,
) -> EncoderDecoder:
    """保存済み checkpoint から model を復元する。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_model(len(vocab_src), len(vocab_tgt))
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def translate_text(
    text: str,
    model_path: str | Path = "multi30k_model_final.pt",
    max_len: int = 72,
    device: str | torch.device | None = None,
) -> str:
    """学習済みモデルで German -> English の翻訳を返す。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    model = load_trained_model(vocab_src, vocab_tgt, model_path=model_path, device=device)

    src = encode_text(text, spacy_de, vocab_src, device)
    src_mask = (src != vocab_src["<blank>"]).unsqueeze(-2)
    output = greedy_decode(
        model,
        src,
        src_mask,
        max_len=max_len,
        start_symbol=vocab_tgt["<s>"],
        end_symbol=vocab_tgt["</s>"],
    )
    tokens = decode_tokens(output[0], vocab_tgt)
    filtered_tokens = [token for token in tokens if token not in {"<s>", "</s>", "<blank>"}]
    return " ".join(filtered_tokens)


def build_parser() -> ArgumentParser:
    """CLI parser を返す。"""
    parser = ArgumentParser(prog="transformer-from-scratch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--num-epochs", type=int, default=8)
    train_parser.add_argument("--accum-iter", type=int, default=1)
    train_parser.add_argument("--max-padding", type=int, default=72)
    train_parser.add_argument("--warmup", type=int, default=3000)
    train_parser.add_argument("--model-path-prefix", default="multi30k_model_")
    train_parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    train_parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use torchrun / DDP when WORLD_SIZE > 1.",
    )

    vocab_parser = subparsers.add_parser("build-vocab")
    vocab_parser.add_argument("--vocab-path", default="vocab.pt")

    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument("text")
    translate_parser.add_argument("--model-path", default="multi30k_model_final.pt")
    translate_parser.add_argument("--max-len", type=int, default=72)
    translate_parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main() -> None:
    """学習・語彙構築・翻訳用の CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-vocab":
        spacy_de, spacy_en = load_tokenizers()
        load_vocab(spacy_de, spacy_en, path=args.vocab_path)
        print(f"Saved vocabulary to {args.vocab_path}")
        return

    if args.command == "train":
        config = TrainConfig(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            accum_iter=args.accum_iter,
            max_padding=args.max_padding,
            warmup=args.warmup,
            model_path_prefix=args.model_path_prefix,
            device=args.device,
            distributed=args.distributed,
        )
        train_model(config)
        return

    if args.command == "translate":
        print(
            translate_text(
                args.text,
                model_path=args.model_path,
                max_len=args.max_len,
                device=args.device,
            )
        )
