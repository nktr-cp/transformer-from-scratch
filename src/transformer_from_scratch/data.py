"""Multi30k 用のデータロード。"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


@dataclass
class Vocab:
    """最小限の語彙ラッパ。"""

    stoi: dict[str, int]
    itos: list[str]
    default_index: int = 0

    def __call__(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.default_index) for token in tokens]

    def __getitem__(self, token: str) -> int:
        return self.stoi[token]

    def __len__(self) -> int:
        return len(self.itos)

    def set_default_index(self, index: int) -> None:
        self.default_index = index

    def lookup_tokens(self, indices: list[int]) -> list[str]:
        return [self.itos[index] for index in indices]


def load_tokenizers() -> tuple[object, object]:
    """German / English の spaCy tokenizer を読み込む。"""
    import spacy

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError as exc:
        msg = "spaCy model `de_core_news_sm` is required"
        raise RuntimeError(msg) from exc

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError as exc:
        msg = "spaCy model `en_core_web_sm` is required"
        raise RuntimeError(msg) from exc

    return spacy_de, spacy_en


def tokenize(text: str, tokenizer: object) -> list[str]:
    """spaCy tokenizer でテキストを token 列へ変換する。"""
    return [token.text for token in tokenizer.tokenizer(text)]


def yield_tokens(
    data_iter: Iterable[tuple[str, str]],
    tokenizer: Callable[[str], list[str]],
    index: int,
) -> Iterable[list[str]]:
    """語彙構築用に token 列を順次返す。"""
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(
    spacy_de: object,
    spacy_en: object,
) -> tuple[Vocab, Vocab]:
    """Multi30k から source / target の語彙を構築する。"""
    from datasets import load_dataset

    def tokenize_de(text: str) -> list[str]:
        return tokenize(text, spacy_de)

    def tokenize_en(text: str) -> list[str]:
        return tokenize(text, spacy_en)

    dataset = load_dataset("bentrevett/multi30k")
    all_examples: list[tuple[str, str]] = []
    for split_name in ("train", "validation", "test"):
        for row in dataset[split_name]:
            all_examples.append((row["de"], row["en"]))

    vocab_src = build_vocab(
        yield_tokens(all_examples, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )
    vocab_tgt = build_vocab(
        yield_tokens(all_examples, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])
    return vocab_src, vocab_tgt


def build_vocab(
    token_iter: Iterable[list[str]],
    min_freq: int,
    specials: list[str],
) -> Vocab:
    """token iterator から最小限の語彙を構築する。"""
    counter: Counter[str] = Counter()
    for tokens in token_iter:
        counter.update(tokens)

    itos = list(specials)
    for token, freq in counter.items():
        if freq >= min_freq and token not in specials:
            itos.append(token)
    stoi = {token: index for index, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def load_vocab(
    spacy_de: object,
    spacy_en: object,
    path: str | Path = "vocab.pt",
) -> tuple[Vocab, Vocab]:
    """語彙をキャッシュから読む。なければ構築して保存する。"""
    vocab_path = Path(path)
    if vocab_path.exists():
        vocab_src, vocab_tgt = torch.load(vocab_path, weights_only=False)
        return vocab_src, vocab_tgt

    vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
    torch.save((vocab_src, vocab_tgt), vocab_path)
    return vocab_src, vocab_tgt


def collate_batch(
    batch: list[tuple[str, str]],
    src_pipeline: Callable[[str], list[str]],
    tgt_pipeline: Callable[[str], list[str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_padding: int = 128,
    pad_id: int = 2,
) -> tuple[Tensor, Tensor]:
    """文字列 batch を固定長の token id tensor へまとめる。"""
    bos_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)
    src_list: list[Tensor] = []
    tgt_list: list[Tensor] = []

    for src_text, tgt_text in batch:
        processed_src = torch.cat(
            [
                bos_id,
                torch.tensor(src_vocab(src_pipeline(src_text)), dtype=torch.int64, device=device),
                eos_id,
            ],
            dim=0,
        )
        processed_tgt = torch.cat(
            [
                bos_id,
                torch.tensor(tgt_vocab(tgt_pipeline(tgt_text)), dtype=torch.int64, device=device),
                eos_id,
            ],
            dim=0,
        )

        src_list.append(
            pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id)
        )
        tgt_list.append(
            pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id)
        )

    return torch.stack(src_list), torch.stack(tgt_list)


def create_dataloaders(
    device: torch.device,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    spacy_de: object,
    spacy_en: object,
    batch_size: int = 32,
    max_padding: int = 128,
    is_distributed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Multi30k の train / valid dataloader を返す。"""
    from datasets import load_dataset

    def tokenize_de(text: str) -> list[str]:
        return tokenize(text, spacy_de)

    def tokenize_en(text: str) -> list[str]:
        return tokenize(text, spacy_en)

    def collate_fn(batch: list[tuple[str, str]]) -> tuple[Tensor, Tensor]:
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src["<blank>"],
        )

    dataset = load_dataset("bentrevett/multi30k")
    train_dataset = [(row["de"], row["en"]) for row in dataset["train"]]
    valid_dataset = [(row["de"], row["en"]) for row in dataset["validation"]]
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if is_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def encode_text(
    text: str,
    tokenizer: object,
    vocab: Vocab,
    device: torch.device,
) -> Tensor:
    """単文を BOS/EOS 付き token id tensor に変換する。"""
    bos_id = torch.tensor([0], dtype=torch.int64, device=device)
    eos_id = torch.tensor([1], dtype=torch.int64, device=device)
    token_ids = torch.tensor(vocab(tokenize(text, tokenizer)), dtype=torch.int64, device=device)
    return torch.cat([bos_id, token_ids, eos_id], dim=0).unsqueeze(0)


def decode_tokens(token_ids: Tensor, vocab: Vocab) -> list[str]:
    """token id 列を token 列へ戻す。"""
    return vocab.lookup_tokens(token_ids.tolist())
