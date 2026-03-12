# Transformer from scratch

https://nlp.seas.harvard.edu/annotated-transformer/

## セットアップ

```bash
uv sync --dev
```

```bash
uv add pip
uv run python -m spacy download de_core_news_sm
uv run python -m spacy download en_core_web_sm
```

## 使い方

### 語彙構築

```bash
uv run python -m transformer_from_scratch build-vocab
```

初回実行時には Hugging Face Datasets から Multi30k を取得し、`vocab.pt` を保存します。

### 学習

単 GPU:

```bash
uv run python -m transformer_from_scratch train --batch-size 32 --num-epochs 8
```

複数 GPU:

```bash
uv run torchrun --nproc_per_node=3 -m transformer_from_scratch train --distributed --batch-size 96
```

## 翻訳

学習後は保存済み checkpoint から翻訳できます。

```bash
uv run python -m transformer_from_scratch translate "Zwei Hunde laufen durch den Schnee."
```

実行例1:

```text
Two dogs run through the snow .
```

実行例2:

```bash
uv run python -m transformer_from_scratch translate "Eine Frau sitzt auf einer Bank und liest ein Buch."
```

```text
A woman sits on a bench reading a book .
```
