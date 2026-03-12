# Transformer from scratch

https://nlp.seas.harvard.edu/annotated-transformer/

Annotated Transformer を PyTorch で実装したプロジェクトです。  
現在は Multi30k を使った German -> English の学習と翻訳まで動きます。

## セットアップ

```bash
uv sync --dev
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
uv run python -m transformer_from_scratch translate "Ein Mann mit einem Hut spielt Gitarre."
uv run python -m transformer_from_scratch translate "Zwei Kinder spielen im Wasser."
uv run python -m transformer_from_scratch translate "Eine Frau in einem blauen Kleid tanzt."
```

実際の出力:

```text
A man in a hat is playing a guitar .
Two children are playing in the water .
A woman in a blue dress is dancing .
```
