# Topic 4 - Transformers + Translator Capstone: Cell-by-Cell Plan

## Overview

Topic 4 opens Day 2 and is the payoff of Day 1: students spent all of Day 1 building
attention from scratch (NumPy, then PyTorch), and now they stack it into a full
Transformer encoder-decoder and run their first GPU training job on SageMaker.
The running narrative is that Barclays needs to translate Spanish customer complaints
to English so the triage system built on Day 1 can route them correctly.
NOTE: notebooks use Spanish (not French) to match 11_Transformers_Translator.ipynb source.
Estimated in-class time: 90 to 120 minutes.

---

## Diagram Index

Diagram 1: slug=transformer-architecture, path=plans/topic_4/diagrams/transformer-architecture.mmd
  Description: Full encoder-decoder Transformer architecture. Left side shows the Encoder
  stack: Input Tokens -> Embedding -> Positional Encoding -> N x [Multi-Head Self-Attention ->
  Add+Norm -> Feed-Forward -> Add+Norm]. Right side shows the Decoder stack: Output Tokens
  (shifted right) -> Embedding -> Positional Encoding -> N x [Masked Multi-Head Self-Attention ->
  Add+Norm -> Multi-Head Cross-Attention -> Add+Norm -> Feed-Forward -> Add+Norm].
  Final Linear + Softmax on top of the Decoder. Arrows connecting Encoder output to every
  Decoder cross-attention block. All sublayer labels shown. d_model=128, N=2 labelled for
  the capstone config.

Diagram 2: slug=positional-encoding-pattern, path=plans/topic_4/diagrams/positional-encoding-pattern.mmd
  Description: Heatmap showing sinusoidal positional encoding values. X-axis: embedding
  dimensions 0 to 31. Y-axis: sequence positions 0 to 15. Color: red for positive values,
  blue for negative. Alternating sin/cos stripes visible diagonally. Annotate that nearby
  positions (rows) have similar colors (smooth) while distant positions diverge. This is
  what the model sees as position information -- no learned parameters, pure math.
  NOTE: actual notebook uses positional-encoding-pattern.mmd (not multi-head-attention.mmd)
  because the exercise focuses on positional encoding as the key Beat 2 concept.

---

## Source Dir (scripts_topic4/)

### train.py

```python
"""
train.py -- Transformer translator (French -> English) for SageMaker GPU job.

Architecture: 2 encoder layers, 2 decoder layers, d_model=128, nhead=4.
Dataset: opus_books en-fr, max 50k sentence pairs.
Target: under 30 min on ml.g4dn.xlarge (NVIDIA T4 16GB).

SageMaker toolkit auto-installs requirements.txt before running this script.
Hyperparameters are passed as CLI args by the PyTorch estimator.
"""

import argparse
import os
import math
import random
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# sacrebleu for BLEU score (no evaluate library -- incompatible with datasets 4.x)
import sacrebleu


# ---------------------------------------------------------------------------
# Argument parsing (SageMaker passes hyperparameters as CLI args)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--d_model",            type=int,   default=128)
    parser.add_argument("--nhead",              type=int,   default=4)
    parser.add_argument("--num_encoder_layers", type=int,   default=2)
    parser.add_argument("--num_decoder_layers", type=int,   default=2)
    parser.add_argument("--dim_feedforward",    type=int,   default=512)
    parser.add_argument("--dropout",            type=float, default=0.1)
    parser.add_argument("--max_seq_len",        type=int,   default=64)
    parser.add_argument("--batch_size",         type=int,   default=128)
    parser.add_argument("--epochs",             type=int,   default=5)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--max_pairs",          type=int,   default=50000)
    parser.add_argument("--vocab_size",         type=int,   default=8000)
    parser.add_argument("--seed",               type=int,   default=42)

    # SageMaker environment
    parser.add_argument("--model-dir",  type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Simple word-level vocabulary (no transformers dependency)
# ---------------------------------------------------------------------------

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def build_vocab(sentences, max_vocab):
    """Count word frequencies and return word-to-index dict."""
    from collections import Counter
    counter = Counter()
    for sent in sentences:
        counter.update(sent.lower().split())
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, _ in counter.most_common(max_vocab - len(SPECIAL_TOKENS)):
        vocab[word] = len(vocab)
    return vocab


def encode(sentence, vocab, max_len):
    """Encode a sentence to a fixed-length tensor with BOS/EOS."""
    tokens = sentence.lower().split()[: max_len - 2]
    ids = [BOS_IDX] + [vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids[:max_len]


def decode_ids(ids, idx2word):
    """Convert ids back to a string for BLEU evaluation."""
    words = []
    for i in ids:
        if i == EOS_IDX:
            break
        if i not in (PAD_IDX, BOS_IDX):
            words.append(idx2word.get(i, "<unk>"))
    return " ".join(words)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(self, src_ids, tgt_ids):
        self.src = torch.tensor(src_ids, dtype=torch.long)
        self.tgt = torch.tensor(tgt_ids, dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq_len + 10)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_key_padding_mask=None):
        x = self.src_emb(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        out = self.transformer.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(out)

    def forward(self, src, tgt, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_len = tgt.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=src.device
        )
        memory = self.encode(src, src_key_padding_mask)
        return self.decode(tgt, memory, tgt_mask,
                           tgt_key_padding_mask, memory_key_padding_mask)


# ---------------------------------------------------------------------------
# Padding mask helper
# ---------------------------------------------------------------------------

def make_pad_mask(token_ids):
    """Return bool mask: True where token is PAD. Shape: (batch, seq)."""
    return token_ids == PAD_IDX


# ---------------------------------------------------------------------------
# BLEU evaluation using sacrebleu (not evaluate library -- L6)
# ---------------------------------------------------------------------------

def evaluate_bleu(model, loader, idx2tgt, device, max_batches=20):
    model.eval()
    hypotheses = []
    references = []
    with torch.no_grad():
        for batch_idx, (src_batch, tgt_batch) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            src = src_batch.T.to(device)
            tgt = tgt_batch.T.to(device)
            src_pad = make_pad_mask(src_batch).to(device)
            memory = model.encode(src, src_pad)
            batch = src.size(1)
            ys = torch.full((1, batch), BOS_IDX, dtype=torch.long, device=device)
            for _ in range(tgt.size(0) - 1):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    ys.size(0), device=device
                )
                logits = model.decode(ys, memory, tgt_mask=tgt_mask)
                next_token = logits[-1].argmax(-1, keepdim=True).T
                ys = torch.cat([ys, next_token], dim=0)
            for i in range(batch):
                hyp = decode_ids(ys[1:, i].tolist(), idx2tgt)
                ref = decode_ids(tgt[1:, i].tolist(), idx2tgt)
                hypotheses.append(hyp)
                references.append([ref])
    bleu = sacrebleu.corpus_bleu(hypotheses, list(zip(*references)))
    return bleu.score


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for src_batch, tgt_batch in loader:
        src = src_batch.T.to(device)
        tgt = tgt_batch.T.to(device)
        src_pad     = make_pad_mask(src_batch).to(device)
        tgt_pad     = make_pad_mask(tgt_batch).to(device)
        tgt_input   = tgt[:-1]
        tgt_labels  = tgt[1:]
        tgt_in_pad  = tgt_pad[:, :-1]
        optimizer.zero_grad()
        logits = model(
            src, tgt_input,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_in_pad,
            memory_key_padding_mask=src_pad,
        )
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_labels.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    print("Loading opus_books en-fr dataset ...")
    from datasets import load_dataset

    raw = load_dataset("opus_books", "en-fr", split="train")
    pairs = [
        (row["translation"]["fr"], row["translation"]["en"])
        for row in raw
        if row["translation"]["fr"].strip() and row["translation"]["en"].strip()
    ]
    random.shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    print(f"Using {len(pairs)} sentence pairs")

    split      = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    src_vocab = build_vocab([p[0] for p in train_pairs], args.vocab_size)
    tgt_vocab = build_vocab([p[1] for p in train_pairs], args.vocab_size)
    idx2tgt   = {v: k for k, v in tgt_vocab.items()}

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    def enc_pairs(plist, sv, tv):
        return (
            [encode(p[0], sv, args.max_seq_len) for p in plist],
            [encode(p[1], tv, args.max_seq_len) for p in plist],
        )

    tr_src, tr_tgt = enc_pairs(train_pairs, src_vocab, tgt_vocab)
    vl_src, vl_tgt = enc_pairs(val_pairs,   src_vocab, tgt_vocab)

    train_loader = DataLoader(TranslationDataset(tr_src, tr_tgt),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(TranslationDataset(vl_src, vl_tgt),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_bleu      = -1.0
    best_model_path = os.path.join(args.model_dir, "best_model.pt")
    metrics        = []

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        bleu       = evaluate_bleu(model, val_loader, idx2tgt, device, max_batches=20)
        elapsed    = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | BLEU={bleu:.2f} | time={elapsed:.0f}s")
        metrics.append({"epoch": epoch, "train_loss": train_loss, "bleu": bleu})
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model (BLEU={best_bleu:.2f})")

    torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pt"))

    config = {
        "d_model": args.d_model, "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout, "max_seq_len": args.max_seq_len,
        "src_vocab_size": len(src_vocab), "tgt_vocab_size": len(tgt_vocab),
    }
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    torch.save(src_vocab, os.path.join(args.model_dir, "src_vocab.pt"))
    torch.save(tgt_vocab, os.path.join(args.model_dir, "tgt_vocab.pt"))
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete. Best BLEU: {best_bleu:.2f}")
    print(f"Artifacts saved to {args.model_dir}")


if __name__ == "__main__":
    main()
```

### requirements.txt

```
sacrebleu
datasets==2.18.0
numpy<2
```

---

## Key Changes from 11_Transformers_Translator.ipynb

**Keeping**:
- The `positional_encoding` function (cell-10) -- adapted to `PositionalEncoding` nn.Module.
- The `EncoderLayer` / `DecoderLayer` structures (cells 34, 41) as Beat 3 reference code.
- The `Translator` class concept (cell-47) -- renamed `TransformerTranslator` in train.py.
- The overall build-up order: positional encoding -> self-attention -> cross-attention ->
  encoder -> decoder -> full translator.

**Restructuring**:
- Source jumps straight to components with no four-beat arc. We wrap every concept in
  Beat 1 (failure/limitation) -> Beat 2 (diagram) -> Beat 3 (working demo) -> Beat 4 (lab).
- Source uses Spanish->English; we switch to French->English (Barclays Channel Islands narrative).
- Source targets Colab; we target SageMaker Studio (canonical install cell + sagemaker.Session()).
- Source has no labs, no safety-nets, no discussion prompts, no STAR method.
- Source uses `textblob`, `gensim`, `pytorch-nlp`, `swifter` -- all removed.
- Source has no GPU training job. We add a full SageMaker PyTorch estimator capstone.
- Source uses `num_heads=5` (not power of 2); we standardise to `nhead=4` for d_model=128.

**Replacing**:
- `textblob`, `gensim`, `pytorch-nlp`, `swifter` imports: removed entirely.
- Source `positional_encoding` is a bare function; we wrap it in `PositionalEncoding(nn.Module)`.
- Training in-notebook (source has none): replaced by SageMaker remote GPU job in capstone.
- Evaluation: sacrebleu instead of evaluate library (L6 from SAGEMAKER_LESSONS_LEARNED).
- Beat 1 failures added for: seq2seq sequential bottleneck, missing positional encoding.

---

## Variable Continuity from Topic 3b

The following names carry forward from `topic_3b_attention_pytorch.md`:

- `ScaledDotProductAttention` -- referenced explicitly in the multi-head attention Beat 3
  demo to show how MHA reuses the same operation from Topic 3b.
- `COMPLAINT_TOKENS` list -- reused in the positional encoding demo and multi-head demo.
- `device` -- same pattern (`torch.device("cuda" if ... else "cpu")`).
- `set_seeds(seed)` -- identical signature and body.
- `attn_weights`, `context`, `Q`, `K`, `V` -- used in multi-head attention section.

New variables introduced in Topic 4 that downstream cells depend on:
- `PositionalEncoding` class -- used in Lab 1, safety-net, and TransformerModel.
- `TransformerModel` class -- used in Lab 2 verification and SageMaker capstone.
- `estimator` -- the `sagemaker.pytorch.PyTorch` object (Cell 31).
- `training_job_name` -- returned by `estimator.fit(wait=False)` (Cell 31, used in 32/34/35).

---

## Cell-by-Cell Plan

### Cell 1: markdown - Title and Learning Objectives

```
# Topic 4 - Transformers + Translator Capstone

Barclays Customer Support Intelligence System | Day 2, Topic 4

## What you will build

Yesterday you implemented scaled dot product attention from scratch in NumPy and PyTorch.
Today you stack it into a full Transformer encoder-decoder and train a French-to-English
complaint translator on a GPU instance -- your first remote training job on SageMaker.

## Why this matters to Barclays

Barclays serves French-speaking customers across the Channel Islands and France.
Today you build the translation layer that converts incoming French complaints to English
so the triage system from Day 1 can route them correctly.

## Learning objectives

1. Identify why seq2seq with attention is still limited (sequential computation, no parallelism)
2. Implement multi-head attention from scratch using your ScaledDotProductAttention from Topic 3b
3. Explain sinusoidal positional encoding and why it is needed when you remove the RNN
4. Assemble a complete Transformer encoder-decoder from sublayers
5. Launch and monitor a GPU training job via the SageMaker PyTorch estimator

## Estimated time

90 to 120 minutes in class.
```

---

### Cell 2: code - Environment Setup and Installs

```python
# Environment setup for SageMaker Studio.
# Heavy training runs as a remote GPU job (Section 4), not in this kernel.
# This kernel handles all architecture demos (CPU is fine).

!pip install -q "sagemaker>=2.200.0,<3.0.0" \
    "numpy<2" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0"

import sagemaker
from sagemaker import get_execution_role
import boto3

sess   = sagemaker.Session()
role   = get_execution_role()
bucket = sess.default_bucket()
region = sess.boto_region_name

print(f"Role:   {role}")
print(f"Bucket: {bucket}")
print(f"Region: {region}")
```

---

### Cell 3: code - PyTorch Imports and Configuration

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seeds(42)

# CPU kernel for demos; GPU job runs in scripts_topic4/train.py.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch:  {torch.__version__}")
print(f"Device:   {device}")
print()

# Complaint tokens carried over from Topic 3b for all demos.
COMPLAINT_TOKENS = [
    "unauthorised", "charge", "account", "fraud",
    "refund", "dispute", "urgent", "branch"
]
print(f"Complaint vocabulary for demos: {COMPLAINT_TOKENS}")
```

---

### Cell 4: markdown - Section 0: The Limitation We Are Solving

```
## Section 0 - Why Attention Alone Is Not Enough

In Topic 3b you built scaled dot product attention.
In seq2seq with attention (the architecture from before the Transformer),
attention was applied at each DECODER step to a set of encoder hidden states.

The encoder itself was still an RNN: it processed tokens one at a time,
left to right. This creates two problems:

1. Sequential dependency: token N cannot be processed until tokens 1..N-1 are done.
   Long documents are slow. You cannot parallelise across the sequence dimension.

2. Long-range vanishing gradient: even with attention in the decoder,
   the ENCODER still compresses information through hidden states.
   Very long sequences still lose early context.

The Transformer (Vaswani et al., 2017) fixes both problems by applying
ONLY attention -- no RNN at all. Every token attends directly to every other token
in one sequential step (all positions processed in parallel within each layer).

The cost: without an RNN, the model has no notion of order.
We must inject position information explicitly.
That is what positional encoding does (Section 1).
```

---

### Cell 5: code - Beat 1: Sequential Bottleneck Demo

```python
# Beat 1: RNN encoder has a sequential dependency -- we cannot parallelize it.
# We time a GRU encoder vs a Transformer encoder layer on increasing sequence lengths.

import time

def time_rnn_encoder(seq_len, batch_size=32, d_model=128, runs=3):
    """Time a GRU encoder. Returns avg seconds per forward pass."""
    gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
    gru.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        gru(x)   # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            gru(x)
    return (time.perf_counter() - t0) / runs

def time_transformer_encoder(seq_len, batch_size=32, d_model=128, nhead=4, runs=3):
    """Time one TransformerEncoderLayer."""
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=512, batch_first=True)
    layer.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        layer(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            layer(x)
    return (time.perf_counter() - t0) / runs

print("RNN vs Transformer Encoder -- forward pass time (CPU, batch=32, d=128)")
print(f"{'seq_len':>10}  {'GRU (ms)':>12}  {'Transformer (ms)':>18}  {'speedup':>10}")
print("-" * 58)

for seq_len in [32, 64, 128, 256, 512]:
    rnn_t = time_rnn_encoder(seq_len) * 1000
    tfm_t = time_transformer_encoder(seq_len) * 1000
    speedup = rnn_t / tfm_t if tfm_t > 0 else float("inf")
    print(f"{seq_len:>10}  {rnn_t:>12.1f}  {tfm_t:>18.1f}  {speedup:>10.2f}x")

print()
print("GRU time grows roughly linearly with seq_len (sequential dependency).")
print("TransformerEncoderLayer processes all positions in parallel.")
print("On GPU the speedup is dramatically larger.")
print()
print("The downside: with no RNN, there is no position information.")
print("FIX: sinusoidal positional encoding (Section 1).")
```

---

### Cell 6: markdown - Discussion: Sequential vs Parallel

```
### Discussion (3 minutes)

You have just seen that a GRU encoder has a sequential bottleneck
while a Transformer encoder layer processes all positions in parallel.

1. For a document with 10,000 tokens (a long complaint thread),
   how many sequential steps does a GRU encoder require?
   How many does a Transformer encoder require?

2. The Transformer attention has O(n^2) memory cost in the sequence length n
   (the attention matrix is n x n). For n=10,000 this is 10^8 elements.
   Is parallelism worth this memory cost? When would you prefer a GRU?

3. The Transformer has no concept of order without positional encoding.
   What would happen if you ran the same Transformer encoder on
   "The customer charged us" and "We charged the customer" (same tokens, different order)
   WITHOUT positional encoding?
```

---

### Cell 7: markdown - Section 1: Positional Encoding

```
## Section 1 - Sinusoidal Positional Encoding

Because the Transformer processes all positions in parallel,
it loses the sequential order that an RNN gets for free.

The fix: add a deterministic positional signal to each token embedding BEFORE
the first attention layer. The signal must:

1. Be unique for each position (no two positions have the same encoding)
2. Generalise to sequences longer than those seen during training
3. Allow the model to learn relative positions (position 5 relative to position 3)

The "Attention is all you need" paper uses sinusoidal encoding:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where pos is the position index and i is the dimension index.
The frequency decreases geometrically as i increases, so each dimension
oscillates at a different period -- like a binary clock with smooth waves.
```

---

### Cell 8: code - Beat 1: Missing Positional Encoding (Order Blind)

```python
# Beat 1: Without positional encoding, the Transformer ignores token order.
# We show this concretely: two sequences with reversed token order produce
# outputs where the SET of norms is the same (just rearranged).

set_seeds(42)

d_model = 32
nhead   = 4

layer_no_pos = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=128, batch_first=True)
layer_no_pos.eval()

emb = nn.Embedding(10, d_model)
emb.eval()

# Two sequences: same tokens, reversed order
seq_a = torch.tensor([[1, 2, 3, 4, 5]])   # "charge account fraud refund dispute"
seq_b = torch.tensor([[5, 4, 3, 2, 1]])   # reversed

with torch.no_grad():
    out_a = layer_no_pos(emb(seq_a))   # (1, 5, d_model)
    out_b = layer_no_pos(emb(seq_b))

print("Beat 1: Token order is invisible without positional encoding")
print("=" * 55)
print(f"Sequence A: {seq_a.tolist()}")
print(f"Sequence B: {seq_b.tolist()} (reversed)")
print()

norm_a = out_a[0].norm(dim=-1).detach().numpy()
norm_b = out_b[0].norm(dim=-1).detach().numpy()
print("Output L2 norms per position:")
print(f"  Seq A: {norm_a.round(3)}")
print(f"  Seq B: {norm_b.round(3)}")
print()

# Sort and compare: the multisets of norms should match (same tokens -> same set of outputs)
sorted_a = sorted(norm_a.tolist())
sorted_b = sorted(norm_b.tolist())
print("Sorted norms (same tokens should give same multiset of outputs):")
print(f"  Sorted A: {[round(x,3) for x in sorted_a]}")
print(f"  Sorted B: {[round(x,3) for x in sorted_b]}")
print()
print("The transformer without positional encoding cannot distinguish")
print("'I was charged unfairly' from 'Unfairly charged I was'.")
print("FIX: add sinusoidal positional encoding before the first attention layer.")
```

---

### Cell 9: code - Beat 3: Positional Encoding -- Full Working Demo

```python
# Beat 3: Sinusoidal positional encoding implementation and visualisation.
# This is the pattern used in the original paper and in train.py.
# batch_first=True convention: input is (batch, seq, d_model).

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from Vaswani et al. (2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Adds a fixed (non-learned) positional signal to token embeddings.
    Convention: input shape is (batch, seq, d_model) -- batch_first=True.
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # position indices: (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # Frequency terms: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  dims: cos

        # Register as buffer (not a learnable parameter, moves with .to(device))
        # Add batch dim for broadcasting: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# --- Demo ---
set_seeds(42)

d_model = 128
max_len = 64
pe_module = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=max_len)

# Zero input -> output IS the positional encoding
dummy_input = torch.zeros(1, max_len, d_model)
pe_output   = pe_module(dummy_input).squeeze(0).detach().numpy()   # (64, 128)

print("PositionalEncoding demo")
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {pe_module(dummy_input).shape}")
print()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].imshow(pe_output[:20, :32], cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
axes[0].set_title("Positional Encoding -- first 20 positions, first 32 dims")
axes[0].set_xlabel("Dimension index")
axes[0].set_ylabel("Position index")

for dim in [0, 1, 4, 5, 8, 9]:
    axes[1].plot(pe_output[:40, dim], label=f"dim {dim}")
axes[1].set_title("Encoding signal vs position for selected dimensions")
axes[1].set_xlabel("Position")
axes[1].set_ylabel("Encoding value")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

print("Low dimensions (0, 1) oscillate quickly -- capture fine-grained position.")
print("High dimensions oscillate slowly -- capture coarse-grained position.")
print("Together they form a unique fingerprint for every position up to max_len.")
```

---

### Cell 10: markdown - Beat 2: Transformer Architecture Diagram

```
<!-- DIAGRAM: Full encoder-decoder Transformer architecture. Left (Encoder): Input Tokens -> Embedding + Positional Encoding -> 2x [Multi-Head Self-Attention -> Add+Norm -> Feed-Forward -> Add+Norm]. Right (Decoder): Output Tokens (shifted right) -> Embedding + Positional Encoding -> 2x [Masked Multi-Head Self-Attention -> Add+Norm -> Multi-Head Cross-Attention (attends to Encoder output) -> Add+Norm -> Feed-Forward -> Add+Norm] -> Linear -> Softmax -> Output Probabilities. Arrows from Encoder output to every Decoder cross-attention block. d_model=128, N=2 labelled. -->
[View diagram](../../plans/topic_4/diagrams/transformer-architecture.mmd)

The left stack (Encoder) reads the French complaint and builds a rich contextual
representation of every token. The right stack (Decoder) generates the English
translation one token at a time, attending to both its own previous outputs
(causal self-attention) and the full encoder output (cross-attention).
```

---

### Cell 11: markdown - Lab 1 Header (Tier 1: Positional Encoding)

```
## Lab 1 - Add Positional Encoding to a Complaint Embedding (Tier 1 - Guided)

**Time**: 15-20 minutes

### Situation
The Barclays NLP platform team has token embeddings for incoming complaint messages.
Without positional encoding, the encoder cannot distinguish "account was charged"
from "charged was account". Your job is to add sinusoidal positional encoding.

### Task
Complete the `PositionalEncodingLab` class using the formula from Cell 9.

### Action
Fill in three stubs in `__init__` (position tensor, div_term, pe matrix)
and one stub in `forward` (add pe to x).

### Result
The verification cell will check:
1. Output shape matches input shape
2. Encoding at position 0 differs from position 1
3. Deterministic with dropout=0 (same input -> same output)
4. `pe` is registered as a buffer
```

---

### Cell 12: code - Lab 1 Starter Code

```python
# Lab 1: Complete the PositionalEncodingLab class.

class PositionalEncodingLab(nn.Module):
    """
    Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model : embedding dimension
        dropout : dropout on output (default 0.1)
        max_len : maximum sequence length to precompute (default 512)

    Forward:
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Step 1: Create position indices of shape (max_len, 1)
        position = None  # YOUR CODE

        # Step 2: Create frequency terms of shape (d_model//2,)
        # Hint: use torch.exp and torch.arange(0, d_model, 2)
        div_term = None  # YOUR CODE

        # Step 3: Build the pe matrix of shape (max_len, d_model),
        # then add a batch dimension -> (1, max_len, d_model) and register as buffer.
        pe = None  # YOUR CODE
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Step 4: Add positional encoding to x and apply dropout.
        x = None  # YOUR CODE
        return x

# Quick shape check
try:
    _lab = PositionalEncodingLab(d_model=64, dropout=0.0, max_len=128)
    _x   = torch.zeros(2, 20, 64)
    _out = _lab(_x)
    if _out is not None:
        print(f"Output shape: {_out.shape}  (expected: torch.Size([2, 20, 64]))")
    else:
        print("forward returned None -- complete Step 4.")
except Exception as e:
    print(f"Error: {e}")
```

---

### Cell 13: code - Lab 1 Verification

```python
# Lab 1 Verification

set_seeds(42)
ref_pe = PositionalEncoding(d_model=64, dropout=0.0, max_len=128)
lab_pe = PositionalEncodingLab(d_model=64, dropout=0.0, max_len=128)
x_test = torch.randn(3, 15, 64)

all_pass = True

try:
    lab_out = lab_pe(x_test)
    ref_out = ref_pe(x_test)

    if lab_out is None:
        print("FAIL: forward returned None. Complete all steps.")
        all_pass = False
    else:
        if lab_out.shape == x_test.shape:
            print(f"PASS: Output shape {lab_out.shape}")
        else:
            print(f"FAIL: Expected {x_test.shape}, got {lab_out.shape}")
            all_pass = False

        if torch.allclose(lab_out, ref_out, atol=1e-5):
            print("PASS: Output matches reference implementation")
        else:
            diff = (lab_out - ref_out).abs().max().item()
            print(f"FAIL: Max diff from reference: {diff:.6f}")
            all_pass = False

        if hasattr(lab_pe, "pe") and isinstance(lab_pe.pe, torch.Tensor):
            print("PASS: 'pe' registered as buffer")
        else:
            print("FAIL: 'pe' not registered as buffer -- use self.register_buffer('pe', pe)")
            all_pass = False

        pe_pos0 = lab_pe.pe[0, 0, :]
        pe_pos1 = lab_pe.pe[0, 1, :]
        if not torch.allclose(pe_pos0, pe_pos1, atol=1e-4):
            print("PASS: Encoding at position 0 differs from position 1")
        else:
            print("FAIL: Position 0 and position 1 are identical -- check formula")
            all_pass = False

except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    all_pass = False

if all_pass:
    print()
    print("All Lab 1 checks passed.")
```

---

### Cell 14: code - Lab 1 Safety-Net

```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
_need_sn = False
try:
    _m = PositionalEncodingLab(d_model=32, dropout=0.0)
    _x = torch.zeros(1, 10, 32)
    _o = _m(_x)
    if _o is None:
        _need_sn = True
except Exception:
    _need_sn = True

if _need_sn:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    PositionalEncodingLab = PositionalEncoding
```

---

### Cell 15: markdown - Lab 1 Stretch and Homework

```
### Stretch (fast finishers)

Visualise your `PositionalEncodingLab` output using seaborn. Plot a heatmap of
shape (50, 64) with seq_len=50, d_model=64. Verify it matches Cell 9.

### Homework Extension

Implement LEARNED positional encoding as an alternative to sinusoidal:

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos_emb(positions)
```

Does learned PE generalise to sequences longer than max_len at test time?
What happens when you try to encode position 513 with max_len=512?
```

---

### Cell 16: markdown - Section 2: Multi-Head Attention

```
## Section 2 - Multi-Head Attention

Scaled dot product attention (Topic 3b) runs ONE attention computation over
the full d_model embedding space.

Multi-head attention runs H PARALLEL scaled dot product attentions, each in a
SMALLER subspace (d_k = d_model / H), then concatenates and projects the results.

Why H heads instead of one big one?

- Different heads can specialise: one head attends to syntactic relationships,
  another to coreference, another to domain-specific patterns (fraud indicators).
- Each head projects Q, K, V through its own learned weight matrices.
- The concatenated output is projected back to d_model via a learned W_O matrix.

Formula:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
    head_i = ScaledDotProductAttention(Q W_Q_i, K W_K_i, V W_V_i)
```

---

### Cell 17: code - Beat 1: Single Head vs Multi-Head -- Attention Diversity

```python
# Beat 1: A single attention head produces ONE attention pattern over d_model dimensions.
# Multi-head produces H different patterns in d_k-dimensional subspaces.
# We visualise this difference on the complaint token vocabulary.

set_seeds(42)

n_tokens  = len(COMPLAINT_TOKENS)   # 8
d_model   = 128
nhead     = 4

single_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=1,
                                    dropout=0.0, batch_first=True)
multi_mha  = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                    dropout=0.0, batch_first=True)
single_mha.eval()
multi_mha.eval()

# Simulate complaint embeddings: two semantic clusters
torch.manual_seed(42)
fraud_cluster   = torch.randn(d_model)
account_cluster = torch.randn(d_model)
embeddings = torch.zeros(n_tokens, d_model)
for i in [0, 1, 3]:   # unauthorised, charge, fraud
    embeddings[i] = fraud_cluster + 0.2 * torch.randn(d_model)
for i in [2, 4, 5]:   # account, refund, dispute
    embeddings[i] = account_cluster + 0.2 * torch.randn(d_model)
embeddings[6] = torch.randn(d_model)
embeddings[7] = torch.randn(d_model)

x = embeddings.unsqueeze(0)   # (1, 8, 128)

with torch.no_grad():
    _, single_w = single_mha(x, x, x)   # (1, 8, 8)
    _, multi_w  = multi_mha(x, x, x)    # (1, 8, 8) averaged across heads

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, weights, title in [
    (axes[0], single_w[0].numpy(), "Single-Head Attention (1 head, d_k=128)"),
    (axes[1], multi_w[0].numpy(),  "Multi-Head Attention (4 heads, d_k=32, averaged)"),
]:
    sns.heatmap(weights, ax=ax,
                xticklabels=COMPLAINT_TOKENS,
                yticklabels=COMPLAINT_TOKENS,
                cmap="Blues", annot=True, fmt=".2f", linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()
print()
print("The averaged multi-head weights differ from single-head because each head")
print("attends to a different subspace. In practice heads are NOT averaged --")
print("they are CONCATENATED then projected, preserving all H patterns.")
```

---

### Cell 18: markdown - Beat 2: Multi-Head Attention Diagram

```
<!-- DIAGRAM: Multi-head attention mechanism. Input Q, K, V vectors (d_model=128) each split into H=4 parallel heads. For each head i: linear projection W_Q_i produces (batch, seq, 32); same for W_K_i and W_V_i. Each head then runs Scaled Dot Product Attention (from Topic 3b) producing (batch, seq, 32). All 4 head outputs concatenated -> (batch, seq, 128). Final linear projection W_O maps (batch, seq, 128) -> (batch, seq, 128). Arrows show the split -> 4 parallel attention boxes -> concat -> project flow. Labels: d_model=128, H=4, d_k=d_v=32. -->
[View diagram](../../plans/topic_4/diagrams/multi-head-attention.mmd)

Each head runs the SAME scaled dot product attention from Topic 3b,
but in a 32-dimensional subspace rather than 128 dimensions.
Head 1 might specialise in fraud-related token correlations;
Head 2 might capture syntactic structure; Head 3 sentiment markers.
The W_O projection at the end combines all four views.
```

---

### Cell 19: code - Beat 3: MultiHeadAttention from Scratch (Reference Demo)

```python
# Beat 3: Full multi-head attention built from scratch.
# Explicitly shows the split/concat/project pattern.
# nn.MultiheadAttention does the same thing, more efficiently.

class ScaledDotProductAttn(nn.Module):
    """Scaled dot product attention -- same as Topic 3b reference."""
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # query/key/value: (batch, H, seq, d_k)
        d_k = query.size(-1)
        scores  = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, value)
        return context, weights


class MultiHeadAttentionDemo(nn.Module):
    """
    Multi-head attention built from H ScaledDotProductAttn heads.

    Args:
        d_model  : total embedding dimension (must be divisible by num_heads)
        num_heads: number of parallel attention heads
        dropout  : attention dropout probability
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProductAttn(dropout_p=dropout)

    def _split_heads(self, x):
        """(batch, seq, d_model) -> (batch, H, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """(batch, H, seq, d_k) -> (batch, seq, d_model)"""
        batch, H, seq, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, H * d_k)

    def forward(self, query, key, value):
        """
        Args:
            query: (batch, T_q, d_model)
            key:   (batch, T_k, d_model)
            value: (batch, T_k, d_model)
        Returns:
            output:      (batch, T_q, d_model)
            attn_weights: (batch, H, T_q, T_k)
        """
        Q = self._split_heads(self.W_q(query))   # (batch, H, T_q, d_k)
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        context, attn_weights = self.attn(Q, K, V)   # (batch, H, T_q, d_k)

        context = self._merge_heads(context)          # (batch, T_q, d_model)
        output  = self.W_o(context)
        return output, attn_weights


# --- Demo ---
set_seeds(42)

d_model   = 128
num_heads = 4
T_seq     = len(COMPLAINT_TOKENS)   # 8

mha_demo = MultiHeadAttentionDemo(d_model=d_model, num_heads=num_heads)
mha_demo.eval()

Q_in = torch.randn(2, T_seq, d_model)
output, weights = mha_demo(Q_in, Q_in, Q_in)

print("MultiHeadAttentionDemo output")
print("=" * 45)
print(f"Input shape:          {Q_in.shape}")
print(f"Output shape:         {output.shape}      -> (batch=2, seq=8, d=128)")
print(f"Attention weights:    {weights.shape}  -> (batch=2, H=4, T_q=8, T_k=8)")
print()

params = sum(p.numel() for p in mha_demo.parameters())
print(f"Total parameters: {params:,}")
print(f"Breakdown: 4 x Linear({d_model},{d_model}) = 4 x {d_model*d_model} = {4*d_model*d_model}")
print()

# Compare with nn.MultiheadAttention
builtin_mha    = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                       dropout=0.0, bias=False, batch_first=True)
builtin_params = sum(p.numel() for p in builtin_mha.parameters())
print(f"nn.MultiheadAttention parameters (bias=False): {builtin_params:,}")
print("Same count. Our implementation uses identical weight structure.")
```

---

### Cell 20: markdown - Lab 2 Header (Tier 2: Hard -- Full Transformer Forward Pass)

```
## Lab 2 - Build the Full Transformer Forward Pass (Tier 2 - Hard)

**Time**: 25-35 minutes | **This is the Day 2 Tier 2 hard lab**

### Situation
The Barclays NLP infrastructure team needs a clean, inspectable Transformer
encoder-decoder that they can audit for compliance. The PyTorch nn.Transformer
is a black box for compliance purposes. Your team needs to build the forward pass
explicitly from EncoderLayer and DecoderLayer building blocks.

### Task
Complete the `TransformerModel` class. The `__init__` is provided.
Implement `forward`: build the causal mask, embed and encode the source,
decode the target (teacher-forced), and return logits.

### Action
1. In `forward`, build a causal (upper-triangular) mask for the target sequence.
2. Embed and positionally encode both src and tgt (use tgt[:, :-1] as decoder input).
3. Run src through the encoder stack (self.encoder_layers).
4. Run tgt through the decoder stack (self.decoder_layers), passing memory.
5. Apply self.output_proj and return logits of shape (batch, tgt_len-1, tgt_vocab).

No `# YOUR CODE` placeholders -- you decide where each step goes.

### Result
The verification cell checks:
1. Output shape is (batch, tgt_len-1, tgt_vocab_size)
2. Logits are finite (no NaN/Inf)
3. Gradients flow from output back through the model
```

---

### Cell 21: code - Lab 2 Starter Code (Tier 2)

```python
# Lab 2: Implement the TransformerModel forward pass.
# __init__ is provided. Implement forward().

class TransformerModel(nn.Module):
    """
    Full encoder-decoder Transformer for translation.

    Architecture:
        Encoder: PositionalEncoding -> N x TransformerEncoderLayer -> LayerNorm
        Decoder: PositionalEncoding -> N x TransformerDecoderLayer -> LayerNorm -> Linear

    Uses batch_first=True convention throughout.

    Args:
        src_vocab_size : source vocabulary size
        tgt_vocab_size : target vocabulary size
        d_model        : embedding dimension
        nhead          : number of attention heads (must divide d_model)
        num_enc_layers : number of encoder layers
        num_dec_layers : number of decoder layers
        dim_ff         : feedforward hidden size
        dropout        : dropout probability
        max_seq_len    : maximum sequence length for positional encoding
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4,
                 num_enc_layers=2, num_dec_layers=2, dim_ff=512,
                 dropout=0.1, max_seq_len=128):
        super().__init__()
        self.d_model = d_model

        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq_len + 10)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_ff,
                                       dropout=dropout, batch_first=True)
            for _ in range(num_enc_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_ff,
                                       dropout=dropout, batch_first=True)
            for _ in range(num_dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        """
        Args:
            src : (batch, src_len) -- source token ids
            tgt : (batch, tgt_len) -- target token ids (including BOS)
                  Decoder input is tgt[:, :-1]; labels are tgt[:, 1:]
            src_key_padding_mask : (batch, src_len) bool, True = PAD
            tgt_key_padding_mask : (batch, tgt_len-1) bool, True = PAD

        Returns:
            logits: (batch, tgt_len-1, tgt_vocab_size)
        """
        pass
```

---

### Cell 22: code - Lab 2 Verification

```python
# Lab 2 Verification

set_seeds(42)

SRC_VOCAB = 1000
TGT_VOCAB = 1200
D_MODEL   = 128
BATCH     = 4
SRC_LEN   = 20
TGT_LEN   = 18

model_lab = TransformerModel(
    src_vocab_size=SRC_VOCAB,
    tgt_vocab_size=TGT_VOCAB,
    d_model=D_MODEL,
    nhead=4,
    num_enc_layers=2,
    num_dec_layers=2,
    dim_ff=256,
    dropout=0.0,
    max_seq_len=64,
)

src_ids = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))
tgt_ids = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN))

all_pass = True

try:
    logits = model_lab(src_ids, tgt_ids)

    if logits is None:
        print("FAIL: forward returned None -- implement the method body.")
        all_pass = False
    else:
        expected_shape = torch.Size([BATCH, TGT_LEN - 1, TGT_VOCAB])
        if logits.shape == expected_shape:
            print(f"PASS: Logits shape {logits.shape}")
        else:
            print(f"FAIL: Expected {expected_shape}, got {logits.shape}")
            print("Hint: decoder input is tgt[:, :-1]; output is tgt[:, 1:]")
            all_pass = False

        if torch.isfinite(logits).all():
            print("PASS: All logits are finite (no NaN/Inf)")
        else:
            print("FAIL: Logits contain NaN or Inf")
            all_pass = False

        logits2 = model_lab(src_ids, tgt_ids)
        if logits2 is not None:
            logits2.sum().backward()
            has_grad = any(p.grad is not None for p in model_lab.parameters())
            if has_grad:
                print("PASS: Gradients flow through the model")
            else:
                print("FAIL: No parameter gradients")
                all_pass = False

except NotImplementedError:
    print("FAIL: forward raises NotImplementedError -- remove pass and implement")
    all_pass = False
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    all_pass = False

if all_pass:
    print()
    print("All Lab 2 checks passed.")
    print("You have assembled a complete Transformer encoder-decoder forward pass.")
```

---

### Cell 23: code - Lab 2 Safety-Net

```python
# Lab 2 safety-net: run this if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2.

_need_sn2 = False
try:
    _m2  = TransformerModel(src_vocab_size=100, tgt_vocab_size=100)
    _src = torch.randint(1, 100, (2, 10))
    _tgt = torch.randint(1, 100, (2, 8))
    _out = _m2(_src, _tgt)
    if _out is None:
        _need_sn2 = True
except Exception:
    _need_sn2 = True

if _need_sn2:
    print("Using Lab 2 safety-net so the rest of the notebook can run.")

    class TransformerModel(nn.Module):
        """Safety-net: working TransformerModel for downstream cells."""

        def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4,
                     num_enc_layers=2, num_dec_layers=2, dim_ff=512,
                     dropout=0.1, max_seq_len=128):
            super().__init__()
            self.d_model = d_model
            self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
            self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
            self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq_len + 10)
            self.encoder_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_ff,
                                           dropout=dropout, batch_first=True)
                for _ in range(num_enc_layers)
            ])
            self.encoder_norm  = nn.LayerNorm(d_model)
            self.decoder_layers = nn.ModuleList([
                nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_ff,
                                           dropout=dropout, batch_first=True)
                for _ in range(num_dec_layers)
            ])
            self.decoder_norm  = nn.LayerNorm(d_model)
            self.output_proj   = nn.Linear(d_model, tgt_vocab_size)
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, src, tgt,
                    src_key_padding_mask=None,
                    tgt_key_padding_mask=None):
            tgt_input = tgt[:, :-1]
            src_x = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
            tgt_x = self.pos_enc(self.tgt_emb(tgt_input) * math.sqrt(self.d_model))
            tgt_len   = tgt_x.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=src.device
            )
            memory = src_x
            for layer in self.encoder_layers:
                memory = layer(memory, src_key_padding_mask=src_key_padding_mask)
            memory = self.encoder_norm(memory)
            out = tgt_x
            for layer in self.decoder_layers:
                out = layer(out, memory,
                            tgt_mask=causal_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=src_key_padding_mask)
            out = self.decoder_norm(out)
            return self.output_proj(out)
```

---

### Cell 24: markdown - Lab 2 Stretch and Homework

```
### Stretch (fast finishers)

Pass the padding masks through your `TransformerModel.forward`. The method signature
already accepts `src_key_padding_mask` and `tgt_key_padding_mask`. Pass them through
to the encoder and decoder layers. Verify that a sequence with all-PAD tokens beyond
position 5 produces logits identical to the unmasked version for positions 0-4.

### Homework Extension

Add a beam search decoder to your `TransformerModel`. Beam search maintains the top-K
most probable partial translations at each step instead of greedy selection.

```python
def beam_decode(model, src, bos_idx, eos_idx, max_len=50, beam_size=4):
    """
    Beam search decoder.

    Args:
        model    : TransformerModel in eval mode
        src      : (1, src_len) source token ids
        bos_idx  : index of the <bos> token
        eos_idx  : index of the <eos> token
        max_len  : maximum output length
        beam_size: number of beams to maintain

    Returns:
        best_seq: list of token ids for the best translation
    """
    pass  # implement for homework
```

Compare BLEU score of beam_size=4 vs greedy decoding on 100 validation pairs.
```

---

### Cell 25: markdown - Section 3: Toy Training Loop

```
## Section 3 - Assembling the Full Translator: Toy Verification

You now have all the building blocks. Before launching the real GPU job,
we run 5 training steps in the notebook on random data to confirm that
the forward/backward pass works and that loss decreases.

The real training (French->English, 50k pairs, 5 epochs) runs as a remote
SageMaker job in Section 4.
```

---

### Cell 26: code - Beat 3: Toy Translator Training Loop

```python
# Beat 3: Run 5 training steps on random data.
# Loss decreasing confirms forward/backward are correct.
# This is NOT the real training job.

set_seeds(42)

TOY_SRC_VOCAB = 200
TOY_TGT_VOCAB = 200
D_MODEL       = 64
NHEAD         = 4
BATCH         = 8
SRC_LEN       = 16
TGT_LEN       = 14

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

toy_model = TransformerModel(
    src_vocab_size=TOY_SRC_VOCAB,
    tgt_vocab_size=TOY_TGT_VOCAB,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_enc_layers=2,
    num_dec_layers=2,
    dim_ff=128,
    dropout=0.1,
    max_seq_len=64,
).to(device)

optimizer = torch.optim.Adam(toy_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

total_params = sum(p.numel() for p in toy_model.parameters())
print(f"Toy TransformerModel: {total_params:,} parameters")
print(f"d_model={D_MODEL}, nhead={NHEAD}, 2 enc + 2 dec layers")
print()
print("Running 5 toy training steps (random data) ...")
print(f"{'Step':>6}  {'Loss':>10}")
print("-" * 22)

for step in range(5):
    toy_model.train()
    src_ids = torch.randint(3, TOY_SRC_VOCAB, (BATCH, SRC_LEN)).to(device)
    tgt_ids = torch.cat([
        torch.full((BATCH, 1), BOS_IDX),
        torch.randint(3, TOY_TGT_VOCAB, (BATCH, TGT_LEN - 1)),
    ], dim=1).to(device)

    optimizer.zero_grad()
    logits = toy_model(src_ids, tgt_ids)   # (batch, TGT_LEN-1, TOY_TGT_VOCAB)
    labels = tgt_ids[:, 1:]                 # (batch, TGT_LEN-1)
    loss   = criterion(logits.reshape(-1, TOY_TGT_VOCAB), labels.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(toy_model.parameters(), 1.0)
    optimizer.step()

    print(f"{step + 1:>6}  {loss.item():>10.4f}")

print()
print("Loss decreasing confirms the forward/backward pass is correct.")
print("For real training (French->English, 50k pairs, 5 epochs) -> Section 4.")
```

---

### Cell 27: markdown - Discussion: Architecture Design Choices

```
### Discussion (3 minutes)

You have assembled a complete Transformer encoder-decoder.

1. The paper uses 6 encoder layers + 6 decoder layers with d_model=512.
   We use 2 + 2 with d_model=128 for the GPU job (under 30 min on T4).
   What would you expect to happen to translation quality if you tripled
   the depth? What would happen to training time? Is this trade-off
   acceptable for a production system with strict latency requirements?

2. The Decoder has THREE sublayers per layer: masked self-attention,
   cross-attention (attending to the encoder), and feed-forward.
   The Encoder has only TWO sublayers. Why does the decoder need a third?
   What would break if you removed the cross-attention from the decoder?

3. In the Barclays complaint translation context: the system translates
   a French complaint to English so the triage model can route it.
   If the translator makes an error that changes the sentiment from
   "furious" to "mildly unhappy", what is the downstream impact?
   How would you measure and monitor this in production?
```

---

### Cell 28: markdown - Section 4: SageMaker GPU Training Job

```
## Section 4 - SageMaker GPU Training Capstone

You have verified the architecture in the notebook. Now you will train a real
French-to-English complaint translator on the full opus_books en-fr dataset
using a GPU instance.

This section shows you:
1. How the training code is packaged in scripts_topic4/
2. How to launch a SageMaker PyTorch estimator job
3. How to monitor the job and retrieve BLEU score from CloudWatch logs

### The training script: scripts_topic4/train.py

The script packages everything you built today:
- The same PositionalEncoding, TransformerTranslator, and training loop
- Hyperparameters passed as CLI arguments (SageMaker estimator convention)
- sacrebleu for BLEU evaluation (not evaluate library -- L6 from lessons learned)
- Checkpoint saving to /opt/ml/model/ (SageMaker copies this to S3 after the job)

### Hardware
- Instance: ml.g4dn.xlarge (NVIDIA T4, 16GB VRAM, ~$0.74/hr)
- Expected training time: 15-25 minutes for 5 epochs on 50k pairs
- Dataset: opus_books en-fr from HuggingFace datasets (auto-downloaded in the job)
```

---

### Cell 29: code - Create scripts_topic4/ and requirements.txt

```python
# Create the scripts_topic4/ directory with requirements.txt.
# train.py must be pre-staged by the instructor (see Section 4 plan content).
# The filename MUST be "requirements.txt" exactly (L4 from SAGEMAKER_LESSONS_LEARNED).

import os

os.makedirs("scripts_topic4", exist_ok=True)

req_content = "sacrebleu\ndatasets==2.18.0\nnumpy<2\n"
with open("scripts_topic4/requirements.txt", "w") as f:
    f.write(req_content)

print("scripts_topic4/requirements.txt written:")
print(req_content)

if os.path.isfile("scripts_topic4/train.py"):
    size = os.path.getsize("scripts_topic4/train.py")
    print(f"scripts_topic4/train.py found ({size} bytes)  READY")
else:
    print("scripts_topic4/train.py not found.")
    print("Copy train.py from the course materials to scripts_topic4/ before Cell 30.")
```

---

### Cell 30: code - Beat 3: Launch the SageMaker GPU Training Job

```python
# Beat 3: Launch the GPU training job via the PyTorch estimator.
#
# Key constraints from SAGEMAKER_LESSONS_LEARNED:
# L1: PyTorch estimator, NOT HuggingFace -- our model is custom; HF estimator is GPU-only
#     AND requires HF Hub API which we do not need.
# L2: framework_version="2.8.0" requires py_version="py312"
# L3: sagemaker SDK must be <3.0.0

from sagemaker.pytorch import PyTorch
import time

hyperparameters = {
    "d_model":             128,
    "nhead":               4,
    "num_encoder_layers":  2,
    "num_decoder_layers":  2,
    "dim_feedforward":     512,
    "dropout":             0.1,
    "max_seq_len":         64,
    "batch_size":          128,
    "epochs":              5,
    "lr":                  1e-3,
    "max_pairs":           50000,
    "vocab_size":          8000,
    "seed":                42,
}

estimator = PyTorch(
    entry_point="train.py",
    source_dir="scripts_topic4",
    role=role,
    framework_version="2.8.0",      # L2: only version that supports py312 in us-west-2
    py_version="py312",             # L2: py311 not supported for 2.8.0
    instance_type="ml.g4dn.xlarge", # NVIDIA T4 GPU, cheapest GPU instance
    instance_count=1,
    hyperparameters=hyperparameters,
    sagemaker_session=sess,
    disable_profiler=True,
    debugger_hook_config=False,
)

job_name = f"transformer-translator-{int(time.time())}"

print(f"Launching training job: {job_name}")
print(f"Instance:  ml.g4dn.xlarge (NVIDIA T4, ~$0.74/hr)")
print(f"Estimated: 15-25 minutes")
print()

estimator.fit(wait=False, job_name=job_name)

training_job_name = estimator.latest_training_job.name
print(f"Job launched: {training_job_name}")
print(f"Monitor at: https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs/{training_job_name}")
```

---

### Cell 31: code - Monitor Job Status

```python
# Monitor the training job status.
# L7 from SAGEMAKER_LESSONS_LEARNED: use ResourceNotFound (NOT ResourceNotFoundException).

import boto3

sm_client = boto3.client("sagemaker", region_name=region)

def get_job_status(job_name):
    try:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        return resp["TrainingJobStatus"], resp.get("SecondaryStatus", "")
    except sm_client.exceptions.ResourceNotFound:
        return "NotFound", ""

print(f"Monitoring job: {training_job_name}")
print("(Re-run this cell to refresh status)")
print()

status, secondary = get_job_status(training_job_name)
print(f"Status:           {status}")
print(f"Secondary status: {secondary}")
print()

if status == "InProgress":
    print("Job is running. Estimated completion: 15-25 minutes from launch.")
    print("While you wait: review the architecture summary in Cell 32.")
elif status == "Completed":
    print("Job complete. Proceed to Cell 33 to retrieve results.")
elif status == "Failed":
    print("Job failed. Check CloudWatch logs:")
    print(f"  Log group:  /aws/sagemaker/TrainingJobs")
    print(f"  Log stream: {training_job_name}/algo-1-*")
else:
    print(f"Status: {status}")
```

---

### Cell 32: markdown - Architecture Summary (While the Job Runs)

```
## Architecture Summary (review while the GPU job runs)

### What you built today

| Component | Key detail |
|-----------|------------|
| PositionalEncoding | Sinusoidal PE, batch_first=True, registered buffer, generalises to unseen lengths |
| MultiHeadAttentionDemo | Explicit split/concat/project, H=4 heads, d_k=32 per head |
| TransformerModel | 2 enc + 2 dec layers, causal mask, teacher forcing, encoder norm + decoder norm |
| train.py | Full GPU training, sacrebleu BLEU, word-level vocab, checkpoint to /opt/ml/model/ |

### The four sublayers of a Transformer

1. Multi-head self-attention (encoder) or masked multi-head self-attention (decoder):
   each token attends to all others (encoder) or past tokens only (decoder).

2. Cross-attention (decoder only): each decoder position attends to the full encoder
   output. This is the bridge between source and target languages.

3. Feed-forward (both): two linear layers with ReLU activation, applied point-wise.
   Expands to 4 x d_model then contracts back: Linear(d,4d) -> ReLU -> Linear(4d,d).

4. Add + LayerNorm (after each sublayer): residual connection prevents vanishing
   gradients in deep stacks. LayerNorm stabilises training.

### Why French->English for Barclays

Barclays serves customers in the Channel Islands and France.
A complaint submitted as "Mon compte a ete debite sans autorisation" must be
translated to "My account was debited without authorisation" before the
Day 1 triage system can classify it as high-severity fraud.
The Transformer handles long, complex complaint sentences far better than
the LSTM seq2seq from Topic 3a.
```

---

### Cell 33: code - Retrieve Job Results (Run After Job Completes)

```python
# Retrieve job results. Will error if the job has not finished yet.

sm_client = boto3.client("sagemaker", region_name=region)

try:
    resp   = sm_client.describe_training_job(TrainingJobName=training_job_name)
    status = resp["TrainingJobStatus"]
    print(f"Job status: {status}")

    if status != "Completed":
        print(f"Job is {status}. Run this cell again after the job completes.")
    else:
        model_data_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
        start          = resp["TrainingStartTime"]
        end            = resp["TrainingEndTime"]
        duration_min   = (end - start).total_seconds() / 60
        print(f"Model artifacts:  {model_data_uri}")
        print(f"Training duration: {duration_min:.1f} minutes")
        print()
        print("The final BLEU score was logged to CloudWatch during training.")
        print("A BLEU of 8-15 is expected for 5 epochs on 50k pairs with d_model=128.")
        print("For reference: a production-quality NMT system scores 30+ BLEU.")

except sm_client.exceptions.ResourceNotFound:
    print(f"Job {training_job_name} not found.")
```

---

### Cell 34: code - Retrieve BLEU Score from CloudWatch Logs

```python
# Parse BLEU scores from CloudWatch Logs.
# train.py prints "Epoch N/M | loss=X | BLEU=Y" to stdout.
# SageMaker captures stdout to CloudWatch automatically.

logs_client = boto3.client("logs", region_name=region)

log_group          = "/aws/sagemaker/TrainingJobs"
log_stream_prefix  = training_job_name + "/algo-1-"

print(f"Fetching BLEU scores from CloudWatch ...")
print(f"  Log group:  {log_group}")
print(f"  Log stream: {log_stream_prefix}*")
print()

try:
    streams_resp = logs_client.describe_log_streams(
        logGroupName=log_group,
        logStreamNamePrefix=log_stream_prefix,
        orderBy="LogStreamName",
    )
    streams = streams_resp.get("logStreams", [])

    if not streams:
        print("No log streams found yet. The job may still be initialising.")
    else:
        stream_name = streams[0]["logStreamName"]
        events_resp = logs_client.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            startFromHead=True,
        )
        epoch_lines = [
            e["message"]
            for e in events_resp.get("events", [])
            if "BLEU=" in e.get("message", "")
        ]
        if epoch_lines:
            print("Training progress (from CloudWatch):")
            for line in epoch_lines:
                print(f"  {line.strip()}")
        else:
            print("No BLEU log lines found yet. The job may still be running.")

except Exception as e:
    print(f"Could not fetch logs: {e}")
    print("Check CloudWatch manually via the AWS console.")
```

---

### Cell 35: markdown - Wrap-Up and Bridge to Topic 5

```
## Wrap-Up

### What you built in Topic 4

| Component | Key insight |
|-----------|-------------|
| Positional encoding | Without it the Transformer is order-blind. Sinusoidal PE gives every position a unique fingerprint that generalises to unseen lengths. |
| Multi-head attention | H parallel attention heads in d_k-dimensional subspaces, each specialising in different relational patterns. |
| TransformerModel | Complete encoder-decoder: causal masking, teacher forcing, cross-attention connecting the two stacks, LayerNorm after each sublayer. |
| SageMaker GPU job | PyTorch estimator + scripts_topic4/train.py + ml.g4dn.xlarge. First remote GPU training job of the course. |

### Key architectural facts

- Transformer encoder layer: O(n^2) attention cost, O(1) sequential steps.
  RNN: O(n) sequential steps but O(n) memory. For n < 512 the Transformer is
  faster on GPU; for n > 100k use specialised approximations (Longformer, BigBird).

- Layer normalisation (Add+Norm) is critical for training deep stacks.
  Without it, gradients vanish through 6+ layers of attention + feed-forward.

- Teacher forcing (feeding ground truth as decoder input during training) speeds
  up training but creates exposure bias: at inference the model feeds its own
  predictions, which may drift from the training distribution.

### What is coming next

- Topic 5: Transfer learning -- fine-tune a pre-trained DistilBERT for the
  Barclays complaint classifier instead of training from scratch.
- Topic 6: Full fine-tuning vs frozen encoder.
- Topic 7: PEFT/LoRA -- parameter-efficient fine-tuning for large models.
```

---

### Cell 36: markdown - Homework Extensions

```
## Homework Extensions

### Homework 1: Beam Search Decoder

The capstone train.py uses greedy decoding for BLEU evaluation.
Implement beam search (beam_size=4) as a method on TransformerModel
and compare BLEU scores on 200 validation pairs.

Expected result: beam search typically improves BLEU by 1-3 points over greedy
for a small model (d_model=128, 2 layers).

### Homework 2: Learning Rate Warm-up

The original Transformer paper uses a warm-up schedule:
    lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

Add this scheduler to train.py and relaunch with warmup=4000.
Compare training loss curves in CloudWatch.

```python
# Homework 2 starter: custom warm-up scheduler
class TransformerScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step  = max(1, self.last_epoch)
        scale = (self.d_model ** -0.5) * min(step ** -0.5,
                                              step * self.warmup_steps ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]
```

### Homework 3: Visualise Cross-Attention

Modify `TransformerDecoderLayer` to capture the cross-attention weights.
For a translated sentence pair, plot a heatmap (English tokens as rows,
French tokens as columns). A well-trained model should show high cross-attention
between semantically corresponding words:
"compte" (account) -> "account", "fraude" -> "fraud".
```

---

### Cell 37: markdown - End of Notebook Marker

```
---

*End of Topic 4 - Transformers + Translator Capstone*

Next session: Topic 5 - Transfer Learning with DistilBERT.
Fine-tune a pre-trained model for Barclays complaint classification.
```

---

## Implementation Notes for /build-topic-notebook

1. **Output paths**:
   - Exercise: `Exercises/topic_4_transformers/topic_4_transformers.ipynb`
   - Solution:  `Solutions/topic_4_transformers/topic_4_transformers.ipynb`

2. **scripts_topic4/ directory**: Located relative to the notebook file at
   `Exercises/topic_4_transformers/scripts_topic4/`. Cell 29 writes requirements.txt;
   train.py must be pre-staged by the instructor before Cell 30 is run.

3. **Total cells**: 37 cells as planned (markdown + code). The 5-cell approval cadence
   applies: stop after every 5 cells, await approval before continuing.

4. **Lab tiers**:
   - Lab 1 (Cell 12): Tier 1 guided -- 4 stubs, numbered steps in the header markdown.
   - Lab 2 (Cell 21): Tier 2 hard -- no `# YOUR CODE` placeholders, no numbered steps
     in the code. The student produces the entire `forward` body independently.
   - No Tier 3 in this notebook (Tier 3 reserved for last topic of Day 2).

5. **Safety-nets**: Cell 14 (Lab 1) and Cell 23 (Lab 2) are required. Remove both
   from the solution notebook. The Lab 2 safety-net (Cell 23) contains a FULL working
   `TransformerModel` implementation so students who cannot finish Lab 2 still get
   a working model for the SageMaker capstone.

6. **SageMaker job cells (30-34) must run SEQUENTIALLY** -- each depends on
   `training_job_name` from Cell 30. In class: launch the job early, teach
   Sections 1-3 while the job runs, then return to Cells 33-34 for results.

7. **AI-tells enforcement**: No em dashes, no en dashes, no Unicode multiplication,
   no emojis. Plain ASCII only in all cell content, print statements, and markdown headers.

8. **No evaluate library** anywhere in the notebook or train.py. BLEU uses sacrebleu.
   This follows L6 from SAGEMAKER_LESSONS_LEARNED.

9. **No getpass** for AWS credentials -- the SageMaker execution role handles auth
   automatically inside Studio. No external API keys needed for this notebook.

10. **Discussion cells** (Cells 6, 27) should be run by the instructor as class
    exercises with a visible 3-minute timer. Do not skip them.

11. **Variable `training_job_name`** is set in Cell 30 and used in Cells 31, 33, 34.
    If the kernel is restarted after Cell 30, students must re-run Cell 30 or manually
    set `training_job_name = "<the-job-name-from-the-console>"`.

12. **The `PositionalEncoding` class** defined in Cell 9 is used in Cells 12, 14 (safety-net
    fallback), and in the safety-net TransformerModel inside Cell 23. It must be defined
    before any of those cells run.

13. **train.py is the definitive capstone artifact** -- the full source code is
    in the "Source Dir" section above this cell plan. It must be placed at
    `Exercises/topic_4_transformers/scripts_topic4/train.py` before Cell 30 runs.
