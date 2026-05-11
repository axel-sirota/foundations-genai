import argparse
import math
import os
import random
import torch
import torch.nn as nn
import numpy as np


def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seeds(42)


def positional_encoding(max_len, d_model):
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    half_d = d_model // 2
    dim_idx = torch.arange(half_d, dtype=torch.float32).unsqueeze(0)
    angle_rates = 1.0 / (10000.0 ** (dim_idx / d_model))
    angles = positions * angle_rates
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.register_buffer("pe", positional_encoding(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return self.embedding(x) * math.sqrt(self.d_model) + self.pe[:seq_len]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embed = PositionalEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = self.embed(src)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_kpm=None, mem_kpm=None):
        a, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_kpm)
        tgt = self.norm1(tgt + self.dropout(a))
        a, _ = self.cross_attn(tgt, memory, memory, key_padding_mask=mem_kpm)
        tgt = self.norm2(tgt + self.dropout(a))
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embed = PositionalEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, tgt_kpm=None, mem_kpm=None):
        x = self.embed(tgt)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_kpm, mem_kpm)
        return self.norm(x)


class Translator(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, num_heads, d_ff, num_layers, dropout)
        self.output_proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool), diagonal=1).to(src.device)
        memory = self.encoder(src)
        dec_out = self.decoder(tgt, memory, tgt_mask)
        return self.output_proj(dec_out)


def make_batch(batch_size, src_len, tgt_len, src_vocab, tgt_vocab, device):
    src = torch.randint(1, src_vocab, (batch_size, src_len), device=device)
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len), device=device)
    return src, tgt


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = Translator(
        src_vocab=args.src_vocab, tgt_vocab=args.tgt_vocab,
        d_model=args.d_model, num_heads=args.num_heads,
        d_ff=args.d_model * 4, num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for step in range(50):
            src, tgt_full = make_batch(args.batch_size, 20, 21, args.src_vocab, args.tgt_vocab, device)
            tgt_in = tgt_full[:, :-1]
            tgt_out = tgt_full[:, 1:]

            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, args.tgt_vocab), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} Step {step}/50 Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} average loss: {epoch_loss / 50:.4f}")

    model_path = os.path.join(os.environ.get("SM_MODEL_DIR", "."), "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model",    type=int,   default=128)
    parser.add_argument("--num-heads",  type=int,   default=4)
    parser.add_argument("--num-layers", type=int,   default=2)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--src-vocab",  type=int,   default=5000)
    parser.add_argument("--tgt-vocab",  type=int,   default=6000)
    args = parser.parse_args()
    train(args)
