import json
from pathlib import Path

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})

def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)})

md('''# Train a ~100M parameter GPT model from scratch on Google Colab

This notebook contains the full PyTorch model code and training loop.

It trains a decoder-only GPT-style language model on the TinyStories dataset.
The default model is about **95M parameters**, close to a 100M GPT.

> Important: this is an educational training notebook. A real high-quality 100M model needs many more tokens and training time than a free Colab session can provide. This notebook is designed so you can see the full code and run a small training job on a Colab GPU.

Recommended Colab setting:

`Runtime -> Change runtime type -> GPU`  
Use T4 or better.
''')

md('''## 1. Check the GPU''')
code('''!nvidia-smi
''')

md('''## 2. Install dependencies

We only use a few external packages:

- `torch` for the model and training
- `datasets` to download TinyStories
- `transformers` only for the GPT-2 tokenizer
- `tqdm` for progress bars
''')
code('''!pip install -q datasets transformers tqdm
''')

md('''## 3. Imports and global configuration''')
code('''import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm.auto import tqdm

SEED = 1337
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
''')

md('''## 4. Load tokenizer

We use the GPT-2 tokenizer. The model itself is implemented below from scratch.
''')
code('''tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

VOCAB_SIZE = len(tokenizer)
EOS_TOKEN_ID = tokenizer.eos_token_id

print("Vocab size:", VOCAB_SIZE)
print("EOS token id:", EOS_TOKEN_ID)
''')

md('''## 5. Training configuration

The default model is approximately 95M parameters:

- 8 transformer blocks
- 768 embedding dimension
- 12 attention heads
- 256-token context length

If you run out of memory, reduce `batch_size`, `block_size`, or `n_layer`.
''')
code('''@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

@dataclass
class TrainConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    eval_interval: int = 100
    eval_steps: int = 25
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 2
    checkpoint_dir: str = "checkpoints_gpt_100m"

model_cfg = GPTConfig()
train_cfg = TrainConfig()

print(model_cfg)
print(train_cfg)
''')

md('''## 6. Dataset code

This uses the TinyStories dataset in streaming mode for training.

The dataset class below:

1. reads stories one by one,
2. tokenizes them,
3. appends EOS tokens,
4. creates fixed-length GPT training blocks,
5. returns `(x, y)` where `y` is `x` shifted by one token.
''')
code('''class TinyStoriesTokenDataset(IterableDataset):
    def __init__(self, split: str, tokenizer, block_size: int, shuffle_buffer_size: int = 10_000):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        dataset = load_dataset("roneneldan/TinyStories", split=self.split, streaming=True)

        if self.split == "train":
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=SEED)

        token_buffer = []

        for row in dataset:
            text = row["text"]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(EOS_TOKEN_ID)
            token_buffer.extend(ids)

            while len(token_buffer) >= self.block_size + 1:
                chunk = token_buffer[: self.block_size + 1]
                token_buffer = token_buffer[self.block_size + 1 :]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

train_dataset = TinyStoriesTokenDataset("train", tokenizer, model_cfg.block_size)
val_dataset = TinyStoriesTokenDataset("validation", tokenizer, model_cfg.block_size)

train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers, pin_memory=True)

print("DataLoaders created.")
''')

md('''## 7. GPT model code

This is a complete GPT implementation:

- token embeddings
- learned positional embeddings
- causal self-attention
- MLP/feed-forward blocks
- residual connections
- layer normalization
- tied output head
''')
code('''class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
''')

md('''## 8. Create the model and count parameters''')
code('''model = GPT(model_cfg).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:     {num_params / 1e6:.2f}M")
print(f"Trainable parameters: {num_trainable / 1e6:.2f}M")
''')

md('''## 9. Optimizer and learning-rate schedule''')
code('''def configure_optimizer(model, weight_decay, learning_rate):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )
    return optimizer


def get_lr(step):
    if step < train_cfg.warmup_steps:
        return train_cfg.learning_rate * (step + 1) / train_cfg.warmup_steps

    progress = (step - train_cfg.warmup_steps) / max(1, train_cfg.max_steps - train_cfg.warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_cfg.min_lr + cosine * (train_cfg.learning_rate - train_cfg.min_lr)

optimizer = configure_optimizer(model, train_cfg.weight_decay, train_cfg.learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
''')

md('''## 10. Evaluation and sampling helpers''')
code('''@torch.no_grad()
def estimate_loss(model, val_loader, eval_steps):
    model.eval()
    losses = []
    val_iter = iter(val_loader)

    for _ in range(eval_steps):
        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, y = next(val_iter)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y)

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(model, prompt="Once upon a time", max_new_tokens=120):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    text = tokenizer.decode(out[0].tolist())
    model.train()
    return text
''')

md('''## 11. Train

This loop trains for `train_cfg.max_steps` optimizer steps.

On a T4, the default settings are intentionally modest. Increase `max_steps` for better results.
''')
code('''model.train()
train_iter = iter(train_loader)

running_loss = 0.0
start_time = time.time()

pbar = tqdm(range(train_cfg.max_steps), desc="training")

for step in pbar:
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    micro_losses = []

    for micro_step in range(train_cfg.gradient_accumulation_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y)
            loss = loss / train_cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()
        micro_losses.append(loss.item() * train_cfg.gradient_accumulation_steps)

    if train_cfg.grad_clip is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    train_loss = sum(micro_losses) / len(micro_losses)
    running_loss = 0.95 * running_loss + 0.05 * train_loss if step > 0 else train_loss

    pbar.set_postfix({"loss": f"{running_loss:.4f}", "lr": f"{lr:.2e}"})

    if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
        val_loss = estimate_loss(model, val_loader, train_cfg.eval_steps)
        elapsed = time.time() - start_time
        print(f"\nstep {step}: train_loss={running_loss:.4f}, val_loss={val_loss:.4f}, elapsed={elapsed/60:.1f} min")
        print(sample_text(model, prompt="Once upon a time", max_new_tokens=80))
        print("-" * 100)

        ckpt_path = Path(train_cfg.checkpoint_dir) / f"gpt_100m_step_{step}.pt"
        torch.save({
            "step": step,
            "model_config": model_cfg.__dict__,
            "train_config": train_cfg.__dict__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_name": "gpt2",
        }, ckpt_path)
        print("saved", ckpt_path)
''')

md('''## 12. Generate text after training''')
code('''prompts = [
    "Once upon a time",
    "A little robot wanted to learn",
    "The dragon was afraid because",
]

for prompt in prompts:
    print("=" * 100)
    print("PROMPT:", prompt)
    print(sample_text(model, prompt=prompt, max_new_tokens=150))
''')

md('''## 13. Save final model''')
code('''final_path = Path(train_cfg.checkpoint_dir) / "gpt_100m_final.pt"

torch.save({
    "model_config": model_cfg.__dict__,
    "train_config": train_cfg.__dict__,
    "model_state_dict": model.state_dict(),
    "tokenizer_name": "gpt2",
}, final_path)

print("Saved final checkpoint to", final_path)
''')

md('''## 14. Load a checkpoint and generate

Use this if you come back later and want to load the model from a `.pt` checkpoint.
''')
code('''ckpt_path = Path(train_cfg.checkpoint_dir) / "gpt_100m_final.pt"
checkpoint = torch.load(ckpt_path, map_location=DEVICE)

loaded_cfg = GPTConfig(**checkpoint["model_config"])
loaded_model = GPT(loaded_cfg).to(DEVICE)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()

print("Loaded", ckpt_path)

prompt = "Once upon a time"
ids = tokenizer.encode(prompt, add_special_tokens=False)
idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
out = loaded_model.generate(idx, max_new_tokens=120, temperature=0.8, top_k=50)
print(tokenizer.decode(out[0].tolist()))
''')

md('''## 15. Optional: save to Google Drive

Colab runtimes are temporary. Save your checkpoint to Drive if you want to keep it.
''')
code('''# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p /content/drive/MyDrive/gpt_100m_checkpoints
# !cp -r checkpoints_gpt_100m /content/drive/MyDrive/gpt_100m_checkpoints/
''')

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path('notebooks').mkdir(exist_ok=True)
out = Path('notebooks/train_100m_gpt_from_scratch_colab.ipynb')
out.write_text(json.dumps(nb, indent=2), encoding='utf-8')
print(out)
