import math
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

VOCAB_SIZE = 4139
BLOCK_SIZE = 256
D_MODEL = 256

NUM_HEADS = 4
NUM_LAYERS = 4

BATCH_SIZE = 16

LEARNING_RATE = 3e-4
MAX_ITERS = 5000

EVAL_INTERVAL = 250

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class GPTDataset(Dataset):

    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):

        x = self.tokens[
            idx : idx + self.block_size
        ]

        y = self.tokens[
            idx + 1 : idx + self.block_size + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


# ============================================================
# LOAD TOKENS
# ============================================================

tokens = torch.from_numpy(
    __import__("numpy").load(
        "tokenizer/tokens.npy"
    )
)

split = int(len(tokens) * 0.9)

train_tokens = tokens[:split]
val_tokens = tokens[split:]

train_dataset = GPTDataset(
    train_tokens,
    BLOCK_SIZE
)

val_dataset = GPTDataset(
    val_tokens,
    BLOCK_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)


# ============================================================
# MULTI-HEAD SELF ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        block_size
    ):

        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads

        self.head_dim = (
            d_model // num_heads
        )

        # Q, K, V
        self.qkv = nn.Linear(
            d_model,
            3 * d_model
        )

        # Final projection
        self.output = nn.Linear(
            d_model,
            d_model
        )

        # Causal mask
        mask = torch.tril(
            torch.ones(
                block_size,
                block_size
            )
        )

        self.register_buffer(
            "mask",
            mask
        )

    def forward(self, x):

        B, T, C = x.shape

        # ------------------------------------------------
        # Create Q, K, V
        # ------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

        # ------------------------------------------------
        # Split into heads
        # ------------------------------------------------

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # ------------------------------------------------
        # Attention
        # ------------------------------------------------

        scores = (
            q @ k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        # Prevent looking into the future
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        out = weights @ v

        # ------------------------------------------------
        # Put heads back together
        # ------------------------------------------------

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            T,
            C
        )

        return self.output(out)


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                d_model,
                4 * d_model
            ),

            nn.GELU(),

            nn.Linear(
                4 * d_model,
                d_model
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        block_size
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            d_model
        )

        self.attention = MultiHeadAttention(
            d_model,
            num_heads,
            block_size
        )

        self.ln2 = nn.LayerNorm(
            d_model
        )

        self.feed_forward = FeedForward(
            d_model
        )

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(
            self.ln1(x)
        )

        # Feed-forward + residual
        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ============================================================
# GPT MODEL
# ============================================================

class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        d_model,
        num_heads,
        num_layers
    ):

        super().__init__()

        # ------------------------------------------------
        # Embeddings
        # ------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )

        # ------------------------------------------------
        # Transformer blocks
        # ------------------------------------------------

        self.blocks = nn.ModuleList([

            TransformerBlock(
                d_model,
                num_heads,
                block_size
            )

            for _ in range(num_layers)

        ])

        # ------------------------------------------------
        # Final normalization
        # ------------------------------------------------

        self.ln_final = nn.LayerNorm(
            d_model
        )

        # ------------------------------------------------
        # Convert representation → vocabulary logits
        # ------------------------------------------------

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

    def forward(self, x, targets=None):

        B, T = x.shape

        # ------------------------------------------------
        # Token + position embeddings
        # ------------------------------------------------

        token_emb = self.token_embedding(x)

        positions = torch.arange(
            T,
            device=x.device
        )

        position_emb = self.position_embedding(
            positions
        )

        x = token_emb + position_emb

        # ------------------------------------------------
        # Transformer
        # ------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ------------------------------------------------
        # Final representation
        # ------------------------------------------------

        x = self.ln_final(x)

        # ------------------------------------------------
        # Vocabulary prediction
        # ------------------------------------------------

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = nn.functional.cross_entropy(
                logits,
                targets
            )

        return logits, loss


# ============================================================
# CREATE MODEL
# ============================================================

model = GPT(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

model = model.to(DEVICE)


print(
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

model.train()

train_iterator = iter(train_loader)

for step in range(MAX_ITERS):

    try:

        x, y = next(train_iterator)

    except StopIteration:

        train_iterator = iter(train_loader)

        x, y = next(train_iterator)

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------

    logits, loss = model(
        x,
        y
    )

    # ------------------------------------------------
    # Backpropagation
    # ------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # ------------------------------------------------
    # Logging
    # ------------------------------------------------

    if step % 50 == 0:

        print(
            f"Step {step:5d} | "
            f"Loss {loss.item():.4f}"
        )

    # ------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------

    if step > 0 and step % EVAL_INTERVAL == 0:

        os.makedirs(
            "checkpoints",
            exist_ok=True
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            },
            f"checkpoints/step_{step}.pt"
        )

        print(
            f"Checkpoint saved at step {step}"
        )