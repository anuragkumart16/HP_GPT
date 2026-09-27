
import json
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 4139
BLOCK_SIZE = 256
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4

CHECKPOINT = "checkpoints/step_4750.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# TOKENIZER
# ============================================================

with open("tokenizer/vocab.json", "r") as f:
    tokenizer_data = json.load(f)

token_to_id = tokenizer_data["token_to_id"]

id_to_token = {
    int(k): v
    for k, v in tokenizer_data["id_to_token"].items()
}

merges = [
    tuple(pair)
    for pair in tokenizer_data["merges"]
]


def encode_word(word):
    tokens = list(word)

    for pair in merges:

        new_tokens = []
        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(
                    tokens[i] + tokens[i + 1]
                )
                i += 2

            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return tokens


def encode(text):

    # Same whitespace handling used during tokenizer training
    text = text.replace(" ", " ▁")

    pieces = re.findall(
        r"\w+|[^\w\s]|▁",
        text
    )

    token_ids = []

    for piece in pieces:

        sub_tokens = encode_word(piece)

        for token in sub_tokens:

            if token not in token_to_id:
                raise ValueError(
                    f"Token '{token}' not found in vocabulary."
                )

            token_ids.append(
                token_to_id[token]
            )

    return token_ids


def decode(token_ids):

    tokens = [
        id_to_token[i]
        for i in token_ids
    ]

    text = ""

    for token in tokens:

        if token == "▁":
            text += " "
        else:
            text += token

    return text


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
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(
            d_model,
            3 * d_model
        )

        self.output = nn.Linear(
            d_model,
            d_model
        )

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

        qkv = self.qkv(x)

        Q, K, V = qkv.chunk(
            3,
            dim=-1
        )

        Q = Q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (
            self.head_dim ** 0.5
        )

        mask = self.mask[:T, :T]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        context = attention @ V

        context = (
            context
            .transpose(1, 2)
            .contiguous()
        )

        context = context.view(
            B,
            T,
            C
        )

        return self.output(context)


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        # IMPORTANT:
        # The checkpoint expects the module to be called "network"
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

        # Attention + residual connection
        x = x + self.attention(
            self.ln1(x)
        )

        # Feed-forward + residual connection
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

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )

        self.blocks = nn.ModuleList([

            TransformerBlock(
                d_model,
                num_heads,
                block_size
            )

            for _ in range(num_layers)
        ])

        # IMPORTANT:
        # Checkpoint calls this "ln_final"
        self.ln_final = nn.LayerNorm(
            d_model
        )

        # IMPORTANT:
        # Checkpoint has no lm_head.bias
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

    def forward(self, x):

        B, T = x.shape

        positions = torch.arange(
            T,
            device=x.device
        )

        x = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        for block in self.blocks:

            x = block(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = GPT(
    VOCAB_SIZE,
    BLOCK_SIZE,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(device)
model.eval()

print()
print("=" * 60)
print("MODEL LOADED")
print("=" * 60)
print(f"Checkpoint : {CHECKPOINT}")
print(f"Device     : {device}")
print("=" * 60)


# ============================================================
# TEXT GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8
):

    token_ids = encode(prompt)

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):

        # GPT can only see the last 256 tokens
        x_cond = x[:, -BLOCK_SIZE:]

        logits = model(x_cond)

        # We only need predictions for
        # the final token position
        logits = logits[:, -1, :]

        # Temperature controls randomness
        logits = logits / temperature

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        # Sample the next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Append it to the sequence
        x = torch.cat(
            [x, next_token],
            dim=1
        )

    return decode(
        x[0].tolist()
    )


# ============================================================
# INTERACTIVE PROMPT
# ============================================================

while True:

    prompt = input(
        "\nEnter prompt (or 'quit'): "
    )

    if prompt.lower() == "quit":
        break

    try:

        output = generate(
            prompt,
            max_new_tokens=100,
            temperature=0.8
        )

        print("\nGenerated:")
        print("-" * 60)
        print(output)
        print("-" * 60)

    except Exception as e:

        print("\nError:")
        print(e)

