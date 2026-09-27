import torch
import torch.nn as nn
import math


# -------------------------
# Configuration
# -------------------------

D_MODEL = 256
BLOCK_SIZE = 256


# -------------------------
# Causal Self-Attention
# -------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, block_size):
        super().__init__()

        self.d_model = d_model

        # Learnable projections:
        # embedding → Query
        # embedding → Key
        # embedding → Value
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Output projection
        self.output = nn.Linear(d_model, d_model)

        # Causal mask
        #
        # Token at position i can only see:
        # positions <= i
        #
        # Example:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        #
        mask = torch.tril(
            torch.ones(block_size, block_size)
        )

        self.register_buffer(
            "mask",
            mask
        )

    def forward(self, x):

        # x:
        # [batch_size, sequence_length, d_model]

        batch_size, sequence_length, _ = x.shape

        # -------------------------
        # Create Q, K, V
        # -------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # -------------------------
        # Attention scores
        # -------------------------

        scores = Q @ K.transpose(-2, -1)

        # Scale by sqrt(d_model)
        scores = scores / math.sqrt(self.d_model)

        # -------------------------
        # Causal masking
        # -------------------------

        mask = self.mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # -------------------------
        # Softmax
        # -------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -------------------------
        # Weighted values
        # -------------------------

        context = attention_weights @ V

        # -------------------------
        # Output projection
        # -------------------------

        output = self.output(context)

        return output


# -------------------------
# Test
# -------------------------

if __name__ == "__main__":

    # Fake embeddings
    x = torch.randn(
        16,
        BLOCK_SIZE,
        D_MODEL
    )

    attention = CausalSelfAttention(
        D_MODEL,
        BLOCK_SIZE
    )

    output = attention(x)

    print("Input:")
    print(x.shape)

    print("\nOutput:")
    print(output.shape)

    print("\nParameters:")
    print(
        sum(
            p.numel()
            for p in attention.parameters()
        )
    )