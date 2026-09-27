import torch
import torch.nn as nn


# -------------------------
# Model configuration
# -------------------------

VOCAB_SIZE = 4139
BLOCK_SIZE = 256
D_MODEL = 256


# -------------------------
# Embedding layer
# -------------------------

class GPTEmbedding(nn.Module):

    def __init__(self, vocab_size, block_size, d_model):
        super().__init__()

        # Converts token IDs → vectors
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Gives every position its own vector
        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )

    def forward(self, x):

        # x shape:
        # [batch_size, sequence_length]

        batch_size, sequence_length = x.shape

        # Token embeddings
        token_emb = self.token_embedding(x)

        # Positions: [0, 1, 2, ..., sequence_length-1]
        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # Position embeddings
        position_emb = self.position_embedding(positions)

        # Broadcasting adds position embedding
        # to every sequence in the batch
        embeddings = token_emb + position_emb

        return embeddings


# -------------------------
# Test
# -------------------------

if __name__ == "__main__":

    # Fake batch for testing
    x = torch.randint(
        0,
        VOCAB_SIZE,
        (16, BLOCK_SIZE)
    )

    model = GPTEmbedding(
        VOCAB_SIZE,
        BLOCK_SIZE,
        D_MODEL
    )

    output = model(x)

    print("Input shape:")
    print(x.shape)

    print("\nToken embedding table:")
    print(model.token_embedding.weight.shape)

    print("\nPosition embedding table:")
    print(model.position_embedding.weight.shape)

    print("\nOutput embedding shape:")
    print(output.shape)