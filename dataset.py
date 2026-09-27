import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


TOKENS_PATH = "tokenizer/tokens.npy"

BLOCK_SIZE = 256
BATCH_SIZE = 16
TRAIN_SPLIT = 0.9


class GPTDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def create_datasets():
    tokens = np.load(TOKENS_PATH)

    print(f"Total tokens: {len(tokens):,}")

    split = int(len(tokens) * TRAIN_SPLIT)

    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    print(f"Training tokens:   {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")

    train_dataset = GPTDataset(train_tokens, BLOCK_SIZE)
    val_dataset = GPTDataset(val_tokens, BLOCK_SIZE)

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )

    x, y = next(iter(train_loader))

    print("\nBatch shapes:")
    print("X:", x.shape)
    print("Y:", y.shape)

    print("\nFirst sequence:")
    print("X:", x[0][:20])
    print("Y:", y[0][:20])