import re
import json
import os
import numpy as np
from collections import Counter


CORPUS_PATH = "data/raw/corpus.txt"
CHECKPOINT_PATH = "tokenizer/bpe_checkpoint.json"

MIN_FREQUENCY = 50
CHECKPOINT_EVERY = 100


# -----------------------------
# 1. Split text
# -----------------------------

def split_text(text):
    text = text.replace(" ", " ▁")
    return re.findall(r"\w+|[^\w\s]|▁", text)


# -----------------------------
# 2. Count adjacent pairs
# -----------------------------

def get_pair_counts(words):
    pair_counts = Counter()

    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += 1

    return pair_counts


# -----------------------------
# 3. Merge a pair
# -----------------------------

def merge_pair(words, pair):
    merged_words = []

    for word in words:
        new_word = []
        i = 0

        while i < len(word):
            if (
                i < len(word) - 1
                and word[i] == pair[0]
                and word[i + 1] == pair[1]
            ):
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        merged_words.append(new_word)

    return merged_words


# -----------------------------
# 4. Save checkpoint
# -----------------------------

def save_checkpoint(merges):
    os.makedirs("tokenizer", exist_ok=True)

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(
            {
                "merges": merges
            },
            f
        )

    print(f"Checkpoint saved at {len(merges)} merges")


# -----------------------------
# 5. Train BPE
# -----------------------------

def train_bpe(text):

    print("Splitting text...")

    tokens = split_text(text)

    print(f"Initial tokens: {len(tokens):,}")

    # Convert every token into characters
    words = [list(token) for token in tokens]

    merges = []

    while True:

        pair_counts = get_pair_counts(words)

        if not pair_counts:
            break

        pair, frequency = pair_counts.most_common(1)[0]

        print(
            f"Merge {len(merges) + 1}: "
            f"{pair} -> frequency {frequency:,}"
        )

        # Stop when frequency falls below threshold
        if frequency < MIN_FREQUENCY:
            print(
                f"\nStopping BPE training."
                f"\nBest pair frequency: {frequency}"
                f"\nMinimum frequency: {MIN_FREQUENCY}"
            )
            break

        # Apply merge
        words = merge_pair(words, pair)

        merges.append(pair)

        # Save every 100 merges
        if len(merges) % CHECKPOINT_EVERY == 0:
            save_checkpoint(merges)

    return merges, words


# -----------------------------
# 6. Build vocabulary
# -----------------------------

def build_vocab(words):

    vocabulary = set()

    for word in words:
        vocabulary.update(word)

    token_to_id = {
        token: i
        for i, token in enumerate(sorted(vocabulary))
    }

    id_to_token = {
        i: token
        for token, i in token_to_id.items()
    }

    return token_to_id, id_to_token


# -----------------------------
# 7. Encode corpus
# -----------------------------

def encode_word(word, merges):

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
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    return tokens


# -----------------------------
# 8. Main
# -----------------------------

if __name__ == "__main__":

    print("Loading corpus...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Corpus characters: {len(text):,}")

    merges, words = train_bpe(text)

    print(f"\nTotal merges learned: {len(merges):,}")

    # Build vocabulary
    token_to_id, id_to_token = build_vocab(words)

    print(f"Vocabulary size: {len(token_to_id):,}")

    # Encode corpus
    print("\nEncoding corpus...")

    corpus_token_ids = []

    for word in words:

        for token in word:
            corpus_token_ids.append(
                token_to_id[token]
            )

    corpus_token_ids = np.array(
        corpus_token_ids,
        dtype=np.int32
    )

    print(
        f"Final corpus tokens: "
        f"{len(corpus_token_ids):,}"
    )

    # Save token IDs
    os.makedirs("tokenizer", exist_ok=True)

    np.save(
        "tokenizer/tokens.npy",
        corpus_token_ids
    )

    # Save vocabulary
    with open(
        "tokenizer/vocab.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "token_to_id": token_to_id,
                "id_to_token": {
                    str(k): v
                    for k, v in id_to_token.items()
                },
                "merges": merges
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nTokenizer saved:")
    print("  tokenizer/tokens.npy")
    print("  tokenizer/vocab.json")