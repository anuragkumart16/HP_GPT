from collections import Counter
import re
import json
import numpy as np


# ============================================
# 1. Split text into words and punctuation
# ============================================

def split_text(text):
    return re.findall(r"\w+|[^\w\s]", text)


# ============================================
# 2. Count adjacent pairs
# ============================================

def get_pair_counts(words):

    pair_counts = Counter()

    for word in words:

        for i in range(len(word) - 1):

            pair = (word[i], word[i + 1])

            pair_counts[pair] += 1

    return pair_counts


# ============================================
# 3. Merge one pair
# ============================================

def merge_pair(words, pair):

    new_words = []

    for word in words:

        new_word = []

        i = 0

        while i < len(word):

            if (
                i < len(word) - 1
                and (word[i], word[i + 1]) == pair
            ):

                merged_token = (
                    word[i] + word[i + 1]
                )

                new_word.append(merged_token)

                i += 2

            else:

                new_word.append(word[i])

                i += 1

        new_words.append(new_word)

    return new_words


# ============================================
# 4. Train BPE
# ============================================

def train_bpe(text, num_merges):

    # ----------------------------------------
    # Step 1: Split text
    # ----------------------------------------

    tokens = split_text(text)

    print("Initial tokens:", len(tokens))

    # ----------------------------------------
    # Step 2: Convert every token into chars
    # ----------------------------------------

    words = [
        list(token)
        for token in tokens
    ]

    print("Character representation created.")

    # ----------------------------------------
    # Step 3: Initial pair counts
    # ----------------------------------------

    pair_counts = get_pair_counts(words)

    # ----------------------------------------
    # Step 4: BPE training
    # ----------------------------------------

    merges = []

    for step in range(num_merges):

        if not pair_counts:
            break

        # Find most frequent pair

        best_pair = max(
            pair_counts,
            key=pair_counts.get
        )

        best_count = pair_counts[best_pair]

        # Store merge rule

        merges.append(best_pair)

        print(
            f"Merge {step + 1}/{num_merges}:",
            best_pair,
            "count =",
            best_count
        )

        # Merge pair

        words = merge_pair(
            words,
            best_pair
        )

        # Recalculate pair counts
        #
        # This is still simple/educational.
        # Later we can optimize this further.

        pair_counts = get_pair_counts(words)

    # ----------------------------------------
    # Step 5: Build vocabulary
    # ----------------------------------------

    vocabulary = set()

    for word in words:

        for token in word:

            vocabulary.add(token)

    vocabulary = sorted(vocabulary)

    return merges, vocabulary


# ============================================
# 5. Encode a word
# ============================================

def encode_word(word, merges):

    tokens = list(word)

    for pair in merges:

        new_tokens = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):

                merged_token = (
                    tokens[i] + tokens[i + 1]
                )

                new_tokens.append(
                    merged_token
                )

                i += 2

            else:

                new_tokens.append(
                    tokens[i]
                )

                i += 1

        tokens = new_tokens

    return tokens


# ============================================
# 6. Load Harry Potter corpus
# ============================================

with open(
    "data/raw/corpus.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print("\n================================")
print("CORPUS INFORMATION")
print("================================")

print(
    "Characters:",
    len(text)
)

print(
    "Words:",
    len(text.split())
)

print(
    "First 200 characters:"
)

print(text[:200])


# ============================================
# 7. Train BPE
# ============================================

print("\n================================")
print("TRAINING BPE")
print("================================")

merges, vocabulary = train_bpe(
    text,
    num_merges=50
)


# ============================================
# 8. Create Token -> ID
# ============================================

token_to_id = {}

for i, token in enumerate(vocabulary):

    token_to_id[token] = i


# ============================================
# 9. Create ID -> Token
# ============================================

id_to_token = {}

for token, token_id in token_to_id.items():

    id_to_token[token_id] = token


# ============================================
# 10. Print results
# ============================================

print("\n================================")
print("BPE RESULTS")
print("================================")

print(
    "Number of merges:",
    len(merges)
)

print(
    "Vocabulary size:",
    len(vocabulary)
)


print("\nFirst 50 vocabulary tokens:")

print(
    vocabulary[:50]
)


print("\nFirst 20 merge rules:")

for merge in merges[:20]:

    print(merge)


# ============================================
# 11. Test encoding
# ============================================

print("\n================================")
print("ENCODING TEST")
print("================================")


test_words = [
    "Harry",
    "Potter",
    "Hogwarts",
    "Dumbledore",
    "Hermione",
    "magic"
]


for word in test_words:

    tokens = encode_word(
        word,
        merges
    )

    print(
        word,
        "->",
        tokens
    )


# ============================================
# 12. Convert tokens to IDs
# ============================================

print("\n================================")
print("TOKEN IDS")
print("================================")


word = "Harry"

tokens = encode_word(
    word,
    merges
)

token_ids = []

for token in tokens:

    if token in token_to_id:

        token_ids.append(
            token_to_id[token]
        )

    else:

        print(
            "Unknown token:",
            token
        )


print(
    word,
    "->",
    tokens
)

print(
    "Token IDs:",
    token_ids
)


# ============================================
# 13. Decode token IDs back to text
# ============================================

def decode(token_ids, id_to_token):

    tokens = [
        id_to_token[token_id]
        for token_id in token_ids
    ]

    return "".join(tokens)


print("\n================================")
print("DECODING TEST")
print("================================")

decoded_word = decode(
    token_ids,
    id_to_token
)

print(
    "Token IDs:",
    token_ids,
    "-> Decoded:",
    decoded_word
)


# ============================================
# 14. Encode the full corpus
# ============================================

print("\n================================")
print("ENCODING FULL CORPUS")
print("================================")

corpus_words = split_text(text)

corpus_tokens = []

for corpus_word in corpus_words:

    corpus_tokens.extend(
        encode_word(corpus_word, merges)
    )

corpus_token_ids = [
    token_to_id[token]
    for token in corpus_tokens
]

print(
    "Words before BPE:",
    len(corpus_words)
)

print(
    "Tokens after BPE:",
    len(corpus_tokens)
)

print(
    "First 50 corpus tokens:",
    corpus_tokens[:50]
)

print(
    "First 50 corpus token IDs:",
    corpus_token_ids[:50]
)


# ============================================
# 15. Save tokens and vocabulary to disk
# ============================================

print("\n================================")
print("SAVING TO DISK")
print("================================")

np.save(
    "tokenizer/tokens.npy",
    np.array(corpus_token_ids, dtype=np.int32)
)

with open(
    "tokenizer/vocab.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "token_to_id": token_to_id,
            "merges": merges
        },
        file
    )

print("Saved tokenizer/tokens.npy")
print("Saved tokenizer/vocab.json")


# ============================================
# 16. Load a saved tokenizer from disk
# ============================================

def load_tokenizer(vocab_path):

    with open(
        vocab_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    loaded_token_to_id = data["token_to_id"]

    # JSON has no tuple type, so merges come back as
    # lists. encode_word() compares against tuples,
    # so they must be converted back.

    loaded_merges = [
        tuple(pair)
        for pair in data["merges"]
    ]

    return loaded_token_to_id, loaded_merges


print("\n================================")
print("LOAD TOKENIZER TEST")
print("================================")

loaded_token_to_id, loaded_merges = load_tokenizer(
    "tokenizer/vocab.json"
)

loaded_tokens = encode_word(
    "Harry",
    loaded_merges
)

print(
    "Harry ->",
    loaded_tokens,
    "(loaded from disk)"
)