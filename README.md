# HarryGPT

> Building a small GPT-style language model **from scratch, in PyTorch** — to actually understand how LLMs work, not just use them.

**Status: 🚧 early development.** Nothing is trained yet. This README is the project's roadmap, learning plan, and setup guide — it doubles as a to-do list.

## Why this project exists

Most "build an LLM" tutorials import a Transformer class from a library and call it a day. HarryGPT does the opposite: every core piece — tokenizer, attention, training loop — is implemented by hand first, then compared against the standard implementation. The long-term goal (once the fundamentals are solid) is a domain-specific chatbot that can talk about the Harry Potter universe, trained only on legally usable/licensed text.

If you're reading this to learn alongside the project: start with the [Learning Resources](#learning-resources) section, then follow the [Roadmap](#roadmap) in order. Each phase builds directly on the last — don't skip to Transformers before self-attention makes sense on paper.

## Roadmap

Rough order of implementation. Check items off as you go.

### Phase 0 — Setup
- [ ] Python + PyTorch environment working (see [Getting Started](#getting-started))
- [ ] Pick a throwaway prototyping corpus (see [Data & Licensing](#data--licensing)) — don't start with Harry Potter text
- [ ] Repo structure in place (see [Project Structure](#project-structure))

### Phase 1 — Tokenizer
- [ ] Character-level tokenizer (simplest possible baseline)
- [ ] Byte-Pair Encoding (BPE) tokenizer implemented from scratch
- [ ] Encode/decode round-trip tests
- [ ] Compare vocab size vs. sequence length trade-offs

### Phase 2 — Data pipeline
- [ ] Text cleaning / normalization
- [ ] Train/val split
- [ ] Batching strategy (fixed context length, random sampling)
- [ ] `Dataset` / `DataLoader` (or manual batching, to understand what those abstract away)

### Phase 3 — Transformer internals (the core learning phase)
- [ ] Token embeddings
- [ ] Positional embeddings (learned, then read about RoPE/sinusoidal)
- [ ] Scaled dot-product self-attention (single head, from raw matmuls)
- [ ] Multi-head attention
- [ ] Feed-forward network (MLP block)
- [ ] Layer normalization
- [ ] Residual connections
- [ ] Full Transformer (decoder-only) block
- [ ] Stack blocks into a GPT-style model

### Phase 4 — Training
- [ ] Loss function (cross-entropy over next-token prediction)
- [ ] Training loop with checkpointing
- [ ] Learning rate schedule + gradient clipping
- [ ] Overfit on a tiny batch first (sanity check before scaling up)
- [ ] Train on the full prototyping corpus, track loss curves

### Phase 5 — Text generation
- [ ] Greedy decoding
- [ ] Temperature sampling
- [ ] Top-k / top-p (nucleus) sampling
- [ ] KV-caching for faster inference

### Phase 6 — Instruction tuning
- [ ] Build/collect a small instruction-style dataset
- [ ] Fine-tune the base model on instructions
- [ ] Basic evaluation of response quality

### Phase 7 — HarryGPT chatbot
- [ ] Swap in legally obtained/licensed Harry Potter–related text
- [ ] Retrain/fine-tune on domain data
- [ ] FastAPI backend to serve the model
- [ ] Simple chat UI (Next.js/React)

## Learning Resources

The single best resource for this project is Andrej Karpathy's from-scratch series — it's the direct inspiration for HarryGPT's structure:

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the core lecture this project follows for Phase 3–5.
- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — step-by-step commit history reproducing GPT-2.
- [karpathy/minbpe](https://github.com/karpathy/minbpe) + [`lecture.md`](https://github.com/karpathy/minbpe/blob/master/lecture.md) — build a real BPE tokenizer from scratch (Phase 1).
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — reference for what a full pretrain → SFT → chat pipeline looks like end-to-end (Phase 6–7).
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — if attention/backprop fundamentals are shaky, start here before Phase 3.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original Transformer paper.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — line-by-line paper-to-code walkthrough, good companion once the from-scratch version works.

Read/watch a resource, then implement it without looking — that's the point of this project.

## Project Structure

```
HP_GPT/
├── data/                 # raw + processed text (gitignored — see Data & Licensing)
├── tokenizer/            # from-scratch tokenizer implementation
├── model/                # embeddings, attention, transformer blocks, GPT model
├── training/             # training loop, config, checkpoints
├── generation/           # sampling / decoding strategies
├── api/                  # FastAPI backend (later, Phase 7)
├── web/                  # Next.js/React chat UI (later, Phase 7)
├── notebooks/            # exploration / debugging scratch work
└── README.md
```

This structure will grow as phases are implemented — treat it as a starting point, not a spec.

## Getting Started

```bash
git clone <this-repo-url>
cd HP_GPT

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install torch numpy
```

FastAPI and the Next.js frontend aren't needed until Phase 7 — no point installing them yet.

## Data & Licensing

- **Copyrighted Harry Potter text will never be committed to this repository.** `data/` is gitignored.
- For prototyping the tokenizer, model, and training loop (Phases 0–5), use a small public-domain or freely licensed corpus — e.g. [tiny-shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) — so architecture bugs are caught before any real content is involved.
- Harry Potter–specific data (Phase 6–7) will only be used if it is legally obtained or appropriately licensed. No pirated or scraped copyrighted text.
- This project is for personal, educational use — not a commercial product.

## Tech Stack

- **Python** — core language
- **PyTorch** — model implementation and training
- **NumPy** — numerical utilities
- **FastAPI** *(later)* — serving the trained model
- **Next.js / React** *(later)* — chat interface

## Notes

- No benchmarks, results, or trained checkpoints exist yet — this README will be updated as each phase actually ships, not before.
- Contributions/suggestions welcome once there's real code to react to; for now this is a solo learning project.
