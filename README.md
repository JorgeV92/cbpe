# cbpe 

A tiny, byte-level BPE tokenizer written in C with Python bindings.

It is inspired by the style of small, readable Karpathy projects such as `minbpe`, `nanoGPT`, and `rustbpe`, but implemented here with a pure C core and a Python wrapper over a compiled shared library.

## Features

- Byte-level BPE training from scratch
- Native C implementation
- Python bindings over a compiled shared library
- `Tokenizer.train_from_iterator(...)`
- `Tokenizer.encode(...)`
- `Tokenizer.decode(...)`
- `Tokenizer.batch_encode(...)`
- Vocabulary inspection with `get_vocab()` and `get_mergeable_ranks()`
- `pytest` test suite
- Small C demo for using the library directly

## How BPE works

Byte Pair Encoding builds subword tokens by repeatedly merging the most common adjacent token pair.

At a high level:
1. Start with a vocabulary of all 256 byte values.
2. Count adjacent token pairs across the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until the target vocabulary size is reached or no useful merges remain.