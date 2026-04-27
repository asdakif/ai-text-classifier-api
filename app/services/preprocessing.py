from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


TOKEN_PATTERN = re.compile(r"\b[\w']+\b")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(texts: Iterable[str], min_freq: int = 1) -> dict[str, int]:
    token_counts: Counter[str] = Counter()
    for text in texts:
        token_counts.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(token_counts.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    tokens = tokenize(text)
    unk_index = vocab[UNK_TOKEN]
    token_ids = [vocab.get(token, unk_index) for token in tokens[:max_length]]

    if len(token_ids) < max_length:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_length - len(token_ids)))

    return token_ids


def vectorize_text(text: str, vocab: dict[str, int], max_length: int) -> list[float]:
    token_ids = encode_text(text=text, vocab=vocab, max_length=max_length)
    vector = [0.0] * len(vocab)

    for token_id in token_ids:
        if token_id == vocab[PAD_TOKEN]:
            continue
        vector[token_id] += 1.0

    return vector
