"""Small dependency-free tokenizer suitable for Chinese BM25 retrieval."""

from __future__ import annotations

import re


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*|[\u4e00-\u9fff]+")


def tokenize_zh(text: str) -> list[str]:
    """Return ASCII terms plus Chinese unigrams and bigrams.

    Character n-grams work for product names and error codes without requiring a
    global dictionary. A production host can inject its own tokenizer later.
    """

    tokens: list[str] = []
    for value in _TOKEN_PATTERN.findall(text.lower()):
        if _is_chinese(value):
            tokens.extend(value)
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens


def _is_chinese(value: str) -> bool:
    return bool(value) and all("\u4e00" <= char <= "\u9fff" for char in value)

