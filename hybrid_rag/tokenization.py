"""适用于中文 BM25 的轻量分词器，不依赖外部分词词典。"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal


# 英文、数字和错误码保持为完整 token；连续中文由 tokenize_zh 继续切分。
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*|[\u4e00-\u9fff]+")

BM25TokenizerMode = Literal["ngram", "jieba", "hybrid"]
BM25Tokenizer = Callable[[str], list[str]]


def tokenize_zh(text: str) -> list[str]:
    """返回英文/数字词，以及中文单字和双字组合。

    中文采用单字和双字组合，不需要维护全局词典，也能覆盖产品名等新词。
    如果业务已有 jieba 等分词方案，可以在创建 BM25Index 时注入替换。
    """

    tokens: list[str] = []
    for value in _TOKEN_PATTERN.findall(text.lower()):
        if _is_chinese(value):
            # 例如“频繁掉线”会生成单字，以及“频繁、繁掉、掉线”等双字词。
            tokens.extend(value)
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens


class JiebaBM25Tokenizer:
    """使用独立 Jieba 词典的 BM25 tokenizer。

    ``jieba`` 模式使用搜索引擎分词；``hybrid`` 在此基础上加入中文双字
    fallback。英文、数字、版本号和错误码在进入 Jieba 前已被保护，因此不会
    被拆成多个 token。
    """

    def __init__(
        self,
        mode: Literal["jieba", "hybrid"],
        user_dict: str | Path | None = None,
        domain_terms: Iterable[str] = (),
    ) -> None:
        if mode not in {"jieba", "hybrid"}:
            raise ValueError(f"unsupported Jieba tokenizer mode: {mode}")

        import jieba

        self.mode = mode
        # 不使用 jieba.dt 全局实例，避免一个客服业务加载的词典影响其他索引。
        self._jieba = jieba.Tokenizer()
        if user_dict is not None:
            path = Path(user_dict).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Jieba user dictionary not found: {path}")
            self._jieba.load_userdict(str(path))

        for term in _normalized_domain_terms(domain_terms):
            self._jieba.add_word(term)

    def __call__(self, text: str) -> list[str]:
        tokens: list[str] = []
        for value in _TOKEN_PATTERN.findall(text.lower()):
            if not _is_chinese(value):
                tokens.append(value)
                continue

            jieba_tokens = [
                token.strip()
                for token in self._jieba.cut_for_search(value)
                if token.strip()
            ]
            if self.mode == "hybrid":
                # 同一中文片段内去重，避免 Jieba 和 fallback 恰好生成相同双字词时
                # 人为增加词频；不同位置重复出现的词仍会保留正常的 BM25 词频。
                jieba_tokens.extend(
                    value[index : index + 2] for index in range(len(value) - 1)
                )
                jieba_tokens = list(dict.fromkeys(jieba_tokens))
            tokens.extend(jieba_tokens)
        return tokens


def create_bm25_tokenizer(
    mode: BM25TokenizerMode = "ngram",
    user_dict: str | Path | None = None,
    domain_terms: Iterable[str] = (),
) -> BM25Tokenizer:
    """根据配置创建 BM25 tokenizer；默认模式保持原有 n-gram 行为。"""

    if mode == "ngram":
        if user_dict is not None or tuple(domain_terms):
            raise ValueError("Jieba dictionaries require bm25_tokenizer='jieba' or 'hybrid'")
        return tokenize_zh
    if mode in {"jieba", "hybrid"}:
        return JiebaBM25Tokenizer(mode, user_dict, domain_terms)
    raise ValueError(f"unsupported BM25 tokenizer mode: {mode}")


def _is_chinese(value: str) -> bool:
    """判断一个 token 是否完全由常用汉字组成。"""
    return bool(value) and all("\u4e00" <= char <= "\u9fff" for char in value)


def _normalized_domain_terms(values: Iterable[str]) -> tuple[str, ...]:
    """清理并稳定去重运行时传入的领域词。"""

    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
