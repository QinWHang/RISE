from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(prediction: str, answer: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class RISEQAMetrics:
    f1: float
    em: float


def score_answer(prediction: str, answer: str) -> RISEQAMetrics:
    pred = _normalize_answer(prediction)
    gold = _normalize_answer(answer)
    return RISEQAMetrics(f1=_f1(prediction, answer), em=1.0 if pred == gold else 0.0)
