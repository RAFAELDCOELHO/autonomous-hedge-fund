"""FinGPT v3-inspired sentiment classifier, backed by claude-haiku-4-5.

Implements the exact FinGPT v3 prompt template
(Instruction/Input/Answer) and maps {negative, neutral, positive} to
{-1, 0, 1}. Calls run sequentially so logs are deterministic for
backtests and academic runs.

Degrades gracefully to None on: missing SDK, missing API key, network
errors, or all-ambiguous responses. The TradingAgents sentiment
analyst continues with its qualitative flow when this returns None.
"""

from __future__ import annotations

import logging
import os
import re
from statistics import mean
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 10

FINGPT_V3_PROMPT = (
    "Instruction: What is the sentiment of this news? "
    "Please choose an answer from {negative/neutral/positive}\n"
    "Input: {text}\n"
    "Answer:"
)

_LABEL_TO_SCORE = {"negative": -1, "neutral": 0, "positive": 1}
_LABEL_PATTERN = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)

_POSITIVE_THRESHOLD = 0.15
_NEGATIVE_THRESHOLD = -0.15


def _get_anthropic_client():
    """Return an Anthropic client or None if unavailable."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set; FinGPT sentiment disabled")
        return None
    try:
        import anthropic

        return anthropic.Anthropic()
    except Exception as e:
        logger.warning("anthropic SDK unavailable (%s); FinGPT sentiment disabled", e)
        return None


def _classify_single(client, text: str) -> Optional[int]:
    """Classify one text into -1/0/1. Returns None on error or ambiguity."""
    prompt = FINGPT_V3_PROMPT.replace("{text}", text)
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.content[0].text if resp.content else ""
        match = _LABEL_PATTERN.search(content)
        if not match:
            logger.warning(
                "FinGPT: unparseable response %r for text=%r",
                content, text[:60],
            )
            return None
        label = match.group(1).lower()
        return _LABEL_TO_SCORE[label]
    except Exception as e:
        logger.warning("FinGPT classification failed (%s) for text=%r", e, text[:60])
        return None


def _aggregate_label(score: float) -> str:
    if score > _POSITIVE_THRESHOLD:
        return "positive"
    if score < _NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def get_fingpt_sentiment(
    texts: list[str],
    ticker: str,
    curr_date: str,
) -> Optional[dict]:
    """Classify a batch of texts and return an aggregate sentiment record.

    Returns None if the client cannot be constructed, texts is empty, or
    every classification fails. Otherwise returns:

        {
          "ticker", "curr_date",
          "score": float in [-1, 1],
          "label": "positive" | "neutral" | "negative",
          "scores": list[int] with per-text labels in {-1, 0, 1},
          "n": number of successful classifications,
          "n_failed": number of texts that could not be classified,
          "model": model id,
        }
    """
    if not texts:
        return None

    client = _get_anthropic_client()
    if client is None:
        return None

    scores: list[int] = []
    for text in texts:
        score = _classify_single(client, text)
        if score is not None:
            scores.append(score)

    if not scores:
        logger.warning("FinGPT: all %d classifications failed for %s", len(texts), ticker)
        return None

    avg = mean(scores)
    return {
        "ticker": ticker,
        "curr_date": curr_date,
        "score": round(avg, 4),
        "label": _aggregate_label(avg),
        "scores": scores,
        "n": len(scores),
        "n_failed": len(texts) - len(scores),
        "model": _MODEL,
    }
