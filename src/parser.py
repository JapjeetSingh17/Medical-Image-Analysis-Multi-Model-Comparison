"""Utilities for extracting reasoning traces and multiple-choice answers from VLM outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ParsedPrediction:
    """Structured prediction parsed from raw model generation."""

    predicted_letter: str | None
    reasoning: str
    raw_answer_span: str
    format_compliant: bool


_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
_GEMMA_THINK_RE = re.compile(
    r"<unused94>thought\s*(.*?)<unused95>", flags=re.DOTALL | re.IGNORECASE
)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def extract_reasoning(raw_text: str) -> tuple[str, str]:
    """Split optional <think> or Gemma thought tokens from the remaining answer text."""
    text = raw_text or ""
    for pattern in (_THINK_RE, _GEMMA_THINK_RE):
        match = pattern.search(text)
        if match:
            reasoning = match.group(1).strip()
            remainder = pattern.sub("", text).strip()
            return reasoning, remainder
    return "", text.strip()


def _match_letter(candidate: str, valid_letters: set[str]) -> str | None:
    """Extract a single valid option letter from a short candidate span."""
    if not candidate:
        return None
    letters_pat = "".join(sorted(valid_letters))
    # Priority 1: Explicit start-of-string or explicit prefix (case-insensitive)
    explicit_patterns = [
        rf"^\s*[\(\[]?([{letters_pat}])[\)\]\.\:]?(?:\s|$)",
        rf"(?:answer|choice|option|diagnosis)\s*(?:is|:|=)?\s*[\(\[]?([{letters_pat}])[\)\]\.\:]?(?:\b|$)",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, candidate, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter

    # Priority 2: Standalone uppercase option letter only (avoids matching English word 'a')
    m_upper = re.search(rf"(?<![A-Za-z])([{letters_pat}])(?=[\)\]\.\:,;\s]|$)", candidate)
    if m_upper:
        letter = m_upper.group(1)
        if letter in valid_letters:
            return letter
    return None


def parse_model_response(
    raw_text: str,
    options: Mapping[str, str],
) -> ParsedPrediction:
    """Parse a model's free-form or tagged response into an option letter (e.g., 'A'-'E').

    Supports:
    - <think>...</think> + <answer>B</answer> (MedVLM-R1 style)
    - Explicit answer prefixes ("Answer: C", "Option B", "(D)")
    - Option-text verbatim match before standalone uppercase letter fallback
    """
    valid_letters = {k.upper() for k in options.keys()}
    reasoning, remainder = extract_reasoning(raw_text)

    # 1. Explicit <answer>...</answer> tag
    ans_match = _ANSWER_TAG_RE.search(remainder) or _ANSWER_TAG_RE.search(raw_text or "")
    if ans_match:
        span = ans_match.group(1).strip()
        letter = _match_letter(span, valid_letters)
        if letter:
            return ParsedPrediction(
                predicted_letter=letter,
                reasoning=reasoning,
                raw_answer_span=span,
                format_compliant=True,
            )

    # 2. Explicit answer prefixes ("Answer: C", leading "B.", etc.)
    letters_pat = "".join(sorted(valid_letters))
    for pat in (
        rf"^\s*[\(\[]?([{letters_pat}])[\)\]\.\:]?(?:\s|$)",
        rf"(?:answer|choice|option|diagnosis)\s*(?:is|:|=)?\s*[\(\[]?([{letters_pat}])[\)\]\.\:]?(?:\b|$)",
        rf"labeled\s+(?:as\s+)?[\"'\(\[]?([{letters_pat}])[\"'\)\]]?",
    ):
        m = re.search(pat, remainder, flags=re.IGNORECASE)
        if m and m.group(1).upper() in valid_letters:
            return ParsedPrediction(
                predicted_letter=m.group(1).upper(),
                reasoning=reasoning,
                raw_answer_span=remainder[:160],
                format_compliant=True,
            )

    # 3. Option-text verbatim match (checked before loose uppercase letter search)
    lowered = remainder.lower()
    matched_letters: list[tuple[int, str]] = []
    for key, text in options.items():
        clean_opt = (text or "").strip().lower()
        if len(clean_opt) >= 4 and clean_opt in lowered:
            matched_letters.append((len(clean_opt), key.upper()))
    if matched_letters:
        matched_letters.sort(reverse=True)
        return ParsedPrediction(
            predicted_letter=matched_letters[0][1],
            reasoning=reasoning,
            raw_answer_span=remainder[:160],
            format_compliant=False,
        )

    # 4. Standalone uppercase letter fallback (e.g., "...most likely C.")
    m_upper = re.search(rf"(?<![A-Za-z])([{letters_pat}])(?=[\)\]\.\:,;\s]|$)", remainder)
    if m_upper and m_upper.group(1) in valid_letters:
        return ParsedPrediction(
            predicted_letter=m_upper.group(1),
            reasoning=reasoning,
            raw_answer_span=remainder[:160],
            format_compliant=False,
        )

    return ParsedPrediction(
        predicted_letter=None,
        reasoning=reasoning,
        raw_answer_span=remainder[:160],
        format_compliant=False,
    )
