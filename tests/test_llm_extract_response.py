"""Tests for local LLM response extraction."""

from __future__ import annotations

from escriba.summarize.llm_summary import _extract_response


def test_extract_response_prefers_response_channel() -> None:
    text = "<|channel>thought\nhmm\n<|channel>response\nHello world"
    assert _extract_response(text) == "Hello world"


def test_extract_response_strips_thought_only_output() -> None:
    text = "<|channel>thought\nunfinished reasoning"
    assert _extract_response(text) == ""


def test_extract_response_strips_leaked_special_tokens() -> None:
    text = "plain answer<|end|>"
    assert _extract_response(text) == "plain answer"
