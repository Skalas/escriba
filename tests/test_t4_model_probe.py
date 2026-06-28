"""T4: Remote model-probe hygiene — warnings not errors, TTL cache, no key no probe."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Each test starts with a clean cache
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_models_cache():
    from escriba.summarize.llm_summary import invalidate_models_cache
    invalidate_models_cache()
    yield
    invalidate_models_cache()


# ---------------------------------------------------------------------------
# T4a: invalid/expired key → warning, not error
# ---------------------------------------------------------------------------


def test_gemini_listing_failure_emits_warning_not_error(caplog) -> None:
    """An exception during Gemini model listing logs at WARNING, not ERROR."""
    import logging
    from escriba.summarize import llm_summary

    # Patch the SDK import and provide a key so the function reaches the error path.
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
            with patch("escriba.summarize.llm_summary._call_with_timeout",
                       side_effect=Exception("403 invalid key")):
                with caplog.at_level(logging.WARNING, logger="escriba.summarize.llm_summary"):
                    result = llm_summary._list_gemini_models()

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not error_records, "listing failure must not log at ERROR"
    assert warning_records, "listing failure must log at WARNING"
    assert result == [llm_summary.DEFAULT_GEMINI_MODEL]


def test_claude_listing_failure_emits_warning_not_error(caplog) -> None:
    """An exception during Claude model listing logs at WARNING, not ERROR."""
    import logging
    from escriba.summarize import llm_summary

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}, clear=False):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            with patch("escriba.summarize.llm_summary._call_with_timeout",
                       side_effect=Exception("401 unauthorized")):
                with caplog.at_level(logging.WARNING, logger="escriba.summarize.llm_summary"):
                    result = llm_summary._list_claude_models()

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not error_records, "listing failure must not log at ERROR"
    assert warning_records, "listing failure must log at WARNING"
    assert result == [llm_summary.DEFAULT_CLAUDE_MODEL]


def test_gemini_timeout_emits_warning_not_error(caplog) -> None:
    """A TimeoutError during Gemini listing logs at WARNING."""
    import logging
    from escriba.summarize import llm_summary

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}, clear=False):
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
            with patch("escriba.summarize.llm_summary._call_with_timeout",
                       side_effect=TimeoutError("timed out")):
                with caplog.at_level(logging.WARNING, logger="escriba.summarize.llm_summary"):
                    llm_summary._list_gemini_models()

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not error_records, "timeout must not log at ERROR"


# ---------------------------------------------------------------------------
# T4b: TTL cache — repeated calls within TTL do not re-probe
# ---------------------------------------------------------------------------


def test_list_available_models_cached_within_ttl() -> None:
    """A second call within TTL returns cached value without re-probing."""
    from escriba.summarize import llm_summary

    call_count = 0

    def fake_uncached():
        nonlocal call_count
        call_count += 1
        return {"models": {}, "recommended": None, "ai_available": False, "ai_unavailable_reason": "x"}

    with patch("escriba.summarize.llm_summary._list_available_models_uncached",
               side_effect=fake_uncached):
        llm_summary.list_available_models()
        llm_summary.list_available_models()  # second call within TTL

    assert call_count == 1, "second call should use cache"


def test_list_available_models_re_probes_after_invalidation() -> None:
    """After invalidate_models_cache(), the next call re-probes."""
    from escriba.summarize import llm_summary

    call_count = 0

    def fake_uncached():
        nonlocal call_count
        call_count += 1
        return {"models": {}, "recommended": None, "ai_available": False, "ai_unavailable_reason": "x"}

    with patch("escriba.summarize.llm_summary._list_available_models_uncached",
               side_effect=fake_uncached):
        llm_summary.list_available_models()
        llm_summary.invalidate_models_cache()
        llm_summary.list_available_models()

    assert call_count == 2, "post-invalidation call must re-probe"


def test_list_available_models_re_probes_after_ttl(monkeypatch) -> None:
    """After TTL expires, the next call re-probes providers."""
    from escriba.summarize import llm_summary

    call_count = 0

    def fake_uncached():
        nonlocal call_count
        call_count += 1
        return {"models": {}, "recommended": None, "ai_available": False, "ai_unavailable_reason": "x"}

    # Monkey-patch TTL to a tiny value and make cache appear stale
    monkeypatch.setattr(llm_summary, "_MODELS_CACHE_TTL", -1.0)

    with patch("escriba.summarize.llm_summary._list_available_models_uncached",
               side_effect=fake_uncached):
        llm_summary.list_available_models()
        llm_summary.list_available_models()  # should re-probe since TTL < 0

    assert call_count == 2, "expired TTL must trigger re-probe"


# ---------------------------------------------------------------------------
# T4c: no probe when key is absent
# ---------------------------------------------------------------------------


def test_no_gemini_probe_when_key_absent() -> None:
    """Gemini models are not probed when GEMINI_API_KEY is missing."""
    from escriba.summarize import llm_summary

    probed = []

    def fake_list_gemini():
        probed.append("gemini")
        return ["gemini-2.5-flash"]

    env = {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        with patch("escriba.summarize.llm_summary._list_gemini_models", side_effect=fake_list_gemini):
            with patch.dict("sys.modules", {"mlx_lm": None}):
                llm_summary._list_available_models_uncached()

    assert "gemini" not in probed, "Gemini probed without a key"


def test_no_claude_probe_when_key_absent() -> None:
    """Claude models are not probed when ANTHROPIC_API_KEY is missing."""
    from escriba.summarize import llm_summary

    probed = []

    def fake_list_claude():
        probed.append("claude")
        return ["claude-sonnet-4-6-20250514"]

    env = {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        with patch("escriba.summarize.llm_summary._list_claude_models", side_effect=fake_list_claude):
            with patch.dict("sys.modules", {"mlx_lm": None}):
                llm_summary._list_available_models_uncached()

    assert "claude" not in probed, "Claude probed without a key"


def test_local_models_listed_without_network_call() -> None:
    """Local model presets appear even when no API keys are set."""
    from escriba.summarize import llm_summary

    env = {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            result = llm_summary._list_available_models_uncached()

    assert "local" in result["models"]
    assert len(result["models"]["local"]) > 0
