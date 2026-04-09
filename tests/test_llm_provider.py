"""Tests for LLM provider dispatch, RAM recommendation, and local model cache."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestRecommendModel:
    """Test RAM-based model selection."""

    def test_16gb_gets_large_model(self):
        from escriba.summarize.llm_summary import recommend_model

        with patch("escriba.summarize.llm_summary.get_system_ram_gb", return_value=16):
            model = recommend_model()
        assert model == "mlx-community/gemma-4-26b-a4b-it-4bit"

    def test_8gb_gets_small_model(self):
        from escriba.summarize.llm_summary import recommend_model

        with patch("escriba.summarize.llm_summary.get_system_ram_gb", return_value=8):
            model = recommend_model()
        assert model == "mlx-community/gemma-4-e4b-it-4bit"

    def test_4gb_gets_none(self):
        from escriba.summarize.llm_summary import recommend_model

        with patch("escriba.summarize.llm_summary.get_system_ram_gb", return_value=4):
            model = recommend_model()
        assert model is None

    def test_boundary_8gb_exact(self):
        from escriba.summarize.llm_summary import recommend_model

        with patch("escriba.summarize.llm_summary.get_system_ram_gb", return_value=8):
            model = recommend_model()
        assert model is not None

    def test_boundary_below_8gb(self):
        from escriba.summarize.llm_summary import recommend_model

        with patch("escriba.summarize.llm_summary.get_system_ram_gb", return_value=7):
            model = recommend_model()
        assert model is None

    def test_sysctl_failure_returns_none(self):
        from escriba.summarize.llm_summary import recommend_model

        with patch("escriba.summarize.llm_summary.get_system_ram_gb", return_value=0):
            model = recommend_model()
        assert model is None


class TestResolveProviderAndModel:
    """Test provider dispatch logic."""

    def test_local_explicit(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        with patch("escriba.summarize.llm_summary.recommend_model", return_value="mlx-community/gemma-4-e4b-it-4bit"):
            provider, model_id = resolve_provider_and_model("local")
        assert provider == "local"
        assert model_id == "mlx-community/gemma-4-e4b-it-4bit"

    def test_local_with_env_override(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        with patch.dict(os.environ, {"LOCAL_LLM_MODEL": "mlx-community/custom-model"}):
            provider, model_id = resolve_provider_and_model("local")
        assert provider == "local"
        assert model_id == "mlx-community/custom-model"

    def test_local_low_ram_falls_back_to_remote(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        with patch("escriba.summarize.llm_summary.recommend_model", return_value=None), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            provider, model_id = resolve_provider_and_model("local")
        assert provider == "gemini"

    def test_auto_prefers_local(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        with patch("escriba.summarize.llm_summary.recommend_model", return_value="mlx-community/gemma-4-e4b-it-4bit"):
            provider, model_id = resolve_provider_and_model("auto")
        assert provider == "local"
        assert model_id == "mlx-community/gemma-4-e4b-it-4bit"

    def test_auto_falls_back_to_gemini(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        with patch("escriba.summarize.llm_summary.recommend_model", return_value=None), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            provider, _ = resolve_provider_and_model("auto")
        assert provider == "gemini"

    def test_auto_falls_back_to_claude(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        env = {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "test-key"}
        with patch("escriba.summarize.llm_summary.recommend_model", return_value=None), \
             patch.dict(os.environ, env, clear=False):
            provider, _ = resolve_provider_and_model("auto")
        assert provider == "claude"

    def test_auto_nothing_available(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        env = {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
        with patch("escriba.summarize.llm_summary.recommend_model", return_value=None), \
             patch.dict(os.environ, env, clear=False):
            provider, model_id = resolve_provider_and_model("auto")
        assert provider == "none"
        assert model_id is None

    def test_gemini_unchanged(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        provider, _ = resolve_provider_and_model("gemini")
        assert provider == "gemini"

    def test_claude_unchanged(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        provider, _ = resolve_provider_and_model("claude")
        assert provider == "claude"

    def test_full_hf_repo_id(self):
        from escriba.summarize.llm_summary import resolve_provider_and_model

        provider, model_id = resolve_provider_and_model("mlx-community/some-model")
        assert provider == "local"
        assert model_id == "mlx-community/some-model"


class TestLocalModelCache:
    """Test hybrid cache lifecycle."""

    def test_cache_loads_and_returns_model(self):
        from escriba.summarize.llm_summary import _LocalModelCache

        cache = _LocalModelCache(ttl=10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
            model, tokenizer = cache.get("test-model")

        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_cache_reuses_on_second_call(self):
        from escriba.summarize.llm_summary import _LocalModelCache

        cache = _LocalModelCache(ttl=10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)) as mock_load:
            cache.get("test-model")
            cache.get("test-model")

        mock_load.assert_called_once()

    def test_cache_evicts_on_model_change(self):
        from escriba.summarize.llm_summary import _LocalModelCache

        cache = _LocalModelCache(ttl=10)
        mock1 = (MagicMock(), MagicMock())
        mock2 = (MagicMock(), MagicMock())

        with patch("mlx_lm.load", side_effect=[mock1, mock2]) as mock_load:
            cache.get("model-a")
            model, _ = cache.get("model-b")

        assert mock_load.call_count == 2
        assert model is mock2[0]

    def test_cache_returns_none_on_import_error(self):
        from escriba.summarize.llm_summary import _LocalModelCache

        cache = _LocalModelCache(ttl=10)

        with patch.dict("sys.modules", {"mlx_lm": None}):
            with patch("builtins.__import__", side_effect=ImportError("no mlx_lm")):
                model, tokenizer = cache.get("test-model")

        assert model is None
        assert tokenizer is None


class TestListAvailableModels:
    """Test model listing with AI availability."""

    def test_includes_local_when_mlx_available(self):
        from escriba.summarize.llm_summary import list_available_models

        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False):
            result = list_available_models()

        assert "local" in result["models"]
        assert result["ai_available"] is True

    def test_ai_unavailable_no_providers(self):
        from escriba.summarize.llm_summary import list_available_models

        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError()) if name == "mlx_lm" else __builtins__.__import__(name, *a, **kw)), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False):
            result = list_available_models()

        assert result["ai_available"] is False
        assert result["ai_unavailable_reason"] is not None
