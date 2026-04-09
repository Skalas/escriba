"""LLM-based summarization for transcriptions."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6-20250514"

# Local model presets ordered by RAM requirement (ascending)
LOCAL_MODEL_PRESETS: list[tuple[int, str]] = [
    (8, "mlx-community/gemma-4-e4b-it-4bit"),
    (16, "mlx-community/gemma-4-26b-a4b-it-4bit"),
]


def get_system_ram_gb() -> int:
    """Return total system RAM in GB (macOS only)."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) // (1024 ** 3)
    except Exception:
        logger.debug("Failed to detect system RAM", exc_info=True)
        return 0


def recommend_model() -> str | None:
    """Pick the best local model for this machine's RAM.

    Returns a HuggingFace repo ID or None if RAM is too low.
    """
    ram_gb = get_system_ram_gb()
    best = None
    for min_ram, repo_id in LOCAL_MODEL_PRESETS:
        if ram_gb >= min_ram:
            best = repo_id
    return best


class _LocalModelCache:
    """Hybrid cache: lazy-load on first call, evict after TTL of inactivity."""

    def __init__(self, ttl: int = 300):
        self._model = None
        self._tokenizer = None
        self._model_id: str | None = None
        self._last_used: float = 0
        self._ttl = ttl
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def get(self, model_id: str):
        """Return (model, tokenizer), loading if needed. Returns (None, None) on failure."""
        with self._lock:
            if self._model is not None and self._model_id == model_id:
                self._last_used = time.time()
                return self._model, self._tokenizer

            # Different model requested — evict old one
            if self._model is not None and self._model_id != model_id:
                self._evict_unlocked()

            try:
                from mlx_lm import load

                logger.info("Loading local model: %s", model_id)
                model, tokenizer = load(model_id)
                self._model = model
                self._tokenizer = tokenizer
                self._model_id = model_id
                self._last_used = time.time()
                self._start_eviction_timer()
                logger.info("Local model loaded: %s", model_id)
                return model, tokenizer
            except ImportError:
                logger.warning("mlx-lm not installed — local models unavailable")
                return None, None
            except (MemoryError, RuntimeError) as e:
                logger.error("Failed to load model %s (likely OOM): %s", model_id, e)
                self._evict_unlocked()
                return None, None
            except Exception:
                logger.error("Failed to load local model %s", model_id, exc_info=True)
                return None, None

    def _evict_unlocked(self):
        """Free model memory. Must be called under self._lock."""
        self._model = None
        self._tokenizer = None
        self._model_id = None
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _start_eviction_timer(self):
        if self._timer:
            self._timer.cancel()

        def _check():
            with self._lock:
                if self._model is None:
                    return
                if time.time() - self._last_used > self._ttl:
                    logger.info("Evicting local model after %ds idle", self._ttl)
                    self._evict_unlocked()
                else:
                    self._start_eviction_timer()

        self._timer = threading.Timer(60, _check)
        self._timer.daemon = True
        self._timer.start()


_model_cache = _LocalModelCache()


def configure_local_cache(ttl: int = 300) -> None:
    """Update the cache TTL (called from app startup with config values)."""
    global _model_cache
    _model_cache = _LocalModelCache(ttl=ttl)


def resolve_provider_and_model(model: str) -> tuple[str, str | None]:
    """Resolve a model string to (provider, model_id).

    Accepts a provider shorthand ("gemini", "claude", "local", "auto") or a
    full model ID like "gemini-2.5-flash" or "mlx-community/gemma-4-e4b-it-4bit".

    "auto" cascade: local (if RAM sufficient + mlx-lm available) → gemini
    (if key set) → claude (if key set) → (None, None).
    """
    m = model.lower().strip()

    if m == "auto":
        return _resolve_auto()
    if m == "local":
        repo = os.getenv("LOCAL_LLM_MODEL", "").strip() or recommend_model()
        return ("local", repo) if repo else _resolve_auto_remote()
    if m.startswith("mlx-community/") or m.startswith("mlx-community\\"):
        return "local", m
    if m == "gemini":
        return "gemini", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if m == "claude":
        return "claude", os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL)
    if m.startswith("gemini"):
        return "gemini", m
    if m.startswith("claude"):
        return "claude", m
    # Unknown — try gemini as fallback
    logger.warning("Unknown model prefix %r, assuming gemini", model)
    return "gemini", m


def _resolve_auto() -> tuple[str, str | None]:
    """Auto-select: local first, then remote if keys available, else None."""
    # Try local
    recommended = recommend_model()
    if recommended:
        try:
            import mlx_lm as _  # noqa: F401

            return "local", recommended
        except ImportError:
            pass

    return _resolve_auto_remote()


def _resolve_auto_remote() -> tuple[str, str | None]:
    """Fall back to remote providers if API keys are set."""
    if os.getenv("GEMINI_API_KEY", "").strip():
        return "gemini", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "claude", os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL)
    return "none", None


def _build_summary_prompt(transcript: str) -> str:
    return f"""Analyze this meeting/call transcript and generate a structured summary as JSON.

IMPORTANT: Respond in the SAME LANGUAGE as the transcript.

Transcript:
{transcript}

Generate a JSON with this structure (use the transcript's language for all values):
{{
  "summary": "Executive summary in 3-5 sentences",
  "key_points": ["key point 1", "key point 2", ...],
  "action_items": [
    {{
      "task": "task description",
      "assignee": "person if mentioned",
      "due_date": "date if mentioned"
    }}
  ],
  "decisions": ["decision 1", "decision 2", ...],
  "topics": ["topic 1", "topic 2", ...]
}}

Respond ONLY with the JSON, no additional text."""


def generate_summary(
    transcript: str,
    model: str = "gemini",
    output_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Genera un resumen de la transcripción usando un LLM.

    Accepts a provider name ("gemini"/"claude") or a full model ID.
    """
    provider, model_id = resolve_provider_and_model(model)
    if provider == "local":
        return _generate_summary_local(transcript, output_path, model_id=model_id)
    elif provider == "gemini":
        return _generate_summary_gemini(transcript, output_path, model_id=model_id)
    elif provider == "claude":
        return _generate_summary_claude(transcript, output_path, model_id=model_id)
    else:
        if provider == "none":
            logger.info("No AI provider available — skipping summary")
        else:
            logger.error("Unsupported provider: %s", provider)
        return None


def _generate_summary_gemini(
    transcript: str, output_path: Path | None = None, *, model_id: str = DEFAULT_GEMINI_MODEL
) -> dict[str, Any] | None:
    """
    Genera resumen usando Google Gemini API.

    Args:
        transcript: Texto de la transcripción
        output_path: Ruta opcional donde guardar el resumen

    Returns:
        Diccionario con el resumen estructurado o None si falla
    """
    try:
        from google import genai
    except ImportError:
        logger.error(
            "google-genai not installed. Install with: pip install google-genai"
        )
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None

    try:
        client = genai.Client(api_key=api_key)

        prompt = _build_summary_prompt(transcript)

        response = client.models.generate_content(model=model_id, contents=prompt)
        response_text = response.text.strip()

        # Limpiar respuesta (puede tener markdown code blocks)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parsear JSON
        summary_data = json.loads(response_text)

        # Guardar si se especificó output_path
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info("Summary saved to: %s", output_path)

        return summary_data

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON: %s", e)
        logger.debug("Response text: %s", response_text[:500])
        return None
    except Exception as e:
        logger.error("Error generating summary with Gemini: %s", e, exc_info=True)
        return None


def _generate_summary_claude(
    transcript: str, output_path: Path | None = None, *, model_id: str = DEFAULT_CLAUDE_MODEL
) -> dict[str, Any] | None:
    """
    Genera resumen usando Anthropic Claude API.

    Args:
        transcript: Texto de la transcripción
        output_path: Ruta opcional donde guardar el resumen

    Returns:
        Diccionario con el resumen estructurado o None si falla
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error(
            "anthropic not installed. Install with: pip install anthropic"
        )
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return None

    try:
        client = Anthropic(api_key=api_key)

        prompt = _build_summary_prompt(transcript)

        message = client.messages.create(
            model=model_id,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text.strip()

        # Limpiar respuesta (puede tener markdown code blocks)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parsear JSON
        summary_data = json.loads(response_text)

        # Guardar si se especificó output_path
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info("Summary saved to: %s", output_path)

        return summary_data

    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude response as JSON: %s", e)
        logger.debug("Response text: %s", response_text[:500])
        return None
    except Exception as e:
        logger.error("Error generating summary with Claude: %s", e, exc_info=True)
        return None


def _generate_summary_local(
    transcript: str, output_path: Path | None = None, *, model_id: str | None = None
) -> dict[str, Any] | None:
    """Generate summary using a local MLX model."""
    if not model_id:
        model_id = recommend_model()
    if not model_id:
        logger.error("No local model available for this hardware")
        return None

    response_text = _call_llm_local(
        _build_summary_prompt(transcript), model_id, max_tokens=4096
    )
    if not response_text:
        return None

    try:
        # Clean markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        summary_data = json.loads(response_text)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info("Summary saved to: %s", output_path)

        return summary_data
    except json.JSONDecodeError as e:
        logger.error("Failed to parse local model response as JSON: %s", e)
        logger.debug("Response text: %s", response_text[:500])
        return None


def _extract_response(text: str) -> str:
    """Extract the response portion from model output, stripping thinking blocks."""
    import re

    # Gemma 4 outputs: <|channel>thought\n...<|channel>response\n...
    # We want only the content after the last <|channel>response or <|channel>default
    match = re.search(r"<\|channel\>(?:response|default)\s*\n?(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no channel markers found, check for <channel|> end markers
    # and return everything after the last one
    if "<channel|>" in text:
        parts = text.split("<channel|>")
        return parts[-1].strip()

    # If only thinking channel is present (model ran out of tokens before
    # emitting the response channel), strip the thinking block entirely.
    thought_match = re.search(r"<\|channel\>thought\s*\n?(.*)", text, re.DOTALL)
    if thought_match:
        # The model never produced a response — return empty so callers
        # know the generation was incomplete.
        return ""

    # Strip any remaining special tokens the model may have leaked.
    cleaned = re.sub(r"<\|[^>]*\>", "", text).strip()
    return cleaned


def _call_llm_local(
    prompt: str,
    model_id: str,
    max_tokens: int = 256,
    enable_thinking: bool = True,
) -> str | None:
    """Call a local MLX model via the hybrid cache."""
    model, tokenizer = _model_cache.get(model_id)
    if model is None:
        return None
    try:
        from mlx_lm import generate

        messages = [{"role": "user", "content": prompt}]
        template_kwargs: dict = {
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": enable_thinking,
        }
        try:
            chat_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Tokenizer doesn't support enable_thinking kwarg
            template_kwargs.pop("enable_thinking", None)
            chat_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

        result = generate(model, tokenizer, prompt=chat_prompt, max_tokens=max_tokens)
        if not result:
            return None
        return _extract_response(result)
    except Exception:
        logger.error("Local model generation failed", exc_info=True)
        return None


def generate_session_title(
    transcript_snippet: str,
    app_name: str | None = None,
    model: str = "gemini",
) -> str | None:
    """Generate a short descriptive title (3-7 words) for a session.

    Returns the title string or None on failure.
    """
    if not transcript_snippet.strip():
        return None

    context = app_name or "meeting"
    prompt = (
        f"Given this transcript snippet from a {context}, "
        "generate a short descriptive title (3-7 words). "
        "Respond in the SAME LANGUAGE as the transcript. "
        "Respond with ONLY the title, no quotes or extra text.\n\n"
        f"Transcript:\n{transcript_snippet}"
    )

    provider, model_id = resolve_provider_and_model(model)
    try:
        if provider == "local":
            title = _call_llm_local(
                prompt, model_id, max_tokens=60, enable_thinking=False,
            )
        elif provider == "gemini":
            title = _call_llm_gemini(prompt, model_id)
        elif provider == "claude":
            title = _call_llm_claude(prompt, model_id)
        else:
            return None

        if title:
            title = title.strip().strip('"').strip("'").strip()
            return title[:60] if title else None
        return None
    except Exception:
        logger.debug("Failed to generate session title", exc_info=True)
        return None


def _call_llm_gemini(prompt: str, model_id: str) -> str | None:
    try:
        from google import genai
    except ImportError:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_id, contents=prompt)
    return response.text.strip() if response.text else None


def _call_llm_claude(prompt: str, model_id: str) -> str | None:
    try:
        from anthropic import Anthropic
    except ImportError:
        return None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model_id,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip() if message.content else None


def list_available_models() -> dict[str, Any]:
    """Fetch available model IDs from all configured providers.

    Returns a dict with model lists, recommended model, and AI availability status.
    """
    models: dict[str, list[str]] = {}

    # Local models
    local_models = [repo for _, repo in LOCAL_MODEL_PRESETS]
    mlx_available = False
    try:
        import mlx_lm as _  # noqa: F401

        mlx_available = True
    except ImportError:
        pass

    if mlx_available:
        models["local"] = local_models

    # Remote models
    if os.getenv("GEMINI_API_KEY", "").strip():
        models["gemini"] = _list_gemini_models()

    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        models["claude"] = _list_claude_models()

    # Recommendation and availability
    recommended = recommend_model() if mlx_available else None
    ai_available = bool(models)
    reason = None
    if not ai_available:
        reason = (
            "AI features (session naming, summaries) need a local model "
            "(install mlx-lm) or an API key (GEMINI_API_KEY / ANTHROPIC_API_KEY)."
        )

    return {
        "models": models,
        "recommended": recommended,
        "ai_available": ai_available,
        "ai_unavailable_reason": reason,
    }


def _list_gemini_models() -> list[str]:
    try:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        models = []
        for model in client.models.list():
            name = model.name
            # API returns "models/gemini-..." — strip the prefix
            if name.startswith("models/"):
                name = name[7:]
            # Only include generative models (skip embedding, etc.)
            if "gemini" in name:
                models.append(name)
        return sorted(models) if models else [DEFAULT_GEMINI_MODEL]
    except ImportError:
        logger.warning("google-genai SDK not installed")
        return [DEFAULT_GEMINI_MODEL]
    except Exception as e:
        logger.error("Failed to list Gemini models: %s", e)
        return [DEFAULT_GEMINI_MODEL]


def _list_claude_models() -> list[str]:
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # models.list() requires anthropic >= 0.39.0
        if not hasattr(client, "models"):
            return [DEFAULT_CLAUDE_MODEL]
        response = client.models.list(limit=100)
        models = [m.id for m in response.data if m.id.startswith("claude")]
        return sorted(models) if models else [DEFAULT_CLAUDE_MODEL]
    except ImportError:
        logger.warning("anthropic SDK not installed")
        return [DEFAULT_CLAUDE_MODEL]
    except Exception as e:
        logger.error("Failed to list Claude models: %s", e)
        return [DEFAULT_CLAUDE_MODEL]
