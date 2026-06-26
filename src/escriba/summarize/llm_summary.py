"""LLM-based summarization for transcriptions."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY_SECONDS = 1.0
LOCAL_MODEL_MAX_ATTEMPTS = 2

# Serialize all mlx-lm inference — parallel Metal calls crash the process.
_local_llm_semaphore = threading.Semaphore(1)

T = TypeVar("T")


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
    except (OSError, subprocess.SubprocessError, ValueError):
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
        self._model: Any = None
        self._tokenizer: Any = None
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
                loaded = load(model_id)
                model, tokenizer = loaded[0], loaded[1]
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
            except OSError as e:
                logger.error(
                    "Failed to load local model %s: %s", model_id, e, exc_info=True
                )
                return None, None

    def evict(self) -> None:
        """Drop the cached model so the next call reloads from disk."""
        with self._lock:
            self._evict_unlocked()

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


def _sleep_retry_backoff(attempt: int) -> None:
    """Exponential backoff with jitter between transient cloud retries."""
    delay = LLM_RETRY_BASE_DELAY_SECONDS * (2**attempt)
    jitter = random.uniform(0, delay * 0.25)
    time.sleep(delay + jitter)


def _http_status_from_error(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction from provider SDK exceptions."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


def _is_transient_api_error(exc: BaseException) -> bool:
    """Return True for rate limits and server-side errors worth retrying."""
    try:
        from anthropic import APIStatusError, InternalServerError, RateLimitError

        if isinstance(exc, (RateLimitError, InternalServerError)):
            return True
        if isinstance(exc, APIStatusError):
            status = exc.status_code
            return status == 429 or status >= 500
    except ImportError:
        pass

    status = _http_status_from_error(exc)
    if status is not None:
        if status == 429 or status >= 500:
            return True
        if 400 <= status < 500:
            return False

    name = type(exc).__name__.lower()
    if any(token in name for token in ("ratelimit", "overloaded", "timeout")):
        return True
    if "server" in name and "error" in name:
        return True
    return False


def _call_with_timeout(func: Callable[[], T], timeout_seconds: float) -> T:
    """Run ``func`` on a worker thread and fail fast if it exceeds the deadline."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                f"LLM call timed out after {timeout_seconds}s"
            ) from exc


def _retry_cloud_call(label: str, func: Callable[[], T]) -> T:
    """Retry transient cloud failures with exponential backoff."""
    last_exc: BaseException | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            return _call_with_timeout(func, LLM_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            last_exc = exc
            logger.warning(
                "%s timed out (attempt %d/%d)",
                label,
                attempt + 1,
                LLM_MAX_RETRIES,
            )
        except Exception as exc:
            if not _is_transient_api_error(exc):
                raise
            last_exc = exc
            logger.warning(
                "%s transient failure (attempt %d/%d): %s",
                label,
                attempt + 1,
                LLM_MAX_RETRIES,
                exc,
            )
        if attempt < LLM_MAX_RETRIES - 1:
            _sleep_retry_backoff(attempt)
    assert last_exc is not None
    raise last_exc


def _make_gemini_client(api_key: str):
    """Build a Gemini client with request timeout when the SDK supports it."""
    from google import genai

    try:
        from google.genai import types

        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=LLM_TIMEOUT_SECONDS * 1000),
        )
    except (ImportError, AttributeError, TypeError):
        return genai.Client(api_key=api_key)


def _make_claude_client(api_key: str):
    """Build an Anthropic client with request timeout."""
    from anthropic import Anthropic

    return Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)


def _gemini_generate_text(model_id: str, prompt: str) -> str:
    """Call Gemini generate_content; raises on failure after retries."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    def _call() -> str:
        client = _make_gemini_client(api_key)
        response = client.models.generate_content(model=model_id, contents=prompt)
        if not response.text:
            raise ValueError("Empty response from Gemini")
        return response.text.strip()

    return _retry_cloud_call(f"Gemini({model_id})", _call)


def _claude_generate_text(
    model_id: str, prompt: str, *, max_tokens: int = 100
) -> str:
    """Call Claude messages.create; raises on failure after retries."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    def _call() -> str:
        client = _make_claude_client(api_key)
        message = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if not message.content:
            raise ValueError("Empty response from Claude")
        return message.content[0].text.strip()

    return _retry_cloud_call(f"Claude({model_id})", _call)


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
    return f"""You are an expert meeting and call analyst. Read the transcript and produce a structured summary.

<transcript>
{transcript}
</transcript>

<instructions>
- Base every value strictly on the transcript; do not invent details.
- Write all values in the SAME LANGUAGE as the transcript. If it is in Spanish, use correct orthography: acentos (á, é, í, ó, ú), ñ, ü, and the opening signs ¿ and ¡.
- Leave an array empty ([]) when the transcript has no relevant items for it.
</instructions>

<output_format>
Respond with ONLY a valid JSON object — no markdown, no code fences, no commentary — using exactly this shape:
{{
  "summary": "Executive summary in 3-5 sentences",
  "key_points": ["key point 1", "key point 2"],
  "action_items": [
    {{
      "task": "task description",
      "assignee": "person if mentioned, else empty string",
      "due_date": "date if mentioned, else empty string"
    }}
  ],
  "decisions": ["decision 1", "decision 2"],
  "topics": ["topic 1", "topic 2"]
}}
</output_format>"""


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
        gemini_model = model_id or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        return _generate_summary_gemini(transcript, output_path, model_id=gemini_model)
    elif provider == "claude":
        claude_model = model_id or os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
        return _generate_summary_claude(transcript, output_path, model_id=claude_model)
    else:
        if provider == "none":
            logger.info("No AI provider available — skipping summary")
        else:
            logger.error("Unsupported provider: %s", provider)
        return None


def _parse_summary_json(response_text: str) -> dict[str, Any]:
    """Parse and normalize a JSON summary payload from model text."""
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return json.loads(response_text.strip())


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
        from google import genai  # noqa: F401
    except ImportError:
        logger.error(
            "google-genai not installed. Install with: pip install google-genai"
        )
        return None

    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set")
        return None

    prompt = _build_summary_prompt(transcript)
    try:
        response_text = _gemini_generate_text(model_id, prompt)
        summary_data = _parse_summary_json(response_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON: %s", e)
        return None
    except ValueError as e:
        logger.error("Gemini summary failed: %s", e)
        return None
    except TimeoutError as e:
        logger.error("Gemini summary timed out: %s", e)
        return None
    except Exception as e:
        logger.error("Error generating summary with Gemini: %s", e, exc_info=True)
        return None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info("Summary saved to: %s", output_path)

    return summary_data


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
        from anthropic import Anthropic  # noqa: F401
    except ImportError:
        logger.error(
            "anthropic not installed. Install with: pip install anthropic"
        )
        return None

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return None

    prompt = _build_summary_prompt(transcript)
    try:
        response_text = _claude_generate_text(model_id, prompt, max_tokens=2000)
        summary_data = _parse_summary_json(response_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude response as JSON: %s", e)
        return None
    except ValueError as e:
        logger.error("Claude summary failed: %s", e)
        return None
    except TimeoutError as e:
        logger.error("Claude summary timed out: %s", e)
        return None
    except Exception as e:
        logger.error("Error generating summary with Claude: %s", e, exc_info=True)
        return None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info("Summary saved to: %s", output_path)

    return summary_data


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
        summary_data = _parse_summary_json(response_text)

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


def _run_local_generation(
    prompt: str,
    model_id: str,
    max_tokens: int,
    enable_thinking: bool,
) -> str | None:
    """Run one local MLX generation pass (caller must hold the semaphore)."""
    model, tokenizer = _model_cache.get(model_id)
    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load local model: {model_id}")

    from mlx_lm import generate

    messages = [{"role": "user", "content": prompt}]
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": enable_thinking,
    }
    try:
        chat_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        chat_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    with _model_cache._lock:
        result = generate(model, tokenizer, prompt=chat_prompt, max_tokens=max_tokens)
        _model_cache._last_used = time.time()

    if not result:
        return None
    return _extract_response(result)


def _call_llm_local(
    prompt: str,
    model_id: str,
    max_tokens: int = 256,
    enable_thinking: bool = True,
) -> str | None:
    """Call a local MLX model via the hybrid cache.

    Serialized with a module-level semaphore so concurrent Metal calls cannot
    run in parallel. On OOM/runtime failure the cache is evicted and retried once.
    """
    with _local_llm_semaphore:
        for attempt in range(LOCAL_MODEL_MAX_ATTEMPTS):
            try:
                return _run_local_generation(
                    prompt, model_id, max_tokens, enable_thinking
                )
            except ImportError:
                logger.warning("mlx-lm not installed — local models unavailable")
                return None
            except (MemoryError, RuntimeError) as e:
                logger.error(
                    "Local model generation failed (attempt %d/%d): %s",
                    attempt + 1,
                    LOCAL_MODEL_MAX_ATTEMPTS,
                    e,
                )
                _model_cache.evict()
                if attempt >= LOCAL_MODEL_MAX_ATTEMPTS - 1:
                    return None
            except TypeError as e:
                logger.error("Local model template error: %s", e, exc_info=True)
                return None
            except OSError as e:
                logger.error("Local model I/O error: %s", e, exc_info=True)
                return None
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
    if provider in ("local", "gemini", "claude") and not model_id:
        if provider == "local":
            model_id = recommend_model()
        elif provider == "gemini":
            model_id = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        else:
            model_id = os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
    if provider == "local":
        if not model_id:
            return None
        title = _call_llm_local(
            prompt, model_id, max_tokens=60, enable_thinking=False,
        )
    elif provider == "gemini":
        gemini_model: str = (
            model_id or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        )
        title = _call_llm_gemini(prompt, gemini_model)
    elif provider == "claude":
        claude_model: str = (
            model_id or os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
        )
        title = _call_llm_claude(prompt, claude_model)
    else:
        return None

    if title:
        title = title.strip().strip('"').strip("'").strip()
        return title[:60] if title else None
    return None


def _call_llm_gemini(prompt: str, model_id: str) -> str | None:
    try:
        from google import genai  # noqa: F401
    except ImportError:
        logger.error("google-genai not installed")
        return None
    if not os.getenv("GEMINI_API_KEY"):
        return None
    try:
        return _gemini_generate_text(model_id, prompt)
    except ValueError as e:
        logger.error("Gemini call failed: %s", e)
        return None
    except TimeoutError as e:
        logger.error("Gemini call timed out: %s", e)
        return None
    except Exception as e:
        logger.error("Gemini call failed: %s", e, exc_info=True)
        return None


def _call_llm_claude(prompt: str, model_id: str, max_tokens: int = 100) -> str | None:
    try:
        from anthropic import Anthropic  # noqa: F401
    except ImportError:
        logger.error("anthropic not installed")
        return None
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        return _claude_generate_text(model_id, prompt, max_tokens=max_tokens)
    except ValueError as e:
        logger.error("Claude call failed: %s", e)
        return None
    except TimeoutError as e:
        logger.error("Claude call timed out: %s", e)
        return None
    except Exception as e:
        logger.error("Claude call failed: %s", e, exc_info=True)
        return None


def _build_enhance_prompt(text: str, preserve_placeholders: bool = False) -> str:
    """Meta-prompt: ask the model to refine a prompt used for transcript notes."""
    if preserve_placeholders:
        target = (
            "Improve the SYSTEM PROMPT below. It is a reusable template that wraps "
            "a transcript and a task instruction before an AI writes notes."
        )
        placeholder_rule = (
            "- It MUST keep the literal placeholders {transcript} and {prompt} "
            "exactly as written, including the curly braces. Do not rename, remove, "
            "translate, or duplicate them.\n"
        )
        tag = "system_prompt"
    else:
        target = (
            "Improve the INSTRUCTION below so that, when applied to a meeting or "
            "call transcript, an AI produces clearer, more useful notes."
        )
        placeholder_rule = ""
        tag = "instruction"

    return (
        f"You are a prompt engineer. {target}\n\n"
        "Requirements:\n"
        "- Preserve the author's intent, and respond in the SAME LANGUAGE as the "
        "input.\n"
        "- Make it specific about scope, desired output format, and level of detail.\n"
        f"{placeholder_rule}"
        "- Do not answer or perform it; only rewrite it.\n"
        f"- Return ONLY the improved {tag.replace('_', ' ')} text — no preamble, "
        "quotes, or explanation.\n\n"
        f"<{tag}>\n"
        f"{text}\n"
        f"</{tag}>"
    )


def enhance_prompt(
    text: str, model: str = "auto", preserve_placeholders: bool = False
) -> str | None:
    """Rewrite a user's prompt into a sharper version via the LLM.

    When ``preserve_placeholders`` is set, the result is only returned if it
    still contains the required ``{transcript}`` and ``{prompt}`` placeholders.
    """
    if not text or not text.strip():
        return None

    meta_prompt = _build_enhance_prompt(text.strip(), preserve_placeholders)
    provider, model_id = resolve_provider_and_model(model)
    if provider in ("local", "gemini", "claude") and not model_id:
        if provider == "local":
            model_id = recommend_model()
        elif provider == "gemini":
            model_id = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        else:
            model_id = os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
    if provider == "local":
        if not model_id:
            return None
        result = _call_llm_local(
            meta_prompt, model_id, max_tokens=600, enable_thinking=False
        )
    elif provider == "gemini":
        gemini_model: str = (
            model_id or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        )
        result = _call_llm_gemini(meta_prompt, gemini_model)
    elif provider == "claude":
        claude_model: str = (
            model_id or os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
        )
        result = _call_llm_claude(meta_prompt, claude_model, max_tokens=600)
    else:
        return None
    if not result:
        return None
    result = result.strip().strip('"').strip("'").strip()
    if not result:
        return None
    if preserve_placeholders and (
        "{transcript}" not in result or "{prompt}" not in result
    ):
        logger.debug("Enhanced system prompt lost a placeholder; discarding")
        return None
    return result


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
        from google import genai  # noqa: F401
    except ImportError:
        logger.warning("google-genai SDK not installed")
        return [DEFAULT_GEMINI_MODEL]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return [DEFAULT_GEMINI_MODEL]

    def _call() -> list[str]:
        client = _make_gemini_client(api_key)
        models: list[str] = []
        for model in client.models.list():
            name = model.name
            if name.startswith("models/"):
                name = name[7:]
            if "gemini" in name:
                models.append(name)
        return sorted(models) if models else [DEFAULT_GEMINI_MODEL]

    try:
        return _call_with_timeout(_call, LLM_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.error("Timed out listing Gemini models")
        return [DEFAULT_GEMINI_MODEL]
    except Exception as e:
        logger.error("Failed to list Gemini models: %s", e, exc_info=True)
        return [DEFAULT_GEMINI_MODEL]


def _list_claude_models() -> list[str]:
    try:
        from anthropic import Anthropic  # noqa: F401
    except ImportError:
        logger.warning("anthropic SDK not installed")
        return [DEFAULT_CLAUDE_MODEL]

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return [DEFAULT_CLAUDE_MODEL]

    def _call() -> list[str]:
        client = _make_claude_client(api_key)
        if not hasattr(client, "models"):
            return [DEFAULT_CLAUDE_MODEL]
        response = client.models.list(limit=100)
        models = [m.id for m in response.data if m.id.startswith("claude")]
        return sorted(models) if models else [DEFAULT_CLAUDE_MODEL]

    try:
        return _call_with_timeout(_call, LLM_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.error("Timed out listing Claude models")
        return [DEFAULT_CLAUDE_MODEL]
    except Exception as e:
        logger.error("Failed to list Claude models: %s", e, exc_info=True)
        return [DEFAULT_CLAUDE_MODEL]
