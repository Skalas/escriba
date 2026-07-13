"""Local inference load vs generation timeout split (#108)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from escriba.summarize import llm_summary
from escriba.summarize.llm_summary import _LocalInferenceProcess


def test_generation_timeout_uses_generation_budget_only() -> None:
    """Hung generate raises with generation timeout message, not load timeout."""
    proc = _LocalInferenceProcess()
    mock_future = MagicMock()
    mock_future.result.side_effect = TimeoutError(
        f"Local inference timed out after {llm_summary._LOCAL_GENERATION_TIMEOUT}s"
    )

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future
    proc._executor = mock_executor

    with patch.object(proc, "_get_executor", return_value=mock_executor):
        with pytest.raises(TimeoutError, match="Local inference timed out after"):
            proc.run("prompt", "model", 100, False)

    mock_executor.shutdown.assert_called()


def test_load_timeout_is_distinct_from_generation() -> None:
    """Cold model load uses the load deadline, not the generation budget."""
    proc = _LocalInferenceProcess()
    mock_future = MagicMock()
    mock_future.result.side_effect = TimeoutError(
        f"Local model load timed out after {llm_summary._LOCAL_MODEL_LOAD_TIMEOUT}s"
    )

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future
    proc._executor = mock_executor

    with patch.object(proc, "_get_executor", return_value=mock_executor):
        with pytest.raises(TimeoutError, match="Local model load timed out after"):
            proc.run("prompt", "model", 100, False)

    assert mock_executor.submit.call_count == 1


def test_inference_uses_single_atomic_worker_job() -> None:
    """Load and generate run in one executor job so callers cannot interleave."""
    proc = _LocalInferenceProcess()
    mock_future = MagicMock()
    mock_future.result.return_value = "notes text"

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future

    with patch.object(proc, "_get_executor", return_value=mock_executor):
        result = proc.run("prompt", "model", 100, False)

    assert result == "notes text"
    assert mock_executor.submit.call_count == 1
    job_timeout = llm_summary._LOCAL_MODEL_LOAD_TIMEOUT + llm_summary._LOCAL_GENERATION_TIMEOUT
    parent_timeout = job_timeout + llm_summary._LOCAL_INFERENCE_PARENT_GRACE_SECONDS
    mock_future.result.assert_called_once_with(timeout=parent_timeout)
