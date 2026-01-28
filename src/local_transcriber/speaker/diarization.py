from __future__ import annotations

import logging
import os
import wave
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Suppress torchcodec warnings globally for this module
# We use audio preloaded in memory, so torchcodec is not needed
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*libtorchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")


@dataclass(frozen=True)
class SpeakerTurn:
    """
    A continuous time interval spoken by a given speaker.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        speaker: Speaker label (e.g. 'SPEAKER_00').
    """

    start: float
    end: float
    speaker: str


def diarize_wav(
    wav_path: Path,
    *,
    huggingface_token: str | None = None,
    pipeline_id: str = "pyannote/speaker-diarization-3.1",
) -> list[SpeakerTurn]:
    """
    Run speaker diarization with pyannote-audio on a WAV file.

    This requires `pyannote-audio` and a Hugging Face token with access to the model.

    Args:
        wav_path: Path to a WAV file.
        huggingface_token: Hugging Face token. If None, uses $HUGGINGFACE_TOKEN.
        pipeline_id: Pyannote pipeline identifier.

    Returns:
        List of diarization turns.

    Raises:
        FileNotFoundError: If wav_path does not exist.
        RuntimeError: If dependencies are missing or diarization fails.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Check file size - pyannote needs at least some audio
    file_size = wav_path.stat().st_size
    if file_size < 1000:  # Less than 1KB is probably too small
        raise RuntimeError(
            f"Audio file too small ({file_size} bytes). "
            f"Pyannote requires at least a few seconds of audio."
        )

    # Check audio duration - pyannote needs at least 2-3 seconds
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / sample_rate if sample_rate > 0 else 0.0

            if duration < 2.0:  # Less than 2 seconds
                raise RuntimeError(
                    f"Audio duration too short ({duration:.2f} seconds). "
                    f"Pyannote requires at least 2-3 seconds of audio for reliable diarization."
                )
            logger.debug(
                f"Audio file duration: {duration:.2f} seconds, sample rate: {sample_rate} Hz"
            )
    except wave.Error as e:
        logger.warning(
            f"Could not read WAV file to check duration: {e}. Proceeding anyway..."
        )
    except Exception as e:
        logger.debug(f"Error checking audio duration: {e}. Proceeding anyway...")

    token = (huggingface_token or os.getenv("HUGGINGFACE_TOKEN", "")).strip()
    if not token:
        raise RuntimeError(
            "HuggingFace token required for pyannote diarization. "
            "Set HUGGINGFACE_TOKEN in your environment."
        )

    # Suppress torchcodec warnings before importing pyannote
    # The warning is generated during module initialization, so we need to suppress it globally
    import warnings

    original_showwarning = warnings.showwarning

    def filtered_warning(message, category, filename, lineno, file=None, line=None):
        """Filter out torchcodec warnings."""
        msg_str = str(message)
        if "torchcodec" in msg_str.lower() or "libtorchcodec" in msg_str.lower():
            return  # Suppress this warning
        # Show other warnings normally
        original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = filtered_warning

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        # Restore original warning handler before raising
        warnings.showwarning = original_showwarning
        raise RuntimeError(
            "pyannote-audio is not installed. Install an optional diarization extra "
            "or run: uv pip install pyannote-audio"
        ) from exc
    finally:
        # Restore original warning handler
        warnings.showwarning = original_showwarning

    try:
        logger.info("Loading pyannote pipeline: %s", pipeline_id)
        # Suppress torchcodec warnings when loading pipeline
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*torchcodec.*")
            warnings.filterwarnings("ignore", module="pyannote.audio.core.io")
            warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
            # Use 'token' instead of deprecated 'use_auth_token'
            pipeline = Pipeline.from_pretrained(pipeline_id, token=token)
    except Exception as exc:
        # Check for gated repo access errors
        error_msg = str(exc)
        if (
            "403" in error_msg
            or "gated" in error_msg.lower()
            or "GatedRepoError" in str(type(exc))
        ):
            raise RuntimeError(
                f"Access denied to pyannote models. You need to:\n"
                f"1. Visit https://hf.co/pyannote/speaker-diarization-3.1 and accept the terms\n"
                f"2. Visit https://hf.co/pyannote/segmentation-3.0 and accept the terms\n"
                f"3. Visit https://hf.co/pyannote/embedding and accept the terms\n"
                f"4. Ensure HUGGINGFACE_TOKEN is set with a valid token\n"
                f"5. The token must have 'read' permissions\n"
                f"\nOriginal error: {exc}"
            ) from exc
        raise RuntimeError(
            f"Failed to load pyannote pipeline {pipeline_id!r}: {exc}"
        ) from exc

    try:
        logger.info("Running diarization on audio file: %s", wav_path)

        # Load audio manually to avoid AudioDecoder issues with torchcodec
        # Pyannote can work with audio preloaded in memory as a dict
        try:
            import torch
            import torchaudio

            # Use torchaudio to load audio
            waveform, sample_rate = torchaudio.load(str(wav_path))
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        except ImportError:
            # Fallback to soundfile if torchaudio not available
            try:
                import soundfile as sf
                import numpy as np

                audio_data, sample_rate = sf.read(str(wav_path))
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                # Convert to torch tensor
                import torch

                waveform = torch.from_numpy(audio_data).float()
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)  # Add channel dimension
            except ImportError:
                raise RuntimeError(
                    "Need either torchaudio or soundfile to load audio. "
                    "Install with: pip install torchaudio or pip install soundfile"
                )

        # Prepare audio dict for pyannote (avoids AudioDecoder)
        audio_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

        # Suppress torchcodec warnings - we're using audio preloaded in memory
        # so torchcodec is not needed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*torchcodec.*")
            warnings.filterwarnings("ignore", module="pyannote.audio.core.io")
            warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
            diarization = pipeline(audio_dict)

        # pyannote pipelines may return either:
        # - Annotation-like object with itertracks()
        # - A dict containing {"diarization": Annotation, ...}
        # - A DiarizeOutput dataclass with attributes:
        #   - speaker_diarization: Annotation
        #   - exclusive_speaker_diarization: Annotation
        diarization_annotation = None
        if hasattr(diarization, "itertracks"):
            diarization_annotation = diarization
        elif isinstance(diarization, dict) and "diarization" in diarization:
            diarization_annotation = diarization["diarization"]
        elif hasattr(diarization, "speaker_diarization"):
            diarization_annotation = getattr(diarization, "speaker_diarization")
        elif hasattr(diarization, "exclusive_speaker_diarization"):
            diarization_annotation = getattr(
                diarization, "exclusive_speaker_diarization"
            )

        if diarization_annotation is None or not hasattr(
            diarization_annotation, "itertracks"
        ):
            raise RuntimeError(
                f"Unexpected pyannote diarization output type: {type(diarization).__name__}. "
                "Could not extract an Annotation with itertracks()."
            )

        turns: list[SpeakerTurn] = []
        for segment, _, speaker in diarization_annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(speaker),
                )
            )
        logger.info("pyannote diarization produced %d turns", len(turns))
        return turns
    except Exception as exc:
        error_type = type(exc).__name__
        error_msg = str(exc)
        logger.error(
            f"pyannote diarization failed: {error_type}: {error_msg}", exc_info=True
        )
        raise RuntimeError(
            f"pyannote diarization failed: {error_type}: {error_msg}\n"
            f"This might be due to:\n"
            f"- Audio file format issues\n"
            f"- Insufficient audio duration (needs at least a few seconds)\n"
            f"- torchcodec/FFmpeg compatibility issues\n"
            f"\nTry using audio preloaded in memory or check the audio file format."
        ) from exc


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    turns: list[SpeakerTurn],
    *,
    min_overlap_seconds: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Assign speaker labels to transcription segments based on time overlap.

    Args:
        segments: Transcription segments, each with 'start' and 'end' (seconds).
        turns: Diarization turns from pyannote.
        min_overlap_seconds: Minimum overlap required to assign a speaker.

    Returns:
        New list of segments with 'speaker' field when assignment is possible.
    """
    if not segments or not turns:
        return list(segments)

    def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    out: list[dict[str, Any]] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        best_speaker: str | None = None
        best_overlap = 0.0

        for t in turns:
            ov = overlap(start, end, t.start, t.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = t.speaker

        new_seg = dict(seg)
        if best_speaker is not None and best_overlap >= min_overlap_seconds:
            new_seg["speaker"] = best_speaker
        out.append(new_seg)

    return out
