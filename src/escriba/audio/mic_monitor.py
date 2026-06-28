"""CoreAudio-based microphone activation monitor for macOS.

Uses ctypes to query the default input device's ``IsRunningSomewhere``
property — returns True when *any* process is capturing from the mic.
No audio data is accessed; only a boolean hardware state flag.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import struct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CoreAudio constants (four-char-codes packed as big-endian UInt32)
# ---------------------------------------------------------------------------
_AUDIO_OBJECT_SYSTEM_OBJECT = 1

_PROP_DEFAULT_INPUT = struct.unpack(">I", b"dIn ")[0]
_PROP_IS_RUNNING = struct.unpack(">I", b"gone")[0]
_SCOPE_GLOBAL = struct.unpack(">I", b"glob")[0]
_ELEMENT_MAIN = 0

# Per-process audio API (macOS 14.4+): enumerate which processes are capturing
# input, so we can tell "someone else is on a call" apart from "Escriba's own
# recording is holding the mic open".
_PROP_PROCESS_LIST = struct.unpack(">I", b"prs#")[0]
_PROP_PROCESS_IS_RUNNING_INPUT = struct.unpack(">I", b"piri")[0]
_PROP_PROCESS_PID = struct.unpack(">I", b"ppid")[0]
_PROP_PROCESS_DEVICES = struct.unpack(">I", b"pdv#")[0]
_SCOPE_INPUT = struct.unpack(">I", b"inpt")[0]


# ---------------------------------------------------------------------------
# ctypes structs matching CoreAudio C types
# ---------------------------------------------------------------------------
class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# Library handle (loaded once at module level)
# ---------------------------------------------------------------------------
_lib_path = ctypes.util.find_library("CoreAudio")
_ca: ctypes.CDLL | None = None
if _lib_path:
    try:
        _ca = ctypes.cdll.LoadLibrary(_lib_path)
    except OSError:
        logger.warning("Could not load CoreAudio library — mic monitor disabled")


def _get_property_u32(object_id: int, selector: int) -> int | None:
    """Query a UInt32 audio-object property, returning None on failure."""
    if _ca is None:
        return None

    address = _AudioObjectPropertyAddress(selector, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    data = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(data))

    status = _ca.AudioObjectGetPropertyData(
        ctypes.c_uint32(object_id),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.byref(size),
        ctypes.byref(data),
    )
    if status != 0:
        logger.debug("AudioObjectGetPropertyData failed: selector=0x%08x status=%d", selector, status)
        return None
    return data.value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_default_input_device_id() -> int:
    """Return the AudioObjectID of the default input device, or 0."""
    val = _get_property_u32(_AUDIO_OBJECT_SYSTEM_OBJECT, _PROP_DEFAULT_INPUT)
    return val if val is not None else 0


def is_mic_running() -> bool:
    """Check whether the default input device is being used by any process."""
    device_id = get_default_input_device_id()
    if device_id == 0:
        return False
    val = _get_property_u32(device_id, _PROP_IS_RUNNING)
    return bool(val)


def _get_property_u32_array(
    object_id: int, selector: int, scope: int = _SCOPE_GLOBAL
) -> list[int] | None:
    """Query a variable-length UInt32 array property, or None on failure."""
    if _ca is None:
        return None

    address = _AudioObjectPropertyAddress(selector, scope, _ELEMENT_MAIN)
    size = ctypes.c_uint32(0)
    status = _ca.AudioObjectGetPropertyDataSize(
        ctypes.c_uint32(object_id),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.byref(size),
    )
    if status != 0:
        return None
    if size.value == 0:
        return []

    count = size.value // ctypes.sizeof(ctypes.c_uint32)
    values = (ctypes.c_uint32 * count)()
    status = _ca.AudioObjectGetPropertyData(
        ctypes.c_uint32(object_id),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.byref(size),
        values,
    )
    if status != 0:
        return None
    return list(values)


def _get_process_object_list() -> list[int] | None:
    """Return the CoreAudio process-object ids, or None if unsupported."""
    return _get_property_u32_array(_AUDIO_OBJECT_SYSTEM_OBJECT, _PROP_PROCESS_LIST)


def _process_input_devices(object_id: int) -> list[int]:
    """Return the input-device ids a process is bound to (empty if none/unknown)."""
    devices = _get_property_u32_array(
        object_id, _PROP_PROCESS_DEVICES, _SCOPE_INPUT
    )
    return devices or []


def _process_pid(object_id: int) -> int | None:
    """Return the OS pid behind a CoreAudio process object, or None."""
    address = _AudioObjectPropertyAddress(
        _PROP_PROCESS_PID, _SCOPE_GLOBAL, _ELEMENT_MAIN
    )
    data = ctypes.c_int32(0)
    size = ctypes.c_uint32(ctypes.sizeof(data))
    if _ca is None:
        return None
    status = _ca.AudioObjectGetPropertyData(
        ctypes.c_uint32(object_id),
        ctypes.byref(address),
        ctypes.c_uint32(0),
        None,
        ctypes.byref(size),
        ctypes.byref(data),
    )
    return data.value if status == 0 else None


def external_mic_active(exclude_pid: int | None = None) -> bool | None:
    """
    Whether a process *other than* ``exclude_pid`` is capturing the default mic.

    Uses the macOS 14.4+ per-process audio API so Escriba's own recording
    capture doesn't mask a meeting app releasing the mic on hangup. Only counts
    processes bound to the default input device, which excludes always-on system
    daemons (e.g. ``corespeechd``/Siri) that hold an unrelated input. Returns
    None when the API is unavailable (caller should fall back to
    :func:`is_mic_running`).
    """
    objects = _get_process_object_list()
    if objects is None:
        return None

    default_input = get_default_input_device_id()
    if default_input == 0:
        return None

    for obj in objects:
        if not _get_property_u32(obj, _PROP_PROCESS_IS_RUNNING_INPUT):
            continue
        if exclude_pid is not None and _process_pid(obj) == exclude_pid:
            continue
        if default_input in _process_input_devices(obj):
            return True
    return False


def call_mic_active() -> bool:
    """
    Mic-activation signal for call detection, ignoring Escriba's own capture.

    Prefers the per-process API (excludes this process so an active recording
    doesn't pin the signal True); falls back to the global device flag when the
    process API is unavailable.
    """
    external = external_mic_active(exclude_pid=os.getpid())
    if external is None:
        return is_mic_running()
    return external
