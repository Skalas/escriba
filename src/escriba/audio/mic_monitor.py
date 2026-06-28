"""CoreAudio-based microphone activation monitor for macOS.

Uses ctypes to query the default input device's ``IsRunningSomewhere``
property — returns True when *any* process is capturing from the mic.
No audio data is accessed; only a boolean hardware state flag.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
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
