"""Read and write audio files (libsndfile through soundfile). Arrays are float64 (channels, samples)."""

import numpy as np
import soundfile as sf

from .ops import LIMITS


def check_shape(sr, channels, frames):
    """Raise ValueError if audio of this shape is outside the limits: sample rate 8-192 kHz, at most 8 channels, at
    most an hour, and at most LIMITS["samples"] samples in all (channels x frames)."""
    if not LIMITS["sr_min"] <= sr <= LIMITS["sr_max"]:
        raise ValueError(f"sample rate {sr} Hz is outside {LIMITS['sr_min']}..{LIMITS['sr_max']} Hz")
    if channels > LIMITS["channels"]:
        raise ValueError(f"{channels} channels: at most {LIMITS['channels']}")
    if frames > LIMITS["duration_s"] * sr:
        raise ValueError(f"{frames / sr:.0f} s of audio: at most {LIMITS['duration_s']} s")
    if channels * frames > LIMITS["samples"]:
        raise ValueError(f"{channels} channels x {frames} samples is more than {LIMITS['samples']} samples in all")


def validate(audio, sr):
    """Refuse audio no program may run on (check_shape's limits) or with non-finite samples. Raises ValueError."""
    a = np.atleast_2d(audio)
    check_shape(sr, a.shape[0], a.shape[1])
    if not np.all(np.isfinite(a)):
        raise ValueError("the audio has non-finite samples (inf or NaN)")


def load(path, start_s=0.0, dur_s=None, mono=False):
    info = sf.info(str(path))
    sr = info.samplerate
    start = max(0, int(start_s * sr))
    frames = -1 if dur_s is None else int(dur_s * sr)
    n = max(0, info.frames - start) if frames < 0 else min(frames, max(0, info.frames - start))
    try:  # before reading anything: the header alone can claim an absurd rate, channel count or length
        check_shape(sr, info.channels, n)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    x, sr = sf.read(str(path), start=start, frames=frames, dtype="float64", always_2d=True)
    x = x.T
    if mono:
        x = x.mean(axis=0, keepdims=True)
    validate(x, sr)
    return x, sr


def save(path, audio, sr):
    a = np.atleast_2d(audio)
    sf.write(str(path), np.clip(a, -1, 1).T, sr, subtype="PCM_16")
