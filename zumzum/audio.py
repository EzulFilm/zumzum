"""Read and write audio files (libsndfile through soundfile). Arrays are float64 (channels, samples)."""

import numpy as np
import soundfile as sf


def load(path, start_s=0.0, dur_s=None, mono=False):
    info = sf.info(str(path))
    sr = info.samplerate
    start = int(start_s * sr)
    frames = -1 if dur_s is None else int(dur_s * sr)
    x, sr = sf.read(str(path), start=start, frames=frames, dtype="float64", always_2d=True)
    x = x.T
    if mono:
        x = x.mean(axis=0, keepdims=True)
    return x, sr


def save(path, audio, sr):
    a = np.atleast_2d(audio)
    sf.write(str(path), np.clip(a, -1, 1).T, sr, subtype="PCM_16")
