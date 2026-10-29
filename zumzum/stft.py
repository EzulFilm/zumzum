"""STFT / ISTFT of the reference engine (Allen 1977; least-squares overlap-add inverse, Griffin & Lim 1984).

Periodic Hann analysis and synthesis windows, hop = size / overlap. The signal is padded with `size` zeros at both ends
so every input sample is covered by `overlap` full frames, and the inverse divides by the summed squared window, so
istft(stft(x)) == x to float rounding for any overlap >= 2 (tested). Frame t is centred at sample t * hop - size / 2 of
the unpadded signal, i.e. at time (t * hop - size / 2) / sr seconds.
"""

import numpy as np


def window(size):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)


def n_frames(n, size, hop):
    return (n + 2 * size - size) // hop + 1


def stft(x, size, hop):
    """x: (n,) float -> complex (size // 2 + 1, frames)."""
    x = np.asarray(x, dtype=np.float64)
    w = window(size)
    xp = np.concatenate([np.zeros(size), x, np.zeros(size + hop)])
    t = n_frames(len(x), size, hop)
    idx = np.arange(size)[None, :] + hop * np.arange(t)[:, None]
    return np.fft.rfft(xp[idx] * w, axis=1).T


def istft(X, size, hop, n):
    """Inverse of `stft` for a signal of n samples (least-squares for a modified X)."""
    w = window(size)
    frames = np.fft.irfft(X.T, n=size, axis=1) * w
    t = frames.shape[0]
    total = hop * (t - 1) + size
    out, wsum = np.zeros(total), np.zeros(total)
    for i in range(t):  # overlap-add; t is at most a few hundred thousand, each step is a vector add
        out[i * hop:i * hop + size] += frames[i]
        wsum[i * hop:i * hop + size] += w * w
    good = wsum > 1e-10
    out[good] /= wsum[good]
    out[~good] = 0.0
    return out[size:size + n]


def frame_times(frames, size, hop, sr):
    """Centre time (s) of each frame."""
    return (np.arange(frames) * hop - size / 2) / sr


def bin_freqs(size, sr):
    return np.arange(size // 2 + 1) * sr / size
