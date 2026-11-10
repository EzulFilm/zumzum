"""Loudness and true peak to the published definitions.

- Integrated loudness (LUFS): ITU-R BS.1770-4 / EBU R128. K-weighting (a high shelf and a high pass, the two biquads
  of BS.1770 given at 48 kHz, rebuilt at any rate from their own corner, Q and gain), mean square per
  channel over 400 ms blocks every 100 ms, channel weights (1 for L / R / C, 1.41 for the surrounds, LFE left out),
  an absolute gate at -70 LUFS and a relative gate 10 LU under the loudness of the blocks above it.
- True peak (dBTP): BS.1770-4 Annex 2, the peak of the signal upsampled to at least 192 kHz (4x at 48 kHz).
- Loudness range (EBU Tech 3342), from the same blocks.
"""

import math

import numpy as np
from scipy import ndimage, signal

ABS_GATE = -70.0   # LUFS
REL_GATE = -10.0   # LU under the ungated loudness of the blocks above the absolute gate
# BS.1770 channel weights by channel count (ITU order: L R C LFE Ls Rs Lrs Rrs; LFE weight 0)
WEIGHTS = {1: [1.0], 2: [1.0, 1.0], 3: [1.0, 1.0, 1.0], 4: [1.0, 1.0, 1.41, 1.41],
           5: [1.0, 1.0, 1.0, 1.41, 1.41], 6: [1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
           7: [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41],
           8: [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41]}


# BS.1770-4 Table 1 and 2: the K-weighting stages as published, at 48 kHz
SHELF_48K = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
HIGHPASS_48K = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def _at_rate(b, a, sr, shelf):
    """A published 48 kHz stage rebuilt at rate sr. Both stages have the bilinear form with K = tan(pi fc / fs):
    a = [1, 2 (K^2 - 1) / a0, (1 - K/Q + K^2) / a0] with a0 = 1 + K/Q + K^2, the shelf's b = [Vh + Vb K/Q + K^2,
    2 (K^2 - Vh), Vh - Vb K/Q + K^2] / a0, the high pass's b = [1, -2, 1]. So 1 + a1 + a2 = 4 K^2 / a0 and
    1 - a1 + a2 = 4 / a0 give K and K/Q; b0 - b1 + b2 = 4 Vh / a0 and b0 - b2 = 2 Vb (K/Q) / a0 give Vh and Vb.
    The corner fc, Q, Vh and Vb are kept and K is taken again at sr (at 48 kHz this returns the published stage)."""
    a1, a2 = a[1], a[2]
    a0 = 4 / (1 - a1 + a2)
    k = math.sqrt((1 + a1 + a2) / (1 - a1 + a2))
    q = k / (a0 - 1 - k * k)
    fc = math.atan(k) * 48000 / math.pi
    k2 = math.tan(math.pi * fc / sr)
    n0 = 1 + k2 / q + k2 * k2
    a_new = [1.0, 2 * (k2 * k2 - 1) / n0, (1 - k2 / q + k2 * k2) / n0]
    if not shelf:  # the published b is the bilinear one times its a0: keep that gain at every rate
        return np.array(b, float) * a0 / n0, np.array(a_new)
    vh = (b[0] - b[1] + b[2]) * a0 / 4
    vb = (b[0] - b[2]) * a0 / 2 / (k / q)
    return (np.array([(vh + vb * k2 / q + k2 * k2) / n0, 2 * (k2 * k2 - vh) / n0, (vh - vb * k2 / q + k2 * k2) / n0]),
            np.array(a_new))


def k_weight(x, sr):
    """K-weighted copy of x (channels, samples): BS.1770-4's two stages, rebuilt at sr from the published ones."""
    b1, a1 = _at_rate(*SHELF_48K, sr, shelf=True)
    b2, a2 = _at_rate(*HIGHPASS_48K, sr, shelf=False)
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def _weights(nch):
    return np.array(WEIGHTS.get(nch, [1.0] * nch))


def block_loudness(x, sr, block_s=0.4, step_s=0.1):
    """Loudness of each block (LUFS; -inf for digital silence): momentary loudness for 400 ms blocks."""
    x = np.atleast_2d(np.asarray(x, float))
    y = k_weight(x, sr) ** 2
    n, h = int(round(block_s * sr)), int(round(step_s * sr))
    if x.shape[1] < n:
        return np.array([])
    c = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y, axis=1)], axis=1)
    starts = np.arange(0, x.shape[1] - n + 1, h)
    ms = (c[:, starts + n] - c[:, starts]) / n                     # (channels, blocks) mean square
    z = _weights(x.shape[0]) @ ms
    with np.errstate(divide="ignore"):
        return -0.691 + 10 * np.log10(z)


def integrated(x, sr):
    """Integrated loudness, LUFS (BS.1770-4 gating). None when nothing is above the absolute gate (or under 0.4 s)."""
    lk = block_loudness(x, sr)
    lk = lk[lk > ABS_GATE]
    if not len(lk):
        return None
    rel = -0.691 + 10 * np.log10(np.mean(10 ** ((lk + 0.691) / 10))) + REL_GATE
    lk = lk[lk > rel]
    return float(-0.691 + 10 * np.log10(np.mean(10 ** ((lk + 0.691) / 10))))


def loudness_range(x, sr):
    """Loudness range, LU (EBU Tech 3342): short-term loudness (3 s blocks, here every 0.1 s; the definition asks for
    at least 2 s of overlap), gated at -70 LUFS and 20 LU under the power mean of those blocks, then the 95th minus
    the 10th percentile. None under 3 s or when silent."""
    st = block_loudness(x, sr, 3.0, 0.1)
    st = st[st > ABS_GATE]
    if len(st) < 2:
        return None
    rel = 10 * np.log10(np.mean(10 ** (st / 10))) - 20
    st = st[st > rel]
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def _factor(sr):
    return max(1, int(math.ceil(192000 / sr)))   # 4x at 48 kHz, 5x at 44.1 kHz, 1 at 192 kHz


PEAK_BLOCK = 1 << 18


def _peak_env(x, sr):
    """Per original sample: the largest upsampled |value| of any channel within one sample of it.

    Resampling uses overlapped blocks, one channel at a time, so low-rate multichannel audio does not allocate a
    whole-file upsampled array. The 32-sample context covers the polyphase filter at both block edges."""
    x = np.atleast_2d(np.asarray(x, float))
    n = x.shape[1]
    f = _factor(sr)
    if f == 1:
        return ndimage.maximum_filter1d(np.max(np.abs(x), axis=0), 3, mode="nearest")
    env = np.zeros(n)
    for ch in x:
        for a in range(0, n, PEAK_BLOCK):
            b = min(n, a + PEAK_BLOCK)
            lo, hi = max(0, a - 32), min(n, b + 32)
            u = np.abs(signal.resample_poly(ch[lo:hi], f, 1))
            base = u[:(len(u) // f) * f].reshape(-1, f).max(axis=1)
            start = a - lo
            env[a:b] = np.maximum(env[a:b], base[start:start + b - a])
    return ndimage.maximum_filter1d(env, 3, mode="nearest")


def true_peak(x, sr):
    """True peak, dBTP (BS.1770-4 Annex 2: the peak after upsampling to at least 192 kHz); -inf for silence."""
    p = float(np.max(_peak_env(x, sr))) if np.size(x) else 0.0
    return 20 * math.log10(p) if p > 0 else -math.inf
