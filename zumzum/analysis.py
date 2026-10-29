"""Numbers measured on the audio, the only thing a model learns about it (it never hears the audio).

`analyze(audio, sr)` returns a small dict: duration, levels, octave-band levels and noise floor, quiet spans (noise
profile candidates), mains hum, clicks, and tonal events (steady narrow peaks standing out of the file's usual
spectrum, e.g. a phone tone). `summary(d)` prints it in a few lines of text.
"""

import numpy as np

from .engine import _runs, detect_clicks
from .measure import hum_excess
from .stft import bin_freqs, frame_times, stft

OCTAVES = [31.25 * 2 ** k for k in range(10)]  # lower edges: 31.25 Hz .. 16 kHz


def _db(x):
    return float(10 * np.log10(max(float(x), 1e-30)))


def analyze(audio, sr):
    a = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    mono = a.mean(axis=0)
    n = a.shape[1]
    size, hop = 4096, 1024
    P = sum(np.abs(stft(ch, size, hop)) ** 2 for ch in a) / a.shape[0]
    freqs, times = bin_freqs(size, sr), frame_times(P.shape[1], size, hop, sr)
    inside = (times >= 0) & (times <= n / sr)
    P, times = P[:, inside], times[inside]
    frame_db = 10 * np.log10(P.sum(axis=0) + 1e-30)
    quiet = frame_db <= np.percentile(frame_db, 10) + 3
    spans = [(times[s], times[e]) for s, e in _runs(quiet) if times[e] - times[s] >= 0.3]
    spans.sort(key=lambda s: s[0] - s[1])
    k = max(5, int(0.1 * P.shape[1]))
    floor_frames = np.argsort(P.sum(axis=0))[:k]
    norm = (size * 0.375) * size / 2  # Hann power sum: a full-scale sine reads about 0 dB in its bin group
    bands, floor = {}, {}
    for lo in OCTAVES:
        b = (freqs >= lo) & (freqs < min(2 * lo, sr / 2))
        if b.any():
            bands[f"{lo:g}"] = round(_db(P[b].sum(axis=0).mean() / norm), 1)
            floor[f"{lo:g}"] = round(_db(P[b][:, floor_frames].sum(axis=0).mean() / norm), 1)
    hum = None
    quarters = [a[:, i * n // 4:(i + 1) * n // 4] for i in range(4)] if n >= 4 * sr else [a]
    for f0 in (50.0, 60.0):
        f, _, _ = hum_excess(a, sr, f0, 10, search=1.0)
        above = 1e9  # hum is steady: it must stand out in every quarter of the file (a held low note does not)
        exc = 0.0
        for q in quarters:
            _, e, flo = hum_excess(q, sr, f, 10, search=0)
            exc += e
            above = min(above, _db(e) - _db(flo) if flo > 0 and e > 0 else -99)
        if exc > 0 and above > 0 and (hum is None or above > hum["above_floor_db"]):
            hum = {"f0_hz": round(f, 2), "above_floor_db": round(above, 1)}
    clicks = sum(len(detect_clicks(ch, sr, 10)) for ch in a)
    return {
        "duration_s": round(n / sr, 3), "sr": sr, "channels": a.shape[0],
        "peak_dbfs": round(_db(np.max(np.abs(a)) ** 2), 1), "rms_dbfs": round(_db(np.mean(mono ** 2)), 1),
        "octave_dbfs": bands, "noise_floor_dbfs": floor,
        "quiet_spans_s": [[round(s, 2), round(e, 2)] for s, e in spans[:3]],
        "hum": hum if hum and hum["above_floor_db"] >= 6 else None,
        "clicks": clicks, "clicks_per_min": round(clicks / max(n / sr / 60, 1e-9), 1),
        "tonal_events": tonal_events(P, freqs, times, max_events=16),
    }


def tonal_events(P, freqs, times, min_s=0.12, rise_db=15, max_events=8, max_drift_bins=3):
    """Steady narrow peaks: a local spectral maximum standing rise_db above the file's median at that frequency and
    12 dB above its neighbouring bins, held for at least min_s while drifting no more than max_drift_bins (speech
    harmonics glide and are left out; a phone tone, a whistle or a held note stays). Returns
    [{"t_s": [a, b], "f_hz": f, "above_db": x}] strongest first."""
    from scipy import ndimage
    L = 10 * np.log10(P + 1e-30)
    med_t = np.median(L, axis=1, keepdims=True)
    local = ndimage.median_filter(L, size=(31, 1), mode="nearest")
    peak = (L > med_t + rise_db) & (L > local + 12)
    peak &= (L >= np.roll(L, 1, axis=0)) & (L >= np.roll(L, -1, axis=0))
    peak[(freqs < 80) | (freqs > 12000)] = False
    lab, nlab = ndimage.label(peak, structure=np.ones((3, 3)))  # one bin of drift from frame to frame
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        fs, ts = sl
        dur = (ts.stop - ts.start) * dt
        if dur < min_s or fs.stop - fs.start > max_drift_bins:
            continue
        reg = lab[sl] == i
        sub = (L - med_t)[sl]
        f_idx = fs.start + int(np.argmax(np.where(reg, sub, -1e9).max(axis=1)))
        out.append({"t_s": [round(times[ts.start] - dt / 2, 2), round(times[ts.stop - 1] + dt / 2, 2)],
                    "f_hz": round(float(freqs[f_idx]), 1), "above_db": round(float(np.where(reg, sub, 0).max()), 1)})
    out.sort(key=lambda e: -e["above_db"])
    return out[:max_events]


def summary(d):
    lines = [f"duration {d['duration_s']} s, {d['sr']} Hz, {d['channels']} channel(s); peak {d['peak_dbfs']} dBFS, "
             f"rms {d['rms_dbfs']} dBFS",
             "octave levels dBFS (band from Hz): " + ", ".join(f"{k}:{v}" for k, v in d["octave_dbfs"].items()),
             "noise floor dBFS (quietest frames): " + ", ".join(f"{k}:{v}" for k, v in d["noise_floor_dbfs"].items()),
             "quiet spans (s): " + (", ".join(f"{a}..{b}" for a, b in d["quiet_spans_s"]) or "none")]
    h = d["hum"]
    lines.append(f"steady low tone (mains hum or a held low note): {h['f0_hz']} Hz, harmonics {h['above_floor_db']} dB "
                 "above the neighbouring bins"
                 if h else "mains hum: none found")
    lines.append(f"clicks: {d['clicks']} ({d['clicks_per_min']} per minute)")
    ev = d["tonal_events"][:8]
    lines.append("tonal events: " + ("; ".join(f"{e['t_s'][0]}..{e['t_s'][1]} s at {e['f_hz']} Hz "
                                                 f"(+{e['above_db']} dB)" for e in ev) if ev else "none"))
    return "\n".join(lines)
