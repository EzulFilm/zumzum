"""Measured checks on a before / after pair (docs/spec.md section 6) and the damage gate.

Every check returns {"name", "check", "pass", "value", "text"}: `value` is the number measured, `text` says it in one
line a person (or a model) can act on.
"""

import numpy as np
from scipy import ndimage, signal

from .engine import detect_clicks, sel_mask
from .stft import bin_freqs, frame_times, stft


def _spec_power(a, sr, size, hop):
    """Summed power over channels on the program's grid: (bins, frames)."""
    return sum(np.abs(stft(ch, size, hop)) ** 2 for ch in np.atleast_2d(a))


def _db(x):
    return 10 * np.log10(max(float(x), 1e-30))


def hum_excess(a, sr, f0, harmonics, search=1.0):
    """Tonal power at f0 and its harmonics above the neighbouring bins (Welch PSD), and that floor's power.
    Returns (f0 refined, excess power, floor power over the same bins)."""
    x = np.atleast_2d(a)
    if x.shape[1] < 4096:  # too short to resolve mains hum lines (under 0.1 s): nothing measurable
        return f0, 0.0, 0.0
    nper = int(min(32768, 2 ** np.floor(np.log2(x.shape[1]))))
    freqs, P = signal.welch(x, sr, nperseg=nper, axis=-1)
    P = P.sum(axis=0)
    df = freqs[1] - freqs[0]
    if search > 0:
        cands = np.arange(f0 - search, f0 + search + 1e-9, 0.01)
        scores = [np.interp(c * np.arange(1, harmonics + 1), freqs, P).sum() for c in cands]
        f0 = float(cands[int(np.argmax(scores))])
    exc = flo = 0.0
    for k in range(1, harmonics + 1):
        fk = k * f0
        if fk + 30 + 6 * df > freqs[-1]:
            break
        d = np.abs(freqs - fk)
        peak = d <= 2 * df
        nb = (d > 3 * df) & (d <= 3 * df + max(20.0, 6 * df))
        floor = np.median(P[nb])
        exc += max(0.0, P[peak].sum() - floor * peak.sum())
        flo += floor * peak.sum()
    return f0, exc, flo


def core(M, overlap):
    """Where a region check measures: the fully selected cells, one analysis window in from every edge (2 bins in
    frequency, `overlap` frames in time), so leakage across the edge of an op's region is not counted. Falls back to
    the fully selected cells, then to any selected cell, when the selection is too small to have an inside."""
    full = M >= 0.999
    c = ndimage.binary_erosion(full, structure=np.ones((5, 2 * overlap + 1)), border_value=1)
    if c.any():
        return c.astype(float)
    return full.astype(float) if full.any() else (M > 0).astype(float)


def _kurtosis(v):
    """Kurtosis (not excess) of power values: musical noise (isolated surviving peaks) raises it (Uemura et al.)."""
    v = np.asarray(v, dtype=np.float64)
    if v.size < 8:
        return 0.0
    m, s2 = v.mean(), v.var()
    return float(((v - m) ** 4).mean() / s2 ** 2) if s2 > 0 else 0.0


def _per_bin(P, M):
    """Power in the selected cells, each bin divided by its own mean over the selected frames, so the measure sees
    the noise's texture over time, not its colour (a tilt in the spectrum is not musical noise)."""
    rows = []
    for f in np.flatnonzero(M.any(axis=1)):
        v = P[f, M[f]]
        if v.mean() > 0:
            rows.append(v / v.mean())
    return np.concatenate(rows) if rows else np.zeros(0)


def run_checks(prog, before, outputs, sr):
    results = []
    size, hop = prog.size, prog.hop
    cache = {}

    def power(key, a):
        if key not in cache:
            cache[key] = _spec_power(a, sr, size, hop)
        return cache[key]

    for c in prog.checks:
        after = outputs[c["layer"]]
        p, kind, r = c["params"], c["check"], {"name": c["name"], "check": c["check"], "layer": c["layer"]}
        if kind in ("drop", "keep", "untouched", "change"):
            Pb = power("before", before)
            freqs, times = bin_freqs(size, sr), frame_times(Pb.shape[1], size, hop, sr)
            M = core(sel_mask(prog.sels[p["at"]], freqs, times), prog.overlap)
            if kind == "untouched":
                d = np.atleast_2d(after) - np.atleast_2d(before)
                v = _db((_spec_power(d, sr, size, hop) * M).sum() / max((Pb * M).sum(), 1e-30))
                r.update({"pass": v <= p["max_db"], "value": round(v, 2),
                          "text": f"change inside {p['at']} is {v:.1f} dB relative to the original "
                                  f"(allowed <= {p['max_db']:g} dB)"})
            else:
                Pa = power("after:" + c["layer"], after)
                if kind == "change":
                    v = _db((Pa * M).sum()) - _db((Pb * M).sum())
                    r.update({"pass": abs(v - p["target_db"]) <= p["tol_db"], "value": round(v, 2),
                              "text": f"{p['at']} changed by {v:+.1f} dB (asked {p['target_db']:+g} +- {p['tol_db']:g})"})
                elif kind == "drop":
                    v = _db((Pb * M).sum()) - _db((Pa * M).sum())
                    r.update({"pass": v >= p["min_db"], "value": round(v, 2),
                              "text": f"{p['at']} went down {v:.1f} dB (needed >= {p['min_db']:g} dB)"})
                else:
                    worst, band = 0.0, None
                    eb_all = [(lo, (Pb * M)[(freqs >= lo) & (freqs < 2 * lo)].sum(),
                               (Pa * M)[(freqs >= lo) & (freqs < 2 * lo)].sum())
                              for lo in 31.25 * 2.0 ** np.arange(0, 10)]
                    eb_all.append((16000.0, (Pb * M)[freqs >= 16000].sum(), (Pa * M)[freqs >= 16000].sum()))
                    top = max(e for _, e, _ in eb_all)
                    for lo, eb, ea in eb_all:
                        if top > 0 and eb >= top * 1e-6:
                            dlt = _db(ea) - _db(eb)
                            if abs(dlt) > abs(worst):
                                worst, band = dlt, lo
                    where = f" (worst: the octave from {band:g} Hz)" if band else ""
                    r.update({"pass": abs(worst) <= p["tol_db"], "value": round(worst, 2),
                              "text": f"{p['at']} changed by {worst:+.2f} dB{where}, allowed +-{p['tol_db']:g} dB"})
        elif kind == "hum_drop" and np.atleast_2d(before).shape[1] < 4096:
            r.update({"pass": False, "value": 0.0,
                      "text": f"hum not measurable: the audio has {np.atleast_2d(before).shape[1]} samples, fewer than "
                              f"the 4096 needed to resolve hum lines (needed a {p['min_db']:g} dB drop)"})
        elif kind == "hum_drop":
            f0, eb, _ = hum_excess(before, sr, p["f0_hz"], p["harmonics"])
            _, ea, fa = hum_excess(after, sr, f0, p["harmonics"], search=0)
            v = min(120.0, _db(eb) - _db(ea)) if eb > 0 else 0.0
            gone = ea < 0.1 * fa
            r.update({"pass": v >= p["min_db"] or gone, "value": round(v, 2),
                      "text": f"hum at {f0:.2f} Hz x{p['harmonics']} went down {v:.1f} dB"
                              + (" and is no longer above the neighbouring bins" if gone else "")
                              + f" (needed >= {p['min_db']:g} dB)"})
        elif kind == "clicks":
            nb = sum(len(detect_clicks(ch, sr, p["sensitivity"])) for ch in np.atleast_2d(before))
            na = sum(len(detect_clicks(ch, sr, p["sensitivity"])) for ch in np.atleast_2d(after))
            r.update({"pass": na <= p["max"], "value": na,
                      "text": f"{na} clicks left (were {nb}; allowed <= {p['max']})"})
        elif kind == "musical_noise":
            Pb = power("before", before)
            Pa = power("after:" + c["layer"], after)
            freqs, times = bin_freqs(size, sr), frame_times(Pb.shape[1], size, hop, sr)
            M = core(sel_mask(prog.sels[p["at"]], freqs, times), prog.overlap) > 0
            M[freqs < 200] = False  # the lowest bins are few and not where musical noise is heard
            kb, ka = _kurtosis(_per_bin(Pb, M)), _kurtosis(_per_bin(Pa, M))
            v = ka / kb if kb > 0 else 1.0
            r.update({"pass": v <= p["max_ratio"], "value": round(v, 2),
                      "text": f"musical noise in {p['at']}: kurtosis ratio {v:.2f} (allowed <= {p['max_ratio']:g}; "
                              "1 = the noise kept its texture)"})
        elif kind == "peak":
            v = _db(np.max(np.abs(after)) ** 2)
            r.update({"pass": v <= p["max_dbfs"], "value": round(v, 2),
                      "text": f"peak {v:.2f} dBFS (allowed <= {p['max_dbfs']:g})"})
        results.append(r)
    return results


def damage(before, after):
    """Problems that make a version unfit to end the loop, whatever the checks say."""
    out = []
    a, b = np.atleast_2d(after), np.atleast_2d(before)
    if not np.all(np.isfinite(a)):
        return ["output has non-finite samples"]
    pk = np.max(np.abs(a))
    if pk > 1.0:
        out.append(f"clipping: peak {_db(pk ** 2):+.2f} dBFS")
    rb, ra = np.sqrt(np.mean(b ** 2)), np.sqrt(np.mean(a ** 2))
    if ra > rb * 2:
        out.append(f"output is {_db(ra ** 2) - _db(rb ** 2):.1f} dB louder than the input")
    if rb > 0 and ra < rb * 10 ** (-40 / 20):  # any input that has sound, however quiet
        out.append(f"output is {_db(rb ** 2) - _db(ra ** 2):.1f} dB quieter than the input (nearly silent)")
    return out
