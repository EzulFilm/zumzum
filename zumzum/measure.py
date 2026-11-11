"""Measured checks on a before / after pair (docs/spec.md section 6) and the damage gate.

Every check returns {"name", "check", "pass", "value", "text"}: `value` is the number measured, `text` says it in one
line a person (or a model) can act on.
"""

import numpy as np
from scipy import ndimage, signal

from . import loudness
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
    freqs, P = signal.welch(x, sr, nperseg=hum_nperseg(x.shape[1]), axis=-1)
    return hum_excess_psd(freqs, P.sum(axis=0), f0, harmonics, search)


def hum_nperseg(n):
    """Welch segment length for the hum checks on a signal of n samples (n >= 4096)."""
    return int(min(32768, 2 ** np.floor(np.log2(n))))


def hum_excess_psd(freqs, P, f0, harmonics, search=1.0):
    """hum_excess from a power spectrum already summed over channels (the block-wise checks build it in blocks)."""
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


def attenuated_excess(before, after, size, hop, floor_db=-40.0):
    """(largest dB by which a cell of after's STFT exceeds the same cell of before's, among cells within floor_db of
    the loudest; the share of those cells over +0.1 dB; how many were compared; the energy by which all cells exceed
    the input, dB of after's energy)."""
    from .stft import stft
    b, a = np.atleast_2d(before), np.atleast_2d(after)
    worst, over, n, ex, tot = None, 0, 0, 0.0, 0.0
    for x, y in zip(b, a):
        X, Y = np.abs(stft(x, size, hop)) ** 2, np.abs(stft(y[:len(x)], size, hop)) ** 2
        ex += float(np.maximum(Y - X, 0).sum())
        tot += float(Y.sum())
        top = X.max()
        if top <= 0:
            continue
        use = X > top * 10 ** (floor_db / 10)
        r = 10 * np.log10(np.maximum(Y[use], 1e-300) / X[use])
        if r.size:
            worst = float(r.max()) if worst is None else max(worst, float(r.max()))
            over += int((r > 0.1).sum())
            n += int(r.size)
    return worst, (over / n if n else 0.0), n, _db(ex / max(tot, 1e-30))


def rms_dbfs(x):
    """RMS of every sample of the louder channel, dBFS (a full-scale sine reads -3.0)."""
    x = np.atleast_2d(x)
    return float(max(_db(np.mean(c ** 2)) for c in x))


def noise_floor_dbfs(x, sr, window_ms=500.0):
    """The quietest window_ms span (100 ms steps) that is not digital silence: RMS of the louder channel, dBFS.
    None when every span is digitally silent or the layer is shorter than one span."""
    x = np.atleast_2d(x)
    w, step = int(window_ms / 1000 * sr), max(1, int(0.1 * sr))
    if w < 1 or x.shape[1] < w:
        return None
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x ** 2, axis=1)], axis=1)
    starts = np.arange(0, x.shape[1] - w + 1, step)
    ms = ((c[:, starts + w] - c[:, starts]) / w).max(axis=0)
    ms = ms[ms > 10 ** (-120 / 10)]  # digital silence (a pad, a fade to zero) is not a noise floor
    return float(_db(ms.min())) if ms.size else None


def _hum_text(what, v, min_db, gone):
    """hum_drop passes in either of two ways; the text says which, so a pass never reads as under its threshold."""
    if v >= min_db:
        return f"{what} went down {v:.1f} dB (needed >= {min_db:g} dB)"
    if gone:
        return (f"{what} is no longer above the neighbouring bins, which passes; it went down {v:.1f} dB "
                f"(the other way to pass: >= {min_db:g} dB)")
    return f"{what} went down {v:.1f} dB and still stands above the neighbouring bins (needed >= {min_db:g} dB, or gone)"


def filled_regions(Pa, M, times, freqs, context_s):
    """For each connected region of mask M (> 0.5): its energy after vs what its surroundings predict (the mean power
    per bin, frame by frame, in the context_s before and after it, same frequency rows, outside every region, times
    the region's cells per row). below_db > 0: the region is that far under its surroundings (None: no surroundings)."""
    from scipy import ndimage
    inside = M > 0.5
    lab, n = ndimage.label(inside, structure=np.ones((3, 3)))
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    k = max(1, int(round(context_s / dt)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        fs, ts = sl
        reg = lab == i
        cells = reg[fs].sum(axis=1)  # cells per frequency row
        lo, hi = max(0, ts.start - k), min(Pa.shape[1], ts.stop + k)
        around = np.zeros(Pa.shape[1], bool)
        around[lo:ts.start] = around[ts.stop:hi] = True
        ok = ~inside[fs][:, around]
        nb = Pa[fs][:, around]
        have = ok.sum(axis=1)
        row = {"t_s": [round(float(times[ts.start]), 2), round(float(times[ts.stop - 1]), 2)],
               "f_hz": [round(float(freqs[fs.start]), 1), round(float(freqs[fs.stop - 1]), 1)]}
        if (have > 0).sum() == 0:
            row["below_db"] = None
        else:
            mean = np.where(have > 0, (nb * ok).sum(axis=1) / np.maximum(have, 1), 0.0)
            want = float((mean * cells)[have > 0].sum())
            got = float((Pa * reg)[fs][have > 0].sum())
            row["below_db"] = round(_db(want) - _db(got), 2)
        out.append(row)
    return out


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
            r.update({"pass": v >= p["min_db"] or gone, "value": round(v, 2), "text": _hum_text(
                f"hum at {f0:.2f} Hz x{p['harmonics']}", v, p["min_db"], gone)})
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
        elif kind == "loudness":
            v = loudness.integrated(after, sr)
            if v is None:
                r.update({"pass": False, "value": None, "valid": False,
                          "text": "no loudness to measure (silent, or under 0.4 s)"})
            else:
                r.update({"pass": abs(v - p["target_lufs"]) <= p["tol_lu"], "value": round(v, 2),
                          "text": f"integrated loudness {v:.1f} LUFS (wanted {p['target_lufs']:g} +- "
                                  f"{p['tol_lu']:g} LU; was {loudness.integrated(before, sr) or float('-inf'):.1f})"})
        elif kind == "loudness_range":
            v, v0 = loudness.loudness_range(after, sr), loudness.loudness_range(before, sr)
            if v is None:
                r.update({"pass": False, "value": None, "valid": False,
                          "text": "no loudness range to measure (silent, or under 3 s)"})
            else:
                r.update({"pass": v <= p["max_lu"], "value": round(v, 2),
                          "text": f"loudness range {v:.1f} LU (allowed <= {p['max_lu']:g}"
                                  + (f"; was {v0:.1f})" if v0 is not None else ")")})
        elif kind == "rms":
            v, v0 = rms_dbfs(after), rms_dbfs(before)
            r.update({"pass": p["min_dbfs"] <= v <= p["max_dbfs"], "value": round(v, 2),
                      "text": f"RMS {v:.1f} dBFS (allowed {p['min_dbfs']:g}..{p['max_dbfs']:g}; was {v0:.1f})"})
        elif kind == "noise_floor":
            v, v0 = noise_floor_dbfs(after, sr, p["window_ms"]), noise_floor_dbfs(before, sr, p["window_ms"])
            if v is None:
                r.update({"pass": False, "value": None, "valid": False,
                          "text": f"no noise floor to measure (only digital silence, or under {p['window_ms']:g} ms)"})
            else:
                r.update({"pass": v <= p["max_dbfs"], "value": round(v, 2),
                          "text": f"noise floor {v:.1f} dBFS RMS, quietest {p['window_ms']:g} ms (allowed <= "
                                  f"{p['max_dbfs']:g}" + (f"; was {v0:.1f})" if v0 is not None else ")")})
        elif kind == "true_peak":
            v = loudness.true_peak(after, sr)
            r.update({"pass": v <= p["max_dbtp"], "value": round(v, 2) if np.isfinite(v) else None,
                      "text": f"true peak {v:.2f} dBTP (allowed <= {p['max_dbtp']:g})"})
        elif kind == "attenuated":
            v, share, n_cells, ex = attenuated_excess(before, after, size, hop, p["floor_db"])
            if v is None:
                r.update({"pass": False, "value": None, "valid": False, "text": "no cells to compare (silent input)"})
            else:
                r.update({"pass": v <= p["max_over_db"] and ex <= p["max_excess_db"], "value": round(v, 3),
                          "excess_db": round(ex, 2),
                          "text": f"largest excess of a cell over the input {v:+.2f} dB (cells within "
                                  f"{-p['floor_db']:g} dB of the loudest; allowed <= {p['max_over_db']:g}), "
                                  f"{100 * share:.3f} % of {n_cells} cells over +0.1 dB; total excess {ex:.1f} dB of "
                                  f"the layer (allowed <= {p['max_excess_db']:g})"})
        elif kind == "filled":
            Pb, Pa = power("before", before), power("after:" + c["layer"], after)
            freqs, times = bin_freqs(size, sr), frame_times(Pb.shape[1], size, hop, sr)
            M = core(sel_mask(prog.sels[p["at"]], freqs, times), prog.overlap)
            rows = filled_regions(Pa, M, times, freqs, p["context_ms"] / 1000)
            meas = [x for x in rows if x["below_db"] is not None]
            if not meas:
                r.update({"pass": False, "value": None, "valid": False,
                          "text": f"{p['at']}: no region with surroundings to compare against"})
            else:
                lo, hi = p["max_below_db"], p["max_above_db"]
                bad = [x for x in meas if x["below_db"] > lo or -x["below_db"] > hi]
                worst = max(meas, key=lambda x: max(x["below_db"] - lo, -x["below_db"] - hi))
                where = lambda x: (f"{x['t_s'][0]:.2f}..{x['t_s'][1]:.2f} s, {x['f_hz'][0]:.0f}..{x['f_hz'][1]:.0f} Hz "  # noqa: E731
                                   f"{-x['below_db']:+.1f} dB")
                limits = f"{lo:g} dB under" + (f" or {hi:g} dB over" if hi < 200 else "")
                r.update({"pass": not bad, "value": round(worst["below_db"], 2), "regions": rows,
                          "text": f"{p['at']}: {len(meas)} regions, {len(bad)} more than {limits} their surroundings"
                                  + (": " + "; ".join(where(x) for x in bad[:8]) if bad else
                                     f"; worst {where(worst)}")})
        elif kind == "mains_hum_drop" and np.atleast_2d(before).shape[1] < 4096:
            r.update({"pass": False, "value": 0.0, "text": "hum not measurable: fewer than 4096 samples"})
        elif kind == "mains_hum_drop":
            cands = []
            for m in (50.0, 60.0):
                f0, eb, fb = hum_excess(before, sr, m, p["harmonics"])
                cands.append((eb / max(fb, 1e-30), f0, eb))
            _, f0, eb = max(cands)  # the series standing higher above its neighbours in the input
            _, ea, fa = hum_excess(after, sr, f0, p["harmonics"], search=0)
            v = min(120.0, _db(eb) - _db(ea)) if eb > 0 else 0.0
            gone = ea < 0.1 * fa
            r.update({"pass": v >= p["min_db"] or gone, "value": round(v, 2), "f0_hz": round(f0, 2), "text": _hum_text(
                f"mains hum (the input's stronger series: {f0:.2f} Hz) x{p['harmonics']}", v, p["min_db"], gone)})
        results.append(r)
    return results


FLAT_LEVEL = 0.99999       # a sample counts as flat at the peak within 0.001 % of it
CLIP_MIN_RUNS = 10         # this many flat tops or more, and
CLIP_SURROUND_DB = -12.0   # sound around them within 12 dB of the peak: the programme is clipped
DECLICK_ON_CLIP_PCT = 50.0  # this share or more of the samples declick rewrote lie within 1 ms of a flat top


def _flat_runs(mask):
    """(start, end) of each run of 3 or more True values."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= 3]


def programme_clipping(x, sr):
    """-> (is the programme clipped, mask of the flat-top samples (channels x samples), text). Flat tops are runs of
    3 or more samples at the peak. A clipped programme has many of them with loud sound around them (the 5 ms each
    side, leaving out 1 ms); a few clicks that hit the ceiling stand alone in quieter sound and do not count."""
    a = np.atleast_2d(np.asarray(x, float))
    pk = float(np.max(np.abs(a))) if a.size else 0.0
    mask = np.zeros(a.shape, bool)
    if pk <= 0:
        return False, mask, "silent"
    w, g, lv, n_runs = int(0.005 * sr), int(0.001 * sr), [], 0
    for ci, ch in enumerate(a):
        for s0, e0 in _flat_runs(np.abs(ch) >= FLAT_LEVEL * pk):
            mask[ci, s0:e0 + 1] = True
            n_runs += 1
            seg = np.concatenate([ch[max(0, s0 - w):max(0, s0 - g)], ch[e0 + g:e0 + w]])
            if seg.size:
                lv.append(10 * np.log10(max(float(np.mean(seg ** 2)), 1e-20) / pk ** 2))
    around = float(np.median(lv)) if lv else -200.0
    yes = n_runs >= CLIP_MIN_RUNS and around >= CLIP_SURROUND_DB
    return yes, mask, f"{n_runs} flat tops at the peak, the sound around them {around:.1f} dB under it"


def damage(before, after, prog=None, sr=None):
    """Problems that make a version unfit to end the loop, whatever the checks say. A program with a loudness node
    asked for the level change: then "louder" means louder than its loudness target (+1 LU), not louder than the
    input."""
    out = []
    a, b = np.atleast_2d(after), np.atleast_2d(before)
    if not np.all(np.isfinite(a)):
        return ["output has non-finite samples"]
    pk = np.max(np.abs(a))
    if pk > 1.0:
        out.append(f"clipping: peak {_db(pk ** 2):+.2f} dBFS")
    rb, ra = np.sqrt(np.mean(b ** 2)), np.sqrt(np.mean(a ** 2))
    feeds = set(prog.output_mix) if prog and prog.output == "mix" else {prog.output} if prog else set()
    used = {n for l, chain in (prog.layers if prog else ()) if l in feeds for n in chain}
    targets = [prog.ops[n]["params"]["target_lufs"] for n in used
               if prog.ops[n]["op"] == "loudness" and not prog.ops[n]["off"]]
    if targets:
        lu = loudness.integrated(a, sr) if sr else None
        if lu is not None and lu > max(targets) + 1:
            out.append(f"output is {lu:.1f} LUFS, louder than the program's loudness target {max(targets):g}")
    elif ra > rb * 2:
        out.append(f"output is {_db(ra ** 2) - _db(rb ** 2):.1f} dB louder than the input")
    declick = [n for n in used if prog.ops[n]["op"] == "declick" and not prog.ops[n]["off"]]
    if declick and sr and a.shape == b.shape:
        # a clipped programme read as clicks: declick rewrites samples at and next to the flat tops by interpolation,
        # which does not rebuild a clipped peak and can leave the file further from the original than it was.
        # Clicks that hit the ceiling on their own are not counted.
        clipped, mask, text = programme_clipping(b, sr)
        if clipped:
            changed = np.abs(a - b) > 1e-4 * float(np.max(np.abs(b)))
            k = np.ones(2 * int(0.001 * sr) + 1)  # within 1 ms of a flat top
            near = np.array([np.convolve(m.astype(float), k, "same") > 0 for m in mask])
            share = 100.0 * float(near[changed].mean()) if changed.any() else 0.0
            if share >= DECLICK_ON_CLIP_PCT:
                out.append(f"the input is clipped ({text}) and {share:.0f} % of what declick rewrote lies at those flat "
                           "tops: it is treating clipping as clicks, which does not repair it")
    if rb > 0 and ra < rb * 10 ** (-40 / 20):  # any input that has sound, however quiet
        out.append(f"output is {_db(rb ** 2) - _db(ra ** 2):.1f} dB quieter than the input (nearly silent)")
    return out
