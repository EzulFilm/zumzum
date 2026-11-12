"""The checks and the damage gate of a program, read from files in blocks: for runs of the block path (stream.py), so
that a whole run, checks included, uses little memory.

Supported checks: loudness, true_peak, loudness_range, rms, hum_drop, mains_hum_drop, on the program's one output
layer, with the input and output of the same length. Damage gate: the same rules as measure.damage for the programs
the fast path runs (non-finite samples, clipping, louder than the loudness target or than the input, nearly silent).
Anything else raises Unsupported. The results equal the whole-file ones (tests/test_stream_checks.py): the same pass /
fail, values within 0.02.

Each file is read once, in blocks: sums of squares, the peak, BS.1770 loudness blocks and true peak (stream.Meter), and
a Welch power spectrum built segment by segment exactly as scipy.signal.welch builds it.
"""

import numpy as np
import soundfile as sf
from scipy import signal

from . import loudness
from .engine import dehum_lines, op_dehum, refine_f0, sel_mask
from .measure import BUDGET_DB, BUDGET_MIN_SHARE, DEHUM_BUDGET_DB, LOW_CUT_TOP_HZ, STEADY_S, _db, _hum_text, hum_excess_psd, hum_nperseg
from .stft import bin_freqs, window
from .stream import Meter, Unsupported, _chunks, plan

KINDS = ("loudness", "true_peak", "loudness_range", "rms", "hum_drop", "mains_hum_drop")
BLOCK = 1 << 16


class Welch:
    """scipy.signal.welch(x, sr, nperseg=nper) (Hann, half overlap, constant detrend, density, one-sided, mean),
    summed over channels, from blocks."""

    def __init__(self, sr, nch, nper):
        self.sr, self.nper, self.step = sr, nper, nper - nper // 2
        self.win = signal.get_window("hann", nper)
        self.scale = 1.0 / (sr * (self.win ** 2).sum())
        self.buf, self.at, self.next = np.zeros((nch, 0)), 0, 0
        self.P, self.count = np.zeros(nper // 2 + 1), 0

    def feed(self, x):
        self.buf = np.concatenate([self.buf, x], axis=1)
        end = self.at + self.buf.shape[1]
        starts = np.arange(self.next, end - self.nper + 1, self.step)
        if len(starts):
            idx = (starts - self.at)[:, None] + np.arange(self.nper)[None, :]
            for ch in self.buf:
                seg = ch[idx]
                seg = (seg - seg.mean(axis=1, keepdims=True)) * self.win
                p = np.abs(np.fft.rfft(seg, axis=1)) ** 2 * self.scale
                p[:, 1:] *= 2
                if self.nper % 2 == 0:
                    p[:, -1] /= 2
                self.P += p.sum(axis=0)
            self.count += len(starts)
            self.next = int(starts[-1]) + self.step
        keep = self.next - self.at
        if keep > 0:
            self.buf, self.at = self.buf[:, keep:], self.at + keep

    def result(self):
        freqs = np.fft.rfftfreq(self.nper, 1 / self.sr)
        return freqs, self.P / max(self.count, 1)


class Stats:
    """Everything the supported checks and the gate need from one file."""

    def __init__(self, path, need_hum):
        self.path = path
        with sf.SoundFile(str(path)) as f:
            self.sr, self.nch, self.n = f.samplerate, f.channels, f.frames
            self.finite, self.peak = True, 0.0
            self.ss = np.zeros(self.nch)
            meter = Meter(self.sr, self.nch)
            welch = Welch(self.sr, self.nch, hum_nperseg(self.n)) if need_hum and self.n >= 4096 else None
            for block in f.blocks(blocksize=BLOCK, dtype="float64", always_2d=True):
                x = block.T
                if not np.all(np.isfinite(x)):
                    self.finite = False
                    x = np.nan_to_num(x)
                self.peak = max(self.peak, float(np.abs(x).max()) if x.size else 0.0)
                self.ss += (x ** 2).sum(axis=1)
                meter.feed(x)
                if welch:
                    welch.feed(x)
        self.lufs, self.tp = meter.result()
        self.lra = _lra(meter)
        self.rms_db = float(max(_db(v / max(self.n, 1)) for v in self.ss))
        self.rms_all = float(np.sqrt(self.ss.sum() / max(self.n * self.nch, 1)))
        self.psd = welch.result() if welch else None


def _lra(meter):
    """loudness.loudness_range from the meter's 100 ms step sums (3 s blocks every 0.1 s)."""
    if not meter.steps:
        return None
    s = np.concatenate(meter.steps, axis=1)
    k = 30  # 3 s of 100 ms steps
    if s.shape[1] < k:
        return None
    c = np.concatenate([np.zeros((s.shape[0], 1)), np.cumsum(s, axis=1)], axis=1)
    ms = (c[:, k:] - c[:, :-k]) / (k * meter.h)
    z = loudness._weights(meter.nch) @ ms
    with np.errstate(divide="ignore"):
        st = -0.691 + 10 * np.log10(z)
    st = st[st > loudness.ABS_GATE]
    if len(st) < 2:
        return None
    rel = 10 * np.log10(np.mean(10 ** (st / 10))) - 20
    st = st[st > rel]
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def _hum(st, f0, harmonics, search):
    if st.psd is None:
        return f0, 0.0, 0.0
    return hum_excess_psd(*st.psd, f0, harmonics, search)


def file_checks(prog, src, dst):
    """-> (check results, damage list), as measure.run_checks and measure.damage give them for the whole arrays."""
    plan(prog)  # only programs the fast path runs (the gate's rules for other ops are not here)
    out_layer = prog.layers[0][0]
    for c in prog.checks:
        if c["check"] not in KINDS:
            raise Unsupported(f"check {c['check']} is not block-wise")
        if c["layer"] != out_layer:
            raise Unsupported("a check on a layer other than the output")
    need_hum = any(c["check"] in ("hum_drop", "mains_hum_drop") for c in prog.checks)
    b, a = Stats(src, need_hum), Stats(dst, need_hum)
    if (a.n, a.nch, a.sr) != (b.n, b.nch, b.sr):
        raise Unsupported("input and output differ in length, channels or rate")
    results = []
    for c in prog.checks:
        p, kind, r = c["params"], c["check"], {"name": c["name"], "check": c["check"], "layer": c["layer"]}
        if kind in ("hum_drop", "mains_hum_drop") and b.n < 4096:
            r.update({"pass": False, "value": 0.0, "text": "hum not measurable: fewer than 4096 samples"})
        elif kind == "mains_hum_drop":
            cands = []
            for m in (50.0, 60.0):
                f0, eb, fb = _hum(b, m, p["harmonics"], 1.0)
                cands.append((eb / max(fb, 1e-30), f0, eb))
            _, f0, eb = max(cands)
            _, ea, fa = _hum(a, f0, p["harmonics"], 0)
            v = min(120.0, _db(eb) - _db(ea)) if eb > 0 else 0.0
            gone = ea < 0.1 * fa
            r.update({"pass": v >= p["min_db"] or gone, "value": round(v, 2), "f0_hz": round(f0, 2), "text": _hum_text(
                f"mains hum (the input's stronger series: {f0:.2f} Hz) x{p['harmonics']}", v, p["min_db"], gone)})
        elif kind == "hum_drop":
            f0, eb, _ = _hum(b, p["f0_hz"], p["harmonics"], 1.0)
            _, ea, fa = _hum(a, f0, p["harmonics"], 0)
            v = min(120.0, _db(eb) - _db(ea)) if eb > 0 else 0.0
            gone = ea < 0.1 * fa
            r.update({"pass": v >= p["min_db"] or gone, "value": round(v, 2), "text": _hum_text(
                f"hum at {f0:.2f} Hz x{p['harmonics']}", v, p["min_db"], gone)})
        elif kind == "loudness":
            v = a.lufs
            if v is None:
                r.update({"pass": False, "value": None, "valid": False,
                          "text": "no loudness to measure (silent, or under 0.4 s)"})
            else:
                r.update({"pass": abs(v - p["target_lufs"]) <= p["tol_lu"], "value": round(v, 2),
                          "text": f"integrated loudness {v:.1f} LUFS (wanted {p['target_lufs']:g} +- "
                                  f"{p['tol_lu']:g} LU; was {b.lufs or float('-inf'):.1f})"})
        elif kind == "loudness_range":
            v, v0 = a.lra, b.lra
            if v is None:
                r.update({"pass": False, "value": None, "valid": False,
                          "text": "no loudness range to measure (silent, or under 3 s)"})
            else:
                r.update({"pass": v <= p["max_lu"], "value": round(v, 2),
                          "text": f"loudness range {v:.1f} LU (allowed <= {p['max_lu']:g}"
                                  + (f"; was {v0:.1f})" if v0 is not None else ")")})
        elif kind == "rms":
            v, v0 = a.rms_db, b.rms_db
            r.update({"pass": p["min_dbfs"] <= v <= p["max_dbfs"], "value": round(v, 2),
                      "text": f"RMS {v:.1f} dBFS (allowed {p['min_dbfs']:g}..{p['max_dbfs']:g}; was {v0:.1f})"})
        elif kind == "true_peak":
            v = a.tp
            r.update({"pass": v <= p["max_dbtp"], "value": round(v, 2) if np.isfinite(v) else None,
                      "text": f"true peak {v:.2f} dBTP (allowed <= {p['max_dbtp']:g})"})
        results.append(r)
    return results, _damage(prog, b, a)


def _steady_rows(rows, hop, sr):
    """measure._steady for the rows kept (bins x frames)."""
    k = max(1, int(round(STEADY_S * sr / hop)))
    if rows.shape[1] <= k:
        return rows.mean(axis=1)
    c = np.cumsum(np.concatenate([np.zeros((rows.shape[0], 1)), rows], axis=1), axis=1)
    return np.percentile((c[:, k:] - c[:, :-k]) / k, 10, axis=1)


def budget(prog, src):
    """measure.budget_problems, reading the input in blocks: for each dehum node and low cut, the power it removes
    and, for only the bins it touches, the time series that the steady level needs."""
    return [r["problem"] for r in budget_numbers(prog, src) if r["problem"]]


def budget_numbers(prog, src):
    """Per dehum node and low cut of the program's layer: {"node", "removed", "steady", "total", "problem"}."""
    out = []
    with sf.SoundFile(str(src)) as f:
        sr, nch, n = f.samplerate, f.channels, f.frames
        for op in [prog.ops[nm] for nm in prog.layers[0][1] if not prog.ops[nm]["off"]]:
            p = op["params"]
            size = p.get("fft") or prog.size
            hop, w, freqs = size // prog.overlap, window(size), bin_freqs(size, sr)
            is_dehum = op["op"] == "dehum" and op["at"] is None
            is_cut = (op["op"] == "gain" and p["db"] < 0 and op["at"] is not None and op["at"][0] not in ("not", "and", "or")
                      and op["at"][1] == "band" and not op["at"][2].get("t_s") and op["at"][2]["f_hz"][1] <= LOW_CUT_TOP_HZ)
            if not (is_dehum or is_cut):
                continue
            total, removed, steady = 0.0, 0.0, 0.0
            if is_dehum:  # pass 1: the mean spectrum per channel, for f0 (as the op refines it on the whole input)
                P = np.zeros((nch, len(freqs)))
                frames = 0
                for _, fr in _chunks(f, n, size, hop):
                    for c in range(nch):
                        X = np.fft.rfft(fr[c] * w, axis=1)
                        P[c] += (np.abs(X) ** 2).sum(axis=0)
                    frames += fr.shape[1]
                f0s = [refine_f0(P[c] / max(frames, 1), freqs, p["f0_hz"], p["harmonics"], p["search_hz"]) for c in range(nch)]
                lines = [dehum_lines(p, f0s[c], freqs) for c in range(nch)]
                keep = [np.unique(np.concatenate([np.concatenate([a_, b_]) for a_, b_ in lines[c]]) if lines[c] else
                                  np.zeros(0, int)) for c in range(nch)]
                q = [dict(p, f0_hz=f0s[c], search_hz=0) for c in range(nch)]
            else:
                M = sel_mask(op["at"], freqs, np.zeros(1))[:, 0] * op["mix"] * (1 - 10 ** (p["db"] / 10))
                keep = [np.flatnonzero(M > 0)] * nch
            series = [[] for _ in range(nch)]
            for _, fr in _chunks(f, n, size, hop):  # pass 2: what the node removes, and the kept bins' power
                for c in range(nch):
                    X = np.fft.rfft(fr[c] * w, axis=1).T
                    PX = np.abs(X) ** 2
                    total += float(PX.sum())
                    if is_dehum:
                        Y = op_dehum(X, q[c], {"freqs": freqs, "info": {}})
                        removed += float((PX - np.abs(Y) ** 2).sum())
                    else:
                        removed += float((PX.sum(axis=1) * M).sum())
                    series[c].append(PX[keep[c]])
            for c in range(nch):
                rows = np.concatenate(series[c], axis=1) if series[c] else np.zeros((len(keep[c]), 0))
                st = np.zeros(len(freqs))
                if rows.size:
                    st[keep[c]] = _steady_rows(rows, hop, sr)
                frames = rows.shape[1]
                if is_dehum:
                    for cut, nb in lines[c]:
                        steady += float(np.maximum(0.0, st[cut] - np.median(st[nb])).sum()) * frames
                else:
                    steady += float((st * M).sum()) * frames
            what = f"dehum at {p['f0_hz']:g} Hz" if is_dehum else f"the low cut under {op['at'][2]['f_hz'][1]:g} Hz"
            problem = None
            margin = DEHUM_BUDGET_DB if is_dehum else BUDGET_DB
            if total > 0 and removed >= BUDGET_MIN_SHARE * total and removed > steady * 10 ** (margin / 10):
                problem = (f"{what} removed {_db(removed) - _db(max(steady, 1e-30)):.1f} dB more than the steady "
                           f"{'hum' if is_dehum else 'low noise'} it targets: it took programme")
            out.append({"node": op["name"] if "name" in op else what, "what": what, "removed": removed, "steady": steady,
                        "total": total, "problem": problem})
    return out


def _damage(prog, b, a):
    """measure.damage's rules for the fast path's programs, from the two files' numbers."""
    if not a.finite:
        return ["output has non-finite samples"]
    out = []
    if a.peak > 1.0:
        out.append(f"clipping: peak {_db(a.peak ** 2):+.2f} dBFS")
    rb, ra = b.rms_all, a.rms_all
    used = [prog.ops[n] for n in prog.layers[0][1] if not prog.ops[n]["off"]]
    level = [o for o in used if o["op"] in ("loudness", "normalize")]
    targets = [o["params"]["target_lufs"] for o in level if o["op"] == "loudness"]
    if targets:
        if a.lufs is not None and a.lufs > max(targets) + 1:
            out.append(f"output is {a.lufs:.1f} LUFS, louder than the program's loudness target {max(targets):g}")
    elif ra > rb * 2 and not level:
        out.append(f"output is {_db(ra ** 2) - _db(rb ** 2):.1f} dB louder than the input")
    out += budget(prog, b.path)
    if rb > 0 and ra < rb * 10 ** (-40 / 20):
        out.append(f"output is {_db(rb ** 2) - _db(ra ** 2):.1f} dB quieter than the input (nearly silent)")
    return out
