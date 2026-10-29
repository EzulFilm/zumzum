"""The NumPy reference engine: the source of truth for what a checked program does (docs/spec.md section 5).

Audio is float64 (channels, samples) in [-1, 1] full scale; channels are processed independently. Each layer runs its
node chain on the input. An STFT op computes its result Y on the op's own grid and blends it into the current signal
inside its selection:  Z = X + mix * M * (Y - X),  then ISTFT. A time op blends per sample with the selection's time
weight. Without `at`, M = 1 everywhere.
"""

import math

import numpy as np
from scipy import linalg, ndimage, special

from .stft import bin_freqs, frame_times, istft, stft

EPS = 1e-20


# ---------------------------------------------------------------------------------------------------- selections

def _edge(x, lo, hi, feather):
    """1 inside [lo, hi], a raised-cosine fall to 0 over `feather` outside it, 0 beyond."""
    if feather <= 0:
        return ((x >= lo) & (x <= hi)).astype(float)
    d = np.maximum(lo - x, x - hi)
    w = np.where(d <= 0, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.clip(d / feather, 0, 1)))
    return np.where(d >= feather, 0.0, w)


def sel_mask(s, freqs, times):
    """Selection -> weights (bins, frames) in [0, 1]."""
    kind = s[0]
    if kind == "not":
        return 1.0 - sel_mask(s[1], freqs, times)
    if kind in ("and", "or"):
        parts = [sel_mask(a, freqs, times) for a in s[1]]
        return np.minimum.reduce(parts) if kind == "and" else np.maximum.reduce(parts)
    _, name, p = s
    ones_t, ones_f = np.ones(len(times)), np.ones(len(freqs))
    if name == "all":
        return np.ones((len(freqs), len(times)))
    tw = ones_t
    if p.get("t_s"):
        tw = _edge(times, p["t_s"][0], p["t_s"][1], p.get("feather_ms", 0) / 1000.0)
    fw = ones_f
    if name in ("band", "box"):
        fw = _edge(freqs, p["f_hz"][0], p["f_hz"][1], p.get("feather_hz", 0))
    elif name == "harmonics":
        df = freqs[1] - freqs[0]
        half = max(p["width_hz"] / 2, df / 2)
        fw = np.zeros(len(freqs))
        for k in range(1, p["count"] + 1):
            fk = k * p["f0_hz"]
            if fk > freqs[-1]:
                break
            fw = np.maximum(fw, (np.abs(freqs - fk) <= half).astype(float))
    return fw[:, None] * tw[None, :]


def time_weight(s, n, sr, size, hop):
    """A selection's time support per sample on the program's analysis grid: the largest weight over frequency of
    each frame, interpolated to samples."""
    if s is None:
        return np.ones(n)
    t = n_frames_for(n, size, hop)
    times, freqs = frame_times(t, size, hop, sr), bin_freqs(size, sr)
    m = sel_mask(s, freqs, times).max(axis=0)
    return np.interp(np.arange(n) / sr, times, m)


def n_frames_for(n, size, hop):
    return (n + size) // hop + 1


# ---------------------------------------------------------------------------------------------------- STFT ops

def op_gain(X, p, ctx):
    return X * 10 ** (p["db"] / 20)


def refine_f0(P, freqs, f0, count, search):
    """The f0 within +- search (0.01 Hz steps) whose harmonics hold the most power in the mean spectrum P."""
    if search <= 0:
        return f0
    best, best_score = f0, -1.0
    for cand in np.arange(f0 - search, f0 + search + 1e-9, 0.01):
        fk = cand * np.arange(1, count + 1)
        fk = fk[fk < freqs[-1]]
        score = np.interp(fk, freqs, P).sum()
        if score > best_score:
            best, best_score = cand, score
    return float(best)


def op_dehum(X, p, ctx):
    freqs = ctx["freqs"]
    df = freqs[1] - freqs[0]
    mag = np.abs(X)
    f0 = refine_f0((mag ** 2).mean(axis=1), freqs, p["f0_hz"], p["harmonics"], p["search_hz"])
    ctx["info"]["f0_hz"] = round(f0, 3)
    g = 10 ** (-p["depth_db"] / 20)
    half = max(p["width_hz"] / 2, 2 * df)        # at least the Hann main lobe of a stationary tone
    gap, span = half + df, half + df + max(20.0, 6 * df)
    target = mag.copy()
    for k in range(1, p["harmonics"] + 1):
        fk = k * f0
        if fk + span > freqs[-1]:
            break
        d = np.abs(freqs - fk)
        cut, nb = d <= half, (d > gap) & (d <= span)
        floor = np.median(mag[nb], axis=0)                 # per frame: the level next to the hum line
        m = mag[cut]
        target[cut] = np.maximum(m * g, np.minimum(m, floor[None, :]))
    return X * (target / np.maximum(mag, EPS))


def noise_psd(P, times, profile, freqs=None, sel=None):
    """Mean power per bin over a noise sample: a selection (`profile: <sel>`; each bin averaged over the frames where it
    is fully selected, bins the selection never covers get the mean of the frames it does select), a time span, or
    else the quietest 10 % of frames (at least 5)."""
    if sel is not None:
        M = sel_mask(sel, freqs, times) >= 0.999
        cols = M.any(axis=0)
        if cols.any():
            span = P[:, cols].mean(axis=1)
            out = np.where(M.any(axis=1), (P * M).sum(axis=1) / np.maximum(M.sum(axis=1), 1), span)
            return out
    if profile is not None:
        sel = (times >= profile[0]) & (times <= profile[1])
        if sel.sum() >= 1:
            return P[:, sel].mean(axis=1)
    k = max(5, int(0.1 * P.shape[1]))
    quiet = np.argsort(P.sum(axis=0))[:k]
    return P[:, quiet].mean(axis=1)


def track_noise(P, hop, sr, window_s, alpha=0.85, bias=1.5):
    """Noise power per bin and frame that follows a changing noise: the minimum of the time-smoothed power over a
    sliding window of window_s, times a bias for the minimum's low reading (after Martin 2001, minimum statistics;
    a fixed bias instead of his optimal smoothing)."""
    S = P.copy()
    for t in range(1, P.shape[1]):
        S[:, t] = alpha * S[:, t - 1] + (1 - alpha) * P[:, t]
    w = max(3, int(round(window_s * sr / hop)))
    return ndimage.minimum_filter1d(S, size=w, axis=1, mode="nearest") * bias


def op_denoise(X, p, ctx):
    P = np.abs(X) ** 2
    if p["track_s"] > 0:
        N = track_noise(P, ctx["hop"], ctx["sr"], p["track_s"])
    else:
        N = noise_psd(P, ctx["times"], p["profile_t_s"], ctx["freqs"], ctx["sels"].get(p["profile"]))[:, None]
    N = np.maximum(N, EPS) * 10 ** (p["noise_db"] / 10)
    floor = 10 ** (-p["reduce_db"] / 20)
    gamma = P / N
    if p["mode"] == "subtract":  # power subtraction with over-subtraction (Boll 1979; Berouti et al. 1979)
        G = np.sqrt(np.maximum(1 - p["over"] / np.maximum(gamma, EPS), 0))
        return X * _smooth_gain(np.clip(G, floor, 1), p, ctx) * _gate(P, N, p, ctx)[None, :]
    # the a-priori SNR floor sets how far noise-only bins can go down; tie it to reduce_db (never above -25 dB)
    a, xi_min = p["smooth"], 10 ** (-max(25.0, p["reduce_db"]) / 10)
    G = np.ones_like(gamma)
    prev = np.ones(gamma.shape[0])          # |A_hat|^2 / N of the previous frame
    for t in range(gamma.shape[1]):
        gm = gamma[:, t]
        xi = np.maximum(a * prev + (1 - a) * np.maximum(gm - 1, 0), xi_min)
        if p["mode"] == "wiener":
            g = xi / (1 + xi)
        elif p["mode"] == "logmmse":  # Ephraim & Malah 1985, MMSE of the log spectral amplitude
            v = np.maximum(xi / (1 + xi) * gm, 1e-10)
            g = xi / (1 + xi) * np.exp(0.5 * special.exp1(v))
        else:  # Ephraim & Malah 1984, MMSE short-time spectral amplitude
            v = xi / (1 + xi) * gm
            g = (math.sqrt(math.pi) / 2) * np.sqrt(v) / np.maximum(gm, EPS) * (
                (1 + v) * special.i0e(v / 2) + v * special.i1e(v / 2))
            g = np.where(np.isfinite(g), g, 1.0)
        g = np.clip(g, floor, 1.0)
        G[:, t] = g
        prev = g * g * gm
    return X * _smooth_gain(G, p, ctx) * _gate(P, N, p, ctx)[None, :]


def _gate(P, N, p, ctx):
    """A downward expander for noise-only moments: per frame, the posterior SNR in 300-4000 Hz says how likely sound
    other than noise is present (1 dB -> none, 6 dB -> certainly); that presence opens at once and closes over
    gate_release_ms, and where it is low every bin goes down a further gate_db."""
    if p["gate_db"] <= 0:
        return np.ones(P.shape[1])
    f = ctx["freqs"]
    band = (f >= 300) & (f <= 4000)
    Nf = np.broadcast_to(N, P.shape)  # a fixed estimate (bins, 1) or a tracked one (bins, frames)
    snr = P[band].mean(axis=0) / np.maximum(Nf[band].mean(axis=0), EPS)  # one SNR per frame
    pres = np.clip((10 * np.log10(np.maximum(snr, 1e-12)) - 1.0) / 5.0, 0, 1)
    r = math.exp(-ctx["hop"] / (ctx["sr"] * p["gate_release_ms"] / 1000))
    held = pres.copy()
    for t in range(1, len(held)):
        held[t] = max(pres[t], r * held[t - 1])
    lead = max(1, int(round(0.02 * ctx["sr"] / ctx["hop"])))  # open 20 ms early so onsets are not clipped
    held = ndimage.maximum_filter1d(held, size=2 * lead + 1, origin=0)
    return 10 ** (-p["gate_db"] * (1 - held) / 20)


def _smooth_gain(G, p, ctx):
    """Average the gain over smooth_hz in frequency: an isolated bin that survives alone (musical noise) is pulled
    down to its neighbours, while a voice harmonic, which is wide in an STFT, keeps its gain."""
    n = int(round(p["smooth_hz"] / (ctx["freqs"][1] - ctx["freqs"][0])))
    if n >= 2:
        G = ndimage.uniform_filter1d(G, size=n | 1, axis=0, mode="nearest")
    m = int(round(p["smooth_ms"] / 1000 * ctx["sr"] / ctx["hop"]))
    if m >= 2:  # and over time: a gain that flickers frame to frame is what musical noise sounds like
        G = ndimage.uniform_filter1d(G, size=m | 1, axis=1, mode="nearest")
    return G


def _inst_phase_step(X, t, hop, size):
    """Phase advance per frame of every bin, measured between frames t - 1 and t."""
    k = np.arange(X.shape[0])
    expect = 2 * np.pi * k * hop / size
    if t < 1:
        return expect
    d = np.angle(X[:, t]) - np.angle(X[:, t - 1]) - expect
    return expect + (d + np.pi) % (2 * np.pi) - np.pi


def op_heal(X, p, ctx):
    M = ctx["mask"]
    R = M > 0
    F, T = X.shape
    hop, size, sr = ctx["hop"], ctx["size"], ctx["sr"]
    logm = np.log(np.abs(X) + 1e-12)
    est_t = est_f = None
    if p["direction"] in ("time", "both"):
        c = max(1, int(round(p["context_ms"] / 1000 * sr / hop)))
        est_t = np.full((F, T), np.nan)
        ph = np.angle(X).copy()
        for f in np.flatnonzero(R.any(axis=1)):
            for t0, t1 in _runs(R[f]):
                left = [t for t in range(max(0, t0 - c), t0) if not R[f, t]]
                right = [t for t in range(t1 + 1, min(T, t1 + 1 + c)) if not R[f, t]]
                if not left and not right:
                    continue
                lv = logm[f, left].mean() if left else logm[f, right].mean()
                rv = logm[f, right].mean() if right else lv
                n = t1 - t0 + 1
                w = (np.arange(n) + 1) / (n + 1)
                est_t[f, t0:t1 + 1] = (1 - w) * lv + w * rv
                if p["phase"] == "continue" and t0 >= 2:  # continue the phase of the frame before the gap
                    step = _inst_phase_step(X, t0 - 1, hop, size)[f]
                    ph[f, t0:t1 + 1] = np.angle(X[f, t0 - 1]) + step * (np.arange(n) + 1)
        phase_t = ph
    if p["direction"] in ("freq", "both"):
        c = 3
        est_f = np.full((F, T), np.nan)
        for t in np.flatnonzero(R.any(axis=0)):
            for f0, f1 in _runs(R[:, t]):
                below = [f for f in range(max(0, f0 - c), f0) if not R[f, t]]
                above = [f for f in range(f1 + 1, min(F, f1 + 1 + c)) if not R[f, t]]
                if not below and not above:
                    continue
                lv = logm[below, t].mean() if below else logm[above, t].mean()
                hv = logm[above, t].mean() if above else lv
                n = f1 - f0 + 1
                w = (np.arange(n) + 1) / (n + 1)
                est_f[f0:f1 + 1, t] = (1 - w) * lv + w * hv
    if est_t is not None and est_f is not None:
        est = np.where(np.isnan(est_t), est_f, np.where(np.isnan(est_f), est_t, 0.5 * (est_t + est_f)))
        phase = np.where(np.isnan(est_t), np.angle(X), phase_t)
    elif est_t is not None:
        est, phase = est_t, phase_t
    else:
        est, phase = est_f, np.angle(X)
    Y = X.copy()
    ok = ~np.isnan(est)
    Y[ok] = np.exp(est[ok]) * np.exp(1j * phase[ok])
    return Y


def _runs(b):
    """Contiguous True runs of a 1-D bool array as (start, end) inclusive."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], b.astype(np.int8), [0]])))
    return list(zip(idx[::2], idx[1::2] - 1))


def op_split_hp(X, p, ctx):
    S = np.abs(X)
    lt = max(1, int(round(p["time_ms"] / 1000 * ctx["sr"] / ctx["hop"]))) | 1
    lf = max(1, int(round(p["freq_hz"] / (ctx["freqs"][1] - ctx["freqs"][0])))) | 1
    H = ndimage.median_filter(S, size=(1, lt), mode="reflect")
    Pc = ndimage.median_filter(S, size=(lf, 1), mode="reflect")
    h2, p2 = H * H, Pc * Pc
    mask = (h2 if p["part"] == "harmonic" else p2) / np.maximum(h2 + p2, EPS)
    rest = 10 ** (p["rest_db"] / 20) if p["rest_db"] > -120 else 0.0
    return X * (mask + rest * (1 - mask))


# ---------------------------------------------------------------------------------------------------- time ops

def _autocorr(x, order):
    if len(x) <= order:
        return np.zeros(order + 1)
    w = x * np.hanning(len(x))
    return np.correlate(w, w, "full")[len(w) - 1:len(w) + order] / len(w)


def ar_coeffs(x, order, *more):
    """Yule-Walker AR coefficients a (x[n] ~ sum a_i x[n-i]) of Hann-windowed blocks (their autocorrelations
    summed, so a gap between blocks does not enter the estimate)."""
    r = sum(_autocorr(b, order) for b in (x,) + more)
    if r[0] <= 0:
        return np.zeros(order)
    r = r.copy()
    r[0] *= 1 + 1e-9
    return linalg.solve_toeplitz(r[:order], r[1:order + 1])


def detect_clicks(x, sr, sensitivity, order=24, block=4096):
    """Clicks as (start, end) inclusive sample spans: AR prediction error above `sensitivity` times its local robust
    level (1.4826 x the running median of |error| over 10 ms), so speech onsets and loud passages raise the bar."""
    e = np.zeros(len(x))
    for b0 in range(0, len(x), block):
        b1 = min(len(x), b0 + block)
        seg = x[max(0, b0 - order):b1]
        if len(seg) <= 2 * order:
            continue
        a = ar_coeffs(seg, order)
        r = seg[order:] - np.convolve(seg, np.concatenate([[0], a]), "full")[order:len(seg)]
        start = max(0, b0 - order) + order
        e[start:start + len(r)] = r
    sigma = 1.4826 * ndimage.median_filter(np.abs(e), size=max(3, int(0.01 * sr)), mode="nearest")
    flags = np.abs(e) > sensitivity * sigma + 1e-9
    spans, gap = [], max(2, order // 4)
    for s, e in _runs(flags):
        if spans and s - spans[-1][1] <= gap:
            spans[-1] = (spans[-1][0], e)
        else:
            spans.append((s, e))
    # the error jumps where a click starts, but a decaying click is predictable after that: cover `order` more samples
    return [(max(0, s - 2), min(len(x) - 1, e + order // 4)) for s, e in spans]


def lsar_fill(x, s, e, order, fit=2048):
    """Least-squares AR interpolation of x[s..e] from both sides at once (Janssen, Veldhuis & Vries 1986). The model
    order is at least 3x the gap length (their rule of thumb), at most 160; it is fitted on up to `fit` samples on each
    side, and the solve uses 3 * order samples (at least 64) on each side."""
    order = int(min(160, max(order, 3 * (e - s + 1))))
    ctxn = max(3 * order, 64)
    a0, a1 = max(0, s - ctxn), min(len(x), e + 1 + ctxn)
    seg = x[a0:a1].copy()
    known = np.ones(len(seg), bool)
    known[s - a0:e - a0 + 1] = False
    if s - a0 < order and a1 - e - 1 < order:
        return x[s:e + 1]
    a = ar_coeffs(x[max(0, s - fit):s], order, x[e + 1:e + 1 + fit])
    c = np.concatenate([[1.0], -a])
    rows = len(seg) - order
    A = np.zeros((rows, len(seg)))
    for i in range(order + 1):  # row r: e[r + order] = sum_i c_i seg[r + order - i]
        A[np.arange(rows), np.arange(rows) + order - i] = c[i]
    Au, Ak = A[:, ~known], A[:, known]
    sol, *_ = np.linalg.lstsq(Au, -Ak @ seg[known], rcond=1e-10)
    peak = np.abs(seg[known]).max()
    if np.abs(sol).max() > 1.2 * peak:  # implausible (nearly singular model): a straight line between the edges
        lo, hi = x[s - 1] if s > 0 else 0.0, x[e + 1] if e + 1 < len(x) else 0.0
        sol = lo + (hi - lo) * (np.arange(1, e - s + 2) / (e - s + 2))
    return np.clip(sol, -0.99, 0.99)  # a fill never clips


def op_declick(x, p, ctx):
    y = x.copy()
    maxlen = max(1, int(p["max_ms"] / 1000 * ctx["sr"]))
    for _ in range(3):  # later passes refit the model on the repaired signal
        for s, e in detect_clicks(y, ctx["sr"], p["sensitivity"], p["order"]):
            if e - s + 1 <= maxlen:
                y[s:e + 1] = lsar_fill(y, s, e, p["order"])
                ctx["info"]["clicks_repaired"] = ctx["info"].get("clicks_repaired", 0) + 1
    return y


STFT_OPS = {"gain": op_gain, "dehum": op_dehum, "denoise": op_denoise, "heal": op_heal, "split_hp": op_split_hp}
TIME_OPS = {"declick": op_declick}


# ---------------------------------------------------------------------------------------------------- render

def apply_op(op, x, sr, prog, info):
    """One node on one channel. Records the node's own measured effect in `info`: the energy change inside its
    selection and outside it, so a revision knows which node did what."""
    p = op["params"]
    if op["op"] in TIME_OPS:
        y = TIME_OPS[op["op"]](x, p, {"sr": sr, "info": info})
        w = time_weight(op["at"], len(x), sr, prog.size, prog.hop) * op["mix"]
        return x + w * (y - x)
    size = p.get("fft") or prog.size
    hop = size // prog.overlap
    X = stft(x, size, hop)
    freqs, times = bin_freqs(size, sr), frame_times(X.shape[1], size, hop, sr)
    M = sel_mask(op["at"], freqs, times) if op["at"] is not None else np.ones(X.shape)
    ctx = {"sr": sr, "size": size, "hop": hop, "freqs": freqs, "times": times, "mask": M, "info": info,
           "sels": prog.sels}
    Y = STFT_OPS[op["op"]](X, p, ctx)
    Z = X + op["mix"] * M * (Y - X)
    for key, w in (("inside_db", M), ("outside_db", 1 - M)):
        eb, ea = float((np.abs(X) ** 2 * w).sum()), float((np.abs(Z) ** 2 * w).sum())
        if eb > 1e-20:
            info.setdefault(key, []).append(round(10 * math.log10(max(ea, 1e-30) / eb), 2))
    return istft(Z, size, hop, len(x))


def render(prog, audio, sr):
    """Checked program, audio (channels, n) or (n,) -> ({layer: audio (channels, n)}, info per node). A program
    whose output mixes layers also gets out["mix"]."""
    a = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    out, info = {}, {}
    for lname, chain in prog.layers:
        y = a.copy()
        for oname in chain:
            inf = info.setdefault(f"{lname}.{oname}", {})
            if prog.ops[oname]["off"]:
                inf["off"] = True
                continue
            y = np.stack([apply_op(prog.ops[oname], ch, sr, prog, inf) for ch in y])
        out[lname] = y
    if prog.output == "mix":
        out["mix"] = sum(10 ** (g / 20) * out[k] for k, g in prog.output_mix.items())
    for inf in info.values():  # one number per node: the mean over channels
        for key in ("inside_db", "outside_db"):
            if key in inf:
                inf[key] = round(float(np.mean(inf[key])), 2)
    return out, info
