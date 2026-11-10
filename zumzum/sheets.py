"""Before / after sheets for people (review output only; never an input to an edit).

Layout, top to bottom, all on one time axis: waveform (before in grey, after in colour on top), spectrogram before,
spectrogram after, and a change map (after vs before in dB: blue = removed, amber = added, dark = unchanged).

How the spectrograms are made legible (display only, never used by an edit):
- multi-resolution analysis: a long window below ~400 Hz (pitch detail for hum and bass), a medium one for the voice
  range and a short one above ~3 kHz (clicks and consonants stay sharp in time), blended over an octave per crossover;
- each analysis is moved onto the display grid (log frequency from 30 Hz, a fixed time step) by linear interpolation
  where the grid is finer than the analysis and by the mean power over each grid cell where it is coarser, so narrow
  lines and single clicks are neither blocky nor skipped; the image is then drawn with smooth (bilinear) resampling;
- a low-sidelobe window (Blackman-Harris) so a 90 dB range is real, not window leakage;
- levels: the top is the loudest 0.1 % of the input; the bottom sits a fixed margin under the input's noise floor
  (at most `range_db` below the top), so hiss reads as a dim haze and anything removed below it goes black;
- Zumzum's own colour scale, black through deep blue and teal to ivory with evenly rising lightness, and a gamma
  curve that keeps the floor dark.
An optional second column zooms into one region on a linear frequency axis.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm  # noqa: E402
from scipy import signal  # noqa: E402

BG, FG = "#0d0f14", "#d8dbe2"
FLOOR = "#030409"  # the colour of silence in a spectrogram (bottom of zz_cmap)


def _spec(a, sr, size=4096, hop=256):
    w = signal.windows.blackmanharris(size, sym=False)
    x = np.atleast_2d(a)
    S = 0
    for ch in x:
        f, t, Z = signal.stft(ch, sr, window=w, nperseg=size, noverlap=size - hop, boundary="even", padded=True)
        S = S + np.abs(Z) ** 2
    S = 4 * S / x.shape[0]  # scipy scales by 1 / sum(window): a full-scale sine reads 0 dB in its peak bin
    return 10 * np.log10(S + 1e-16), f, t


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#444a56")
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def _freq_axis(ax, fmin, fmax, log):
    if log:
        ax.set_yscale("log")
        ticks = [t for t in (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000) if fmin <= t <= fmax]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t / 1000:g}k" if t >= 1000 else f"{t:g}" for t in ticks])
        ax.minorticks_off()
    ax.set_ylim(fmin, fmax)
    ax.set_ylabel("Hz")


# ------------------------------------------------------------------------------- multi-resolution spectrogram

_CROSS_HZ = (400.0, 3000.0)  # long | medium | short window crossovers, each blended over one octave


def _sizes(sr):
    """Window lengths (long, medium, short): 8192 / 2048 / 512 samples at 44.1 kHz, the same durations at other
    rates (rounded to a power of two)."""
    return tuple(int(2 ** max(6, round(np.log2(s * sr / 44100)))) for s in (8192, 2048, 512))


def _band_weights(f, cross=_CROSS_HZ):
    """Blend weight of each window length for each display frequency (columns long, medium, short; rows sum to 1)."""
    lf = np.log2(np.maximum(f, 1e-3))
    w_lo = np.clip(np.log2(cross[0]) - lf + 0.5, 0, 1)
    w_hi = np.clip(lf - np.log2(cross[1]) + 0.5, 0, 1)
    return np.stack([w_lo, np.clip(1 - w_lo - w_hi, 0, 1), w_hi], axis=1)


def _edges(x):
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    m = (x[1:] + x[:-1]) / 2
    return np.concatenate([[2 * x[0] - m[0]], m, [2 * x[-1] - m[-1]]])


def _interp0(x, xp, Y):
    """Linear interpolation of the rows of Y (sampled at increasing xp) at positions x, held flat outside xp."""
    if len(xp) == 1:
        return np.repeat(Y[:1], len(x), axis=0)
    i = np.clip(np.searchsorted(xp, x), 1, len(xp) - 1)
    w = np.clip((x - xp[i - 1]) / (xp[i] - xp[i - 1]), 0, 1)[:, None]
    return Y[i - 1] * (1 - w) + Y[i] * w


def _resample(D, src, dst):
    """Rows of dB values D sampled at increasing positions src, moved to positions dst: linear interpolation (in dB)
    where dst is finer than src, the mean power over each dst cell where it is coarser."""
    lin = _interp0(dst, src, D)
    se, de = _edges(src), _edges(dst)
    coarse = np.diff(de) > np.interp(dst, src, np.diff(se))
    if not coarse.any():
        return lin
    P = 10 ** (D / 10)
    C = np.concatenate([np.zeros((1,) + D.shape[1:]), np.cumsum(P * np.diff(se)[:, None], axis=0)])
    lo, hi = de[:-1][coarse], de[1:][coarse]
    mean = (_interp0(hi, se, C) - _interp0(lo, se, C)) / (hi - lo)[:, None]
    lin[coarse] = 10 * np.log10(np.maximum(mean, 1e-20))
    return lin


def _stft_db(x, sr, size, hop, fmax, chunk=256):
    """Power spectrogram in dB of the channel mean, frames centred at k * hop, bins up to fmax only; computed in
    chunks of frames so long files stay small in memory. A full-scale sine reads 0 dB in its peak bin."""
    w = signal.windows.blackmanharris(size, sym=False)
    scale = 4 / w.sum() ** 2 / x.shape[0]
    nb = min(size // 2 + 1, int(np.ceil(fmax * size / sr)) + 3)
    n = x.shape[1]
    pad = ((0, 0), (size // 2, size // 2 + hop))
    xp = np.pad(x, pad, mode="reflect" if n > size // 2 + hop else "constant")
    nfr = n // hop + 1
    frames = np.lib.stride_tricks.sliding_window_view(xp, size, axis=-1)[:, ::hop][:, :nfr]
    out = np.empty((nb, nfr), dtype=np.float64)
    for k in range(0, nfr, chunk):
        Z = np.fft.rfft(frames[:, k:k + chunk] * w, axis=-1)[..., :nb]
        out[:, k:k + chunk] = (np.abs(Z) ** 2).sum(axis=0).T * scale
    return 10 * np.log10(out + 1e-16), np.arange(nb) * sr / size, np.arange(nfr) * hop / sr


def _mr_spec(a, sr, f, t, cross=_CROSS_HZ):
    """Multi-resolution spectrogram (dB) of `a` on the display grid f (Hz, increasing) x t (s, increasing).

    Levels follow the medium window: steady noise reads the same in every band; a pure tone reads about +6 dB in the
    long-window band and -6 dB in the short-window band."""
    x = np.atleast_2d(np.asarray(a, dtype=np.float64))
    dur = x.shape[1] / sr
    S = np.full((len(f), len(t)), np.nan)
    if t[0] > dur:
        return S
    weights = _band_weights(f, cross)
    sizes = _sizes(sr)
    # only the stretch of audio the time grid needs, with room for the longest window
    m = sizes[0]
    i0 = max(0, int(np.floor(t[0] * sr)) - m)
    i1 = min(x.shape[1], int(np.ceil(t[-1] * sr)) + m + 1)
    x = x[:, i0:i1]
    S[:] = 0.0
    dt = (t[-1] - t[0]) / max(1, len(t) - 1) if len(t) > 1 else 0.01
    for j, size in enumerate(sizes):
        rows = weights[:, j] > 0
        if not rows.any():
            continue
        hop = int(np.clip(dt * sr / 2, max(16, size // 8), size // 2))
        D, ff, tt = _stft_db(x, sr, size, hop, f[rows][-1])
        D += 10 * np.log10(size / sizes[1])
        Df = _resample(D, ff, f[rows])                      # frequency first, onto the display rows
        S[rows] += weights[rows, j][:, None] * _resample(Df.T, tt + i0 / sr, t).T
    S[:, t > dur] = np.nan
    return S


def pretty_spec(a, sr, fmin=30.0, fmax=8000.0, rows=480, dt=0.004):
    """A display-only multi-resolution spectrogram on a log-frequency grid: long windows where the ear needs pitch
    detail (low frequencies), short ones where it needs timing (high), blended smoothly in dB. Returns (S_db, f, t)."""
    n = np.atleast_2d(a).shape[-1]
    dur = n / sr
    dt = max(dt, dur / 8000)  # at most 8000 columns, whatever the length
    t = np.arange(0, dur, dt)
    f = np.geomspace(fmin, min(fmax, sr / 2 * 0.98), rows)
    return _mr_spec(a, sr, f, t), f, t


def zz_cmap():
    """Zumzum's own display colours: near black, deep blue, teal, mint, ivory; lightness rises evenly (CIE L* about
    1, 14, 27, 42, 58, 73, 88, 99) so equal steps in level look like equal steps in brightness."""
    return LinearSegmentedColormap.from_list("zumzum", [FLOOR, "#122150", "#173f74", "#13698f", "#1f9aa6", "#4fc7b0",
                                                        "#a6ebc8", "#f6fff0"])


def zz_change_cmap():
    """Change map colours: blue = removed, dark = unchanged, amber = added (lightness grows with the change)."""
    return LinearSegmentedColormap.from_list("zumzum_change", ["#5cb6ff", "#3567a8", "#11141c", "#a3641c",
                                                               "#ffb347"])


def levels(S, range_db=90, margin_db=18, min_range_db=45):
    """(vmin, vmax) for a spectrogram: top = the loudest 0.1 % of cells; bottom = `margin_db` under the noise floor
    (the typical level of each band in its quietest 10 % of moments), never more than `range_db` below the top and
    never less than `min_range_db`."""
    top = float(np.nanpercentile(S, 99.9))
    floor = float(np.nanmedian(np.nanpercentile(S, 10, axis=1)))
    vmin = max(top - range_db, floor - margin_db)
    return min(vmin, top - min_range_db), top


def _log_ticks(ax, f0, f1):
    ticks = [v for v in (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000) if f0 <= v <= f1]
    ax.set_yticks(np.log2(ticks))
    ax.set_yticklabels([f"{v / 1000:g}k" if v >= 1000 else f"{v:g}" for v in ticks])
    ax.set_ylim(np.log2(f0), np.log2(f1))


def _show(ax, S, f, t, log, cmap, norm):
    """Draw a spectrogram computed on grid f x t (log-spaced f when `log`) with smooth resampling."""
    y0, y1 = (np.log2(f[0]), np.log2(f[-1])) if log else (f[0], f[-1])
    im = ax.imshow(S, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="bilinear",
                   extent=[t[0], t[-1], y0, y1], rasterized=True)
    if log:
        _log_ticks(ax, f[0], f[-1])
    else:
        ax.set_ylim(f[0], f[-1])
    ax.set_ylabel("Hz")
    return im


def sheet(path, before, after, sr, title, checks, program_name, fmax=8000, zoom=None, range_db=90, gamma=0.7,
          **_):
    b, a = np.atleast_2d(before), np.atleast_2d(after)
    dur = max(b.shape[1], a.shape[1]) / sr
    ncol = 2 if zoom else 1
    # display grids: about two cells per screen pixel in time, 420 log-spaced rows
    cols = [(0, dur, 30, min(fmax, sr / 2 * 0.98), True)]
    if zoom:
        cols.append((zoom[0], min(zoom[1], dur), max(zoom[2], 1), min(zoom[3], sr / 2 * 0.98), False))
    grids = []
    for j, (t0, t1, f0, f1, log) in enumerate(cols):
        nt = 2400 if ncol == 1 else (2000 if j == 0 else 1400)
        t = np.linspace(t0, t1, nt)
        f = np.geomspace(f0, f1, 420) if log else np.linspace(f0, f1, 420)
        # a linear zoom shows low frequencies stretched out: keep the long window up to half its top
        cross = _CROSS_HZ if log else (max(_CROSS_HZ[0], f1 / 2), max(_CROSS_HZ[1], f1 * 2))
        # the change map skips the long window: it smears each click's change into a 0.2 s stripe in the bass
        ccross = (min(20.0, f0), cross[1])
        grids.append((f, t, _mr_spec(b, sr, f, t, cross), _mr_spec(a, sr, f, t, cross),
                      _mr_spec(b, sr, f, t, ccross), _mr_spec(a, sr, f, t, ccross)))
    vmin, vmax = levels(grids[0][2], range_db)
    norm = PowerNorm(gamma=1 / gamma, vmin=vmin, vmax=vmax, clip=True)
    cnorm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
    cmap, ccmap = zz_cmap(), zz_change_cmap()
    fig = plt.figure(figsize=(17 if zoom else 14, 13), facecolor=BG)
    widths = [3, 2] if zoom else [1]
    gs = fig.add_gridspec(4, ncol, height_ratios=[1, 2.2, 2.2, 2.2], width_ratios=widths, hspace=0.32, wspace=0.16,
                          left=0.06, right=0.92, top=0.93, bottom=0.07 + 0.017 * len(checks))
    for j, (t0, t1, f0, f1, log) in enumerate(cols):
        f, t, Sb, Sa, Cb, Ca = grids[j]
        ax = fig.add_subplot(gs[0, j])
        _style(ax)
        ts = np.arange(b.shape[1]) / sr
        step = max(1, int((t1 - t0) * sr / 6000))
        sl = slice(int(t0 * sr), int(t1 * sr), step)
        ax.plot(ts[sl], b.mean(axis=0)[sl], color="#7a808c", lw=0.6, label="before")
        ta = np.arange(a.shape[1]) / sr  # after may differ in length (speed, cut, crop)
        ax.plot(ta[sl], a.mean(axis=0)[sl], color="#4fc7b0", lw=0.6, alpha=0.9, label="after")
        ax.set_xlim(t0, t1)
        # both waveforms fit: an after made louder (gain, loudness) must not run off the plot
        lim = max([1e-3] + [np.abs(w).max() for w in (b.mean(axis=0)[sl], a.mean(axis=0)[sl]) if w.size]) * 1.05
        ax.set_ylim(-lim, lim)
        ax.set_yticks([])
        ax.set_title("waveform: before (grey), after (teal)" + ("" if j == 0 else "  [zoom]"), fontsize=10, loc="left",
                     color=FG)
        # change map only where both signals exist and something was there to change
        both = np.isfinite(Cb) & np.isfinite(Ca)
        ch = np.where(both, np.where(np.maximum(Cb, Ca) > vmin + 10, Ca - Cb, 0.0), np.nan)
        for i, (S, cap, cm, nm) in enumerate([(Sb, "before (input)", cmap, norm),
                                              (Sa, "after (program output)", cmap, norm),
                                              (ch, "change, after vs before (dB): blue removed, amber added",
                                               ccmap, cnorm)]):
            ax = fig.add_subplot(gs[i + 1, j])
            _style(ax)
            ax.set_facecolor(FLOOR)
            im = _show(ax, S, f, t, log, cm, nm)
            ax.set_xlim(t0, t1)
            ax.set_title(cap + ("" if j == 0 else "  [zoom, linear Hz]"), fontsize=10, loc="left", color=FG)
            if i == 2:
                ax.set_xlabel("time (s)")
            if j == ncol - 1:
                cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
                if i < 2:  # the power scale squeezes the quiet end: keep only labels that do not touch
                    keep, last = [], None
                    for v in np.arange(np.floor(vmax / 10) * 10, vmin - 1, -10):
                        y = float(norm(v))
                        if last is None or last - y >= 0.07:
                            keep.append(v)
                            last = y
                    cb.set_ticks(keep)
                cb.ax.tick_params(colors=FG, labelsize=8)
                cb.outline.set_edgecolor("#444a56")
                cb.set_label("dB change" if i == 2 else "dBFS", color=FG, fontsize=9)
    fig.suptitle(title, color=FG, fontsize=14, x=0.06, y=0.99, ha="left", parse_math=False)
    fig.text(0.06, 0.957, f"program: {program_name}", color="#9aa0ac", fontsize=10, parse_math=False)
    lines = [f"{'PASS' if c['pass'] else 'FAIL'}  {c['name']}: {c['text']}" for c in checks]
    for k, line in enumerate(lines):
        fig.text(0.06, 0.012 + 0.017 * (len(lines) - 1 - k), line, fontsize=10, family="monospace", parse_math=False,
                 color="#8fe388" if line.startswith("PASS") else "#ff8a80")
    fig.savefig(path, dpi=90, facecolor=BG)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- display panels

def pretty_panel(ax, S, f, t, top, title, range_db=80, gamma=0.75):
    norm = PowerNorm(gamma=1 / gamma, vmin=top - range_db, vmax=top, clip=True)
    ax.set_facecolor(FLOOR)
    ax.imshow(S, origin="lower", aspect="auto", cmap=zz_cmap(), norm=norm, interpolation="bilinear",
              extent=[t[0], t[-1], 0, len(f)])
    ticks = [t_ for t_ in (63, 125, 250, 500, 1000, 2000, 4000, 8000) if f[0] <= t_ <= f[-1]]
    ax.set_yticks([np.interp(np.log2(v), np.log2(f), np.arange(len(f))) for v in ticks])
    ax.set_yticklabels([f"{v / 1000:g}k" if v >= 1000 else f"{v}" for v in ticks])
    for s in ax.spines.values():
        s.set_color("#2a2f3a")
    ax.tick_params(colors="#9aa0ac", labelsize=9)
    ax.set_title(title, fontsize=11, loc="left", color="#d8dbe2", parse_math=False)
