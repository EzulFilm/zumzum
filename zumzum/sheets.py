"""Before / after sheets for people (review output only; never an input to an edit).

Layout, top to bottom, all on one time axis: waveform (before in grey, after in colour on top), spectrogram before,
spectrogram after, and a change map (after vs before in dB: blue = removed, red = added, white = unchanged). Display
conventions common to spectral editors, chosen for legibility: a log frequency axis from 30 Hz so hum and rumble are
visible, a high-resolution analysis with a low-sidelobe window (Blackman-Harris, display only), a fixed level range
of `range_db` below the loudest cell of the input, a gamma brightness curve, and a level legend beside each plot.
An optional second column zooms into one region on a linear frequency axis.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm, TwoSlopeNorm  # noqa: E402
from scipy import signal  # noqa: E402

BG, FG = "#0d0f14", "#d8dbe2"


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


def sheet(path, before, after, sr, title, checks, program_name, fmax=8000, zoom=None, range_db=90, gamma=0.7,
          **_):
    b, a = np.atleast_2d(before), np.atleast_2d(after)
    dur = b.shape[1] / sr
    Sb, f, t = _spec(b, sr)
    Sa, _, _ = _spec(a, sr)
    top = np.percentile(Sb, 99.9)
    vmin, vmax = top - range_db, top
    norm = PowerNorm(gamma=1 / gamma, vmin=vmin, vmax=vmax, clip=True)
    # change map only where something was there to change (either side above the bottom of the range)
    ch = np.where(np.maximum(Sb, Sa) > vmin + 10, Sa - Sb, 0.0)
    cnorm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
    ncol = 2 if zoom else 1
    fig = plt.figure(figsize=(17 if zoom else 14, 13), facecolor=BG)
    widths = [3, 2] if zoom else [1]
    gs = fig.add_gridspec(4, ncol, height_ratios=[1, 2.2, 2.2, 2.2], width_ratios=widths, hspace=0.32, wspace=0.16,
                          left=0.06, right=0.92, top=0.93, bottom=0.07 + 0.017 * len(checks))
    cols = [(0, dur, 30, min(fmax, sr / 2), True)]
    if zoom:
        cols.append((zoom[0], zoom[1], zoom[2], zoom[3], False))
    for j, (t0, t1, f0, f1, log) in enumerate(cols):
        ax = fig.add_subplot(gs[0, j])
        _style(ax)
        ts = np.arange(b.shape[1]) / sr
        step = max(1, int((t1 - t0) * sr / 6000))
        sl = slice(int(t0 * sr), int(t1 * sr), step)
        ax.plot(ts[sl], b.mean(axis=0)[sl], color="#7a808c", lw=0.6, label="before")
        ax.plot(ts[sl], a.mean(axis=0)[sl], color="#4fc3f7", lw=0.6, alpha=0.9, label="after")
        ax.set_xlim(t0, t1)
        lim = max(1e-3, np.abs(b.mean(axis=0)[sl]).max()) * 1.05
        ax.set_ylim(-lim, lim)
        ax.set_yticks([])
        ax.set_title("waveform: before (grey), after (blue)" + ("" if j == 0 else "  [zoom]"), fontsize=10, loc="left",
                     color=FG)
        for i, (S, cap, cmap, nm) in enumerate([(Sb, "before (input)", "inferno", norm),
                                                (Sa, "after (program output)", "inferno", norm),
                                                (ch, "change, after vs before (dB): blue removed, red added",
                                                 "RdBu_r", cnorm)]):
            ax = fig.add_subplot(gs[i + 1, j])
            _style(ax)
            im = ax.pcolormesh(t, np.maximum(f, 1), S, cmap=cmap, norm=nm, shading="auto", rasterized=True)
            ax.set_xlim(t0, t1)
            _freq_axis(ax, f0, f1, log)
            ax.set_title(cap + ("" if j == 0 else "  [zoom, linear Hz]"), fontsize=10, loc="left", color=FG)
            if i == 2:
                ax.set_xlabel("time (s)")
            if j == ncol - 1:
                cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
                cb.ax.tick_params(colors=FG, labelsize=8)
                cb.set_label("dB change" if i == 2 else "dBFS", color=FG, fontsize=9)
    fig.suptitle(title, color=FG, fontsize=14, x=0.06, y=0.99, ha="left", parse_math=False)
    fig.text(0.06, 0.957, f"program: {program_name}", color="#9aa0ac", fontsize=10, parse_math=False)
    lines = [f"{'PASS' if c['pass'] else 'FAIL'}  {c['name']}: {c['text']}" for c in checks]
    for k, line in enumerate(lines):
        fig.text(0.06, 0.012 + 0.017 * (len(lines) - 1 - k), line, fontsize=10, family="monospace", parse_math=False,
                 color="#8fe388" if line.startswith("PASS") else "#ff8a80")
    fig.savefig(path, dpi=90, facecolor=BG)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- display spectrogram

def pretty_spec(a, sr, fmin=30.0, fmax=8000.0, rows=480, dt=0.004):
    """A display-only multi-resolution spectrogram on a log-frequency grid: long windows where the ear needs pitch
    detail (low frequencies), short ones where it needs timing (high), blended smoothly in dB. Returns (S_db, f, t)."""
    from scipy import signal as sg
    x = np.atleast_2d(a).mean(axis=0)
    n = len(x)
    t = np.arange(0, n / sr, dt)
    f = np.geomspace(fmin, min(fmax, sr / 2 * 0.98), rows)
    layers = []
    for size in (8192, 2048, 512):
        hop = max(64, int(dt * sr / 2))
        w = sg.windows.blackmanharris(size, sym=False)
        ff, tt, Z = sg.stft(x, sr, window=w, nperseg=size, noverlap=size - hop, boundary="even", padded=True)
        D = 10 * np.log10(4 * np.abs(Z) ** 2 + 1e-14)
        D += 10 * np.log10(size / 2048)  # put the three sizes on one level scale for broadband sound
        # frequency first (log grid), then time
        Df = np.array([np.interp(f, ff, D[:, j]) for j in range(D.shape[1])]).T
        layers.append(np.array([np.interp(t, tt, row) for row in Df]))
    lf = np.log2(f)
    w_lo = np.clip((np.log2(400) - lf) / 1.0 + 0.5, 0, 1)        # 8192 below ~400 Hz
    w_hi = np.clip((lf - np.log2(3000)) / 1.0 + 0.5, 0, 1)       # 512 above ~3 kHz
    w_mid = 1 - w_lo - w_hi
    S = layers[0] * w_lo[:, None] + layers[1] * np.clip(w_mid, 0, 1)[:, None] + layers[2] * w_hi[:, None]
    return S, f, t


def zz_cmap():
    """Zumzum's own display colours: black, deep blue, cyan, near white."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("zumzum", ["#05060a", "#0b1d3d", "#0f5f8f", "#21b6c9", "#9ff2e6",
                                                        "#ffffff"])


def pretty_panel(ax, S, f, t, top, title, range_db=80, gamma=0.75):
    from matplotlib.colors import PowerNorm
    norm = PowerNorm(gamma=1 / gamma, vmin=top - range_db, vmax=top, clip=True)
    ax.set_facecolor("#05060a")
    ax.imshow(S, origin="lower", aspect="auto", cmap=zz_cmap(), norm=norm, interpolation="bilinear",
              extent=[t[0], t[-1], 0, len(f)])
    ticks = [t_ for t_ in (63, 125, 250, 500, 1000, 2000, 4000, 8000) if f[0] <= t_ <= f[-1]]
    ax.set_yticks([np.interp(np.log2(v), np.log2(f), np.arange(len(f))) for v in ticks])
    ax.set_yticklabels([f"{v / 1000:g}k" if v >= 1000 else f"{v}" for v in ticks])
    for s in ax.spines.values():
        s.set_color("#2a2f3a")
    ax.tick_params(colors="#9aa0ac", labelsize=9)
    ax.set_title(title, fontsize=11, loc="left", color="#d8dbe2", parse_math=False)
