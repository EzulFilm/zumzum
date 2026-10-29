"""README figures: docs/img/readme_program.png (a program beside its before / after) and docs/img/readme_examples.png
(every example, before | after). Run tools/make_examples.py (and the mp3 example, see README) first.

    python tools/make_readme_figures.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

from zumzum.audio import load  # noqa: E402
from zumzum.sheets import BG, FG, _freq_axis, _spec, _style, pretty_panel, pretty_spec  # noqa: E402

OUT = ROOT / "out" / "examples"
IMG = ROOT / "docs" / "img"
EXAMPLES = [  # folder, before, after, title, program
    ("cylinder_denoise", "before.wav", "after.wav", "1888 wax cylinder: less hiss, keep the words", "cylinder_denoise.zz"),
    ("hum_voice", "before.wav", "after.wav", "60 Hz hum under a voice", "hum_voice.zz"),
    ("record_declick", "before.wav", "after.wav", "1917 78 rpm record: clicks out", "record_declick.zz"),
    ("phone_heal", "before.wav", "after.wav", "Phone dialling over a reading: tones and rumble healed",
     "phone_heal.zz"),
    ("drums_down", "before.wav", "after.wav", "Drums 12 dB down under a piano", "drums_down.zz"),
    ("mp3_record", "before.mp3", "after.mp3", "1920 acoustic record, mp3 in, mp3 out: clicks and hiss",
     "mp3_record.zz"),
]


def spec(path, t1=None):
    x, sr = load(path, 0, t1)
    S, f, t = _spec(x, sr, size=2048, hop=256)
    return S, f, t, sr


def panel(ax, S, f, t, norm, title, fmax=8000):
    _style(ax)
    ax.pcolormesh(t, np.maximum(f, 1), S, cmap="inferno", norm=norm, shading="auto", rasterized=True)
    _freq_axis(ax, 30, fmax, True)
    ax.set_xlim(t[0], t[-1])
    ax.set_title(title, fontsize=11, loc="left", color=FG)


# token colours for the program panel (our own palette)
KEYWORDS = {"zumzum", "name", "about", "analysis", "sel", "node", "layer", "output", "check", "at", "mix", "off", "on",
            "pass", "and", "or", "not", "true", "false"}
COL = {"kw": "#c792ea", "call": "#82aaff", "num": "#f78c6c", "str": "#c3e88d", "key": "#8b93a7", "text": "#e6e9ef",
       "op": "#89ddff"}
TOKEN = re.compile(r'(?P<str>"[^"]*")|(?P<num>-?\d+(?:\.\d+)?)|(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>->|\.\.|[=(),:])'
                   r'|(?P<sp>\s+)|(?P<other>.)')


def program_lines(src):
    """The program as shown: comments and the name / about lines left out, at most one blank line in a row."""
    out = []
    for ln in src.splitlines():
        ln = ln.split("#")[0].rstrip()
        if ln.startswith(("name", "about")) or (not ln and out and not out[-1]):
            continue
        out.append(ln)
    while out and not out[-1]:
        out.pop()
    return out


def tokens(line):
    """(text, colour) pieces of one line: keywords, op / selection / check calls, parameter names, numbers."""
    out = []
    parts = [(m.lastgroup, m.group()) for m in TOKEN.finditer(line)]
    for i, (kind, text) in enumerate(parts):
        nxt = next((t for k, t in parts[i + 1:] if k != "sp"), "")
        if kind == "word":
            kind = "kw" if text in KEYWORDS else "call" if nxt == "(" else "key" if nxt == ":" else "text"
        elif kind in ("sp", "other"):
            kind = "text"
        out.append((text, COL.get(kind, COL["text"])))
    return out


def code_panel(ax, lines, size=11.5):
    """Monospace lines with highlighted tokens; positions come from the measured width of one character."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(matplotlib.patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.025",
                                                   transform=ax.transAxes, facecolor="#161922", edgecolor="#2a2f3a"))
    fig = ax.figure
    probe = ax.text(0, 0, "M" * 40, family="monospace", fontsize=size)
    fig.canvas.draw()
    bb = probe.get_window_extent()
    probe.remove()
    axw, axh = ax.get_window_extent().width, ax.get_window_extent().height
    cw, lh = bb.width / 40 / axw, bb.height * 1.55 / axh
    for row, line in enumerate(lines):
        x, y = 0.025, 1 - 0.035 - 0.5 * lh - row * lh
        for text, colour in tokens(line):
            ax.text(x, y, text, family="monospace", fontsize=size, color=colour, va="top", ha="left")
            x += cw * len(text)


def program_figure():
    import json
    name, b, a, title, prog = EXAMPLES[0]
    lines = program_lines((ROOT / "examples" / prog).read_text(encoding="utf-8"))
    xb, sr = load(OUT / name / b, 0, 20)
    xa, _ = load(OUT / name / a, 0, 20)
    Sb, f, t = pretty_spec(xb, sr)
    Sa, _, _ = pretty_spec(xa, sr)
    top = np.percentile(Sb, 99.7)
    res = {c["name"]: c["value"] for c in json.loads((OUT / name / "results.json").read_text())["checks"]}
    code_h = 0.45 + 0.25 * len(lines)
    fig = plt.figure(figsize=(16, 4.6 + code_h + 1.0), facecolor=BG)
    H = 4.6 + code_h + 1.0
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.09, 1], height_ratios=[4.6, code_h], left=0.04, right=0.98,
                          top=1 - 0.45 / H, bottom=0.5 / H, wspace=0.03, hspace=0.95 / 4.6)
    ax_b, ax_a = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])
    pretty_panel(ax_b, Sb, f, t, top, "before: Edison wax cylinder, 1888")
    pretty_panel(ax_a, Sa, f, t, top, f"after: hiss in a pause {-res['hiss_down']:.0f} dB, words band "
                                f"{res['words_kept']:+.1f} dB (measured)")
    for ax in (ax_b, ax_a):
        ax.set_xlabel("time (s)", color="#9aa0ac")
    ax_b.set_ylabel("Hz", color="#9aa0ac")
    arrow = fig.add_subplot(gs[0, 1])
    arrow.axis("off")
    arrow.annotate("", xy=(0.92, 0.5), xytext=(0.08, 0.5), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.5", color="#9aa0ac", lw=2.5))
    ax_c = fig.add_subplot(gs[1, :])
    code_panel(ax_c, lines)
    ax_c.set_title(f"the Zumzum program between them (examples/{prog})", color=FG, fontsize=13, loc="left",
                   fontweight="bold")
    fig.text(0.5, 0.15 / H, "Audio: Arthur Sullivan's speech on an Edison wax cylinder, 1888, public domain "
             "(Wikimedia Commons)", color="#8b93a7", fontsize=9.5, ha="center")
    fig.savefig(IMG / "readme_program.png", dpi=90, facecolor=BG)
    plt.close(fig)


def examples_figure():
    rows = [e for e in EXAMPLES if (OUT / e[0] / e[1]).exists()]
    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 2.7 * len(rows) + 0.8), facecolor=BG, squeeze=False)
    for i, (name, b, a, title, prog) in enumerate(rows):
        Sb, f, t, _ = spec(OUT / name / b, 20)
        Sa, _, _, _ = spec(OUT / name / a, 20)
        top = np.percentile(Sb, 99.9)
        norm = PowerNorm(gamma=1 / 0.7, vmin=top - 90, vmax=top, clip=True)
        panel(axes[i][0], Sb, f, t, norm, f"{title}  (before)")
        panel(axes[i][1], Sa, f, t, norm, f"after: examples/{prog}")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Before | after, every example (log frequency from 30 Hz, 90 dB range)", color=FG, fontsize=15,
                 x=0.04, ha="left")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.04, hspace=0.45, wspace=0.08)
    fig.savefig(IMG / "readme_examples.png", dpi=80, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    program_figure()
    examples_figure()
    print("wrote", IMG / "readme_program.png", IMG / "readme_examples.png")
