"""Build the five examples: input audio (a public-domain / CC0 recording, with a documented, seeded damage added where
the recording has none), run the example program through the checker and the reference engine, measure its checks
and the damage gate, and write clips, a before / after / removed spectrogram sheet and results.json per example.

    python tools/make_examples.py [name ...]     # outputs in out/examples/<name>/
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zumzum.analysis import analyze, summary  # noqa: E402
from zumzum.audio import load, save  # noqa: E402
from zumzum.checker import check  # noqa: E402
from zumzum.engine import render  # noqa: E402
from zumzum.measure import damage, run_checks  # noqa: E402
from zumzum.sheets import sheet  # noqa: E402
from tools.fetch_audio import fetch  # noqa: E402

OUT = ROOT / "out" / "examples"


def hum(n, sr, f0=60.0, harmonics=16, level_dbfs=-32, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    a = 10 ** (level_dbfs / 20)
    return sum(a / k ** 0.7 * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
               for k in range(1, harmonics + 1))


def clicks(x, sr, count, seed=1):
    """Short decaying impulses of random sign, width 0.05-0.6 ms and amplitude 0.05-0.5, at random places (seeded)."""
    rng = np.random.default_rng(seed)
    y = x.copy()
    for _ in range(count):
        w = int(rng.uniform(0.05, 0.6) / 1000 * sr) + 2
        p = rng.integers(w, x.shape[1] - w)
        shape = rng.choice([-1, 1]) * rng.uniform(0.05, 0.5) * np.exp(-np.arange(w) / (w / 4))
        y[:, p:p + w] += shape
    return y


def build(name, files):
    if name == "hum_voice":
        x, sr = load(files["speech_whitman"], 0, 20, mono=True)
        return x + hum(x.shape[1], sr), sr, ("LibriVox 'O Captain! My Captain!' (public domain), 0-20 s, mono; "
                                             "added: 60 Hz hum, 16 harmonics, fundamental -32 dBFS (seed 0)"), \
            {"fmax": 4000, "zoom": (0, 20, 0, 1000)}
    if name == "cylinder_denoise":
        x, sr = load(files["cylinder_1888"], 0, 30, mono=True)
        return x, sr, "Edison wax cylinder, Arthur Sullivan's speech, 1888 (public domain), 0-30 s; nothing added", \
            {"fmax": 6000, "zoom": None}
    if name == "record_declick":
        x, sr = load(files["jazz_1917"], 20, 30)
        return clicks(x, sr, 150), sr, ("Original Dixieland Jass Band, Livery Stable Blues, 1917 78 rpm (public "
                                        "domain), 20-50 s; added: 150 clicks 0.05-0.6 ms (seed 1) on top of the "
                                        "record's own"), {"fmax": 8000, "zoom": None}
    if name == "phone_heal":
        x, sr = load(files["speech_whitman"], 0, 20, mono=True)
        tel, sr2 = load(files["telephone"], 4.9, 3.5, mono=True)
        assert sr2 == sr
        i = int(9.0 * sr)
        clean = x.copy()
        x[:, i:i + tel.shape[1]] += 4.0 * tel
        return x, sr, ("LibriVox 'O Captain! My Captain!' (public domain), 0-20 s, mono; added at 9.0 s: touch-tone "
                       "dialling 4.9-8.4 s of Telephone.ogg (public domain), +12 dB"), \
            {"fmax": 4000, "zoom": (8.5, 13, 500, 1700), "added": x - clean}
    if name == "drums_down":
        p, sr = load(files["piano_bach"], 0, 20)
        d, sr2 = load(files["drum_break"], 0, 20)
        assert sr2 == sr
        drums = 10 ** (-8 / 20) * d
        return p + drums, sr, ("Bach Goldberg Aria, Musopen (CC0), 0-20 s + '130 old drum break' by Bauchamp (CC0), "
                               "0-20 s, at -8 dB"), {"fmax": 12000, "zoom": None, "truth": (p, drums)}
    raise KeyError(name)


NAMES = ["hum_voice", "cylinder_denoise", "record_declick", "phone_heal", "drums_down"]


def main(names):
    files = fetch()
    summary_rows = []
    for name in names:
        t0 = time.time()
        x, sr, source, view = build(name, files)
        if np.max(np.abs(x)) > 0.95:  # the added damage may push the peak over full scale: scale to -0.45 dBFS
            g = 0.95 / np.max(np.abs(x))
            x = x * g
            source += f"; scaled by {20 * np.log10(g):.2f} dB to keep the peak under full scale"
            if "truth" in view:
                view["truth"] = tuple(t * g for t in view["truth"])
        src = (ROOT / "examples" / f"{name}.zz").read_text(encoding="utf-8")
        prog = check(src, duration=x.shape[1] / sr, sr=sr)
        outs, info = render(prog, x, sr)
        y = outs[prog.output]
        res = run_checks(prog, x, outs, sr)
        dmg = damage(x, y)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        save(d / "before.wav", x, sr)
        save(d / "after.wav", y, sr)
        save(d / "removed.wav", x - y, sr)
        extra = {}
        if "truth" in view:  # the mix is synthetic, so we know what the result should be
            piano, drums = view["truth"]
            want = piano + 10 ** (-12 / 20) * drums
            extra["error_vs_ideal_db"] = round(10 * np.log10(np.mean((y - want) ** 2) / np.mean(want ** 2)), 2)
            extra["error_input_vs_ideal_db"] = round(10 * np.log10(np.mean((x - want) ** 2) / np.mean(want ** 2)), 2)
        if "added" in view:  # how much of what we added is still there, inside the program's selection
            from zumzum.engine import sel_mask
            from zumzum.stft import bin_freqs, frame_times, stft
            sname = next(iter(prog.sels))
            def boxed(a):
                X = stft(a[0], prog.size, prog.hop)
                M = sel_mask(prog.sels[sname], bin_freqs(prog.size, sr), frame_times(X.shape[1], prog.size, prog.hop, sr))
                return float((np.abs(X) ** 2 * M).sum())
            clean = x - view["added"]
            extra["added_sound_left_db"] = round(10 * np.log10(boxed(y - clean) / boxed(view["added"])), 2)
        sheet(d / "sheet.png", x, y, sr, f"{prog.name} - {source}" if len(source) < 90 else prog.name, res,
              f"examples/{name}.zz", fmax=view["fmax"], zoom=view["zoom"])
        result = {"example": name, "program": f"examples/{name}.zz", "source": source, "sr": sr,
                  "checks": [{k: (bool(v) if k == "pass" else float(v) if k == "value" else v)
                              for k, v in r.items()} for r in res],
                  "damage": dmg, "info": info, "analysis_before": summary(analyze(x, sr)),
                  "analysis_after": summary(analyze(y, sr)), "seconds": round(time.time() - t0, 1), **extra}
        (d / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        ok = all(r["pass"] for r in res) and not dmg
        summary_rows.append((name, ok, res, dmg, extra))
        print(f"{name}: {'ALL PASS' if ok else 'NOT PASSING'} ({result['seconds']} s)")
        for r in res:
            print(f"   {'PASS' if r['pass'] else 'FAIL'} {r['name']}: {r['text']}")
        for m in dmg:
            print(f"   DAMAGE {m}")
        if extra:
            print("  ", extra)
    return summary_rows


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
