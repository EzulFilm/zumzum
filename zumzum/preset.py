"""A voice clean-up preset that chooses hum removal and a low cut only where its checks find steady hum or rumble.
File in, file out, in blocks.

Candidates: dehum at 50 Hz, dehum at 60 Hz, a low cut under 75 Hz (gain -30 dB, 15 Hz feather). Each is measured on
the input, as the damage gate's change budget measures it (stream_checks.budget_numbers). At most one mains series is
kept: the one holding more steady hum, and only within PRESET_DEHUM_DB of its budget. The low cut is measured with that
dehum applied and kept when it removes at least PRESET_CUT_SHARE of the energy within PRESET_CUT_DB of its budget.
Then one loudness gain to the target. A recording where the checks find neither steady hum nor rumble gets only the
loudness gain. The constants are fixed; the preset only keeps or leaves out these nodes.
"""

import math
import tempfile

from .checker import check
from .measure import BUDGET_MIN_SHARE
from .stream import run
from .stream_checks import budget_numbers, file_checks

CANDIDATES = {"h50": "node h50 = dehum(f0_hz: 50, harmonics: 16, depth_db: 50)",
              "h60": "node h60 = dehum(f0_hz: 60, harmonics: 16, depth_db: 50)",
              "r": "node r = gain(db: -30) at rule_low"}
LOW = "sel rule_low = band(f_hz: 0..75, feather_hz: 15)"


def _program(nodes, target_lufs, ceiling_dbtp, checks=True):
    lines = ["zumzum 0.1"] + ([LOW] if "r" in nodes else []) + [CANDIDATES[k] for k in nodes]
    lines.append(f"node lift = loudness(target_lufs: {target_lufs:g}, max_true_peak_dbtp: {ceiling_dbtp:g})")
    lines.append("layer main = " + " -> ".join([*nodes, "lift"]))
    if checks:
        lines += [f"check level = loudness(target_lufs: {target_lufs:g}, tol_lu: 1)",
                  f"check peaks = true_peak(max_dbtp: {ceiling_dbtp:g})"]
    return "\n".join(lines) + "\n"


PRESET_DEHUM_DB = 3.0   # the preset keeps a dehum only this far over its budget at most (the gate allows 6: it must not
#                         refuse a repair that helps; the preset's job is never to make a file worse)
PRESET_CUT_DB = 4.0     # and a low cut only this far over
PRESET_CUT_SHARE = 0.01  # and only when the cut takes at least 1 % of the energy (after the dehum: what is left of a
#                          hum's own 50 / 60 Hz line is steady, but small)


def _over(r):
    return None if r["steady"] <= 0 else 10 * math.log10(max(r["removed"], 1e-30) / r["steady"])


def _row(name, r, margin, min_share):
    does = r["total"] > 0 and r["removed"] >= min_share * r["total"]
    over = _over(r)
    within = over is not None and over <= margin
    return {"node": name, "does_something": bool(does), "within_budget": bool(within), "kept": bool(does and within),
            "over_db": None if over is None else round(over, 1), "share_pct": round(100 * r["removed"] / max(r["total"], 1e-30), 3)}


def choose(src, sr, duration, target_lufs=-16.0, ceiling_dbtp=-1.0):
    """-> (program text, report). One mains series at most: the one holding more steady hum, kept when within
    PRESET_DEHUM_DB of its budget. Then the low cut, measured on the input with that dehum applied, kept when it takes
    at least PRESET_CUT_SHARE and is within PRESET_CUT_DB."""
    probe = check(_program(["h50", "h60"], target_lufs, ceiling_dbtp, checks=False), duration=duration, sr=sr)
    h = budget_numbers(probe, src)
    best = 0 if h[0]["steady"] >= h[1]["steady"] else 1
    report = []
    for i, name in enumerate(("h50", "h60")):
        row = _row(name, h[i], PRESET_DEHUM_DB, BUDGET_MIN_SHARE)
        if i != best:
            row["kept"] = False
            row["note"] = "the other series holds more steady hum"
        report.append(row)
    keep = [r["node"] for r in report if r["kept"]]
    low_src = src
    with tempfile.TemporaryDirectory() as tmp:
        if keep:
            low_src = f"{tmp}/dehummed.wav"
            text = "zumzum 0.1\n" + "\n".join(CANDIDATES[k] for k in keep) + "\nlayer main = " + " -> ".join(keep) + "\n"
            run(check(text, duration=duration, sr=sr), src, low_src, subtype="FLOAT")
        rp = check(_program(["r"], target_lufs, ceiling_dbtp, checks=False), duration=duration, sr=sr)
        low = _row("r", budget_numbers(rp, low_src)[0], PRESET_CUT_DB, PRESET_CUT_SHARE)
        report.append(low)
        if low["kept"]:
            keep.append("r")
            if not any(k.startswith("h") for k in keep):
                # rumble also sits in the notched bins at 50 / 100 Hz and inflates a dehum's removal: judge the
                # series again with the low cut applied (hum and rumble together)
                cut_src = f"{tmp}/cut.wav"
                text = "zumzum 0.1\n" + LOW + "\n" + CANDIDATES["r"] + "\nlayer main = r\n"
                run(check(text, duration=duration, sr=sr), src, cut_src, subtype="FLOAT")
                h2 = budget_numbers(probe, cut_src)
                name = ("h50", "h60")[best]
                again = _row(name, h2[best], PRESET_DEHUM_DB, BUDGET_MIN_SHARE)
                again["note"] = "judged again after the low cut"
                report.append(again)
                if again["kept"]:
                    keep.insert(0, name)
    return _program(keep, target_lufs, ceiling_dbtp), report


def clean_up(src, dst, target_lufs=-16.0, ceiling_dbtp=-1.0):
    """Choose, run in blocks, check in blocks. -> {"program", "choice", "info", "checks", "damage"}."""
    import soundfile as sf
    i = sf.info(str(src))
    text, report = choose(src, i.samplerate, i.frames / i.samplerate, target_lufs, ceiling_dbtp)
    prog = check(text, duration=i.frames / i.samplerate, sr=i.samplerate)
    info = run(prog, src, dst)
    checks, problems = file_checks(prog, src, dst)
    return {"program": text, "choice": report, "info": info, "checks": checks, "damage": problems}
