"""Repair a recording to a named delivery rule set and prove it on the delivered file.

    python tools/fix.py --spec podcast-apple in.ogg out_dir/        # or audiobook-acx, ebu-r128, atsc-a85
    python tools/fix.py --list

Writes into out_dir/: <name>.<ext> (encoded as the rule set asks: MP3 for Apple and ACX, 24-bit WAV for broadcast),
<name>.<ext>.zzr.json (the repair record, also inside the file; check it with tools/verify.py),
<name>.zz (the program that ran: read it, re-run it), <name>.json (numbers before / after / delivered, every check,
the damage gate, the verdict per requirement) and <name>_sheet.png (spectrograms before / after and the checks).

The program is written from the input's own numbers, checked and rendered by the reference engine, its checks and
the damage gate measured; then the encoded file is decoded and measured again, because an encoder can push a true
peak up or take loudness off. If the delivered file misses (over a limit, or off the target by more than 0.1), the
program is rewritten from what was delivered (a lower true-peak ceiling, the target moved by the miss) and run again, at
most 3 versions. The level is one gain, capped by the true-peak ceiling; nothing is limited or compressed, so a file
whose peaks leave too little room is reported as needing limiting, which fix does not apply. A noise floor over a rule's limit is reported as a FAIL; fix does not remove noise. Every version is kept in the .json. Nothing else is touched: no denoise, no EQ, no compression.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zumzum.audio import load, save  # noqa: E402
from zumzum.checker import check  # noqa: E402
from zumzum.engine import render  # noqa: E402
from zumzum.measure import damage, run_checks  # noqa: E402
from zumzum.specs import SPECS, measure_all, program, verdict  # noqa: E402
from tools.convert import ffmpeg, local, probe  # noqa: E402


ENCODER_TOL = 0.1      # a delivered level further than this from the target gets another version (encoder loss)


def _jsonable(res):
    return [{k: (bool(v) if k == "pass" else (None if v is None else float(v)) if k == "value" else v)
             for k, v in r.items()} for r in res]


def _plan(name, delivered=None, last=None, reached=True):
    """(margin_db, shift_db) for the next version. First version: 0.5 dB margin, no shift. shift_db moves the level
    target by what the delivered file missed it by (an MP3 encode can take 1 LU off a bright recording; ACX's RMS
    depends on the file)."""
    s = SPECS[name]
    if delivered is None:
        return 0.5, 0.0
    margin, shift = last
    over = delivered["true_peak_dbtp"] - s["true_peak"]
    if over > -0.1:  # the encoder pushed it to (or over) the limit: make room for what it added
        margin += max(over, 0) + 0.3
    got = delivered["rms_dbfs"] if s["kind"] == "rms" else delivered["lufs"]
    miss = 0.0 if got is None else s["target"] - got
    # move the target only when the loudness node reached its own target (the miss came from the encoder, or for ACX
    # from the RMS / loudness relation): when the true-peak ceiling is what stopped the level, there is no more gain to give,
    # so nothing is shifted and the miss is reported
    if abs(miss) > 0.1 and reached:
        shift += miss
    return margin, shift


def _render(name, src_prog, x, sr):
    prog = check(src_prog, duration=x.shape[1] / sr, sr=sr)
    outs, info = render(prog, x, sr)
    y = outs[prog.output]
    return prog, y, info, run_checks(prog, x, outs, sr), damage(x, y, prog, sr)


def fix(name, src, out_dir, sheet=True, overwrite=False):
    s = SPECS[name]
    src, out_dir = Path(src), Path(out_dir)
    ext = s["encode"][0]
    stem = out_dir / src.stem
    dst = stem.with_suffix(ext)
    if not overwrite:
        for p in (dst, stem.with_suffix(".json"), stem.with_suffix(".zz")):
            if p.exists():
                raise SystemExit(f"{p} exists; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav_in, wav_out, back = Path(tmp) / "in.wav", Path(tmp) / "out.wav", Path(tmp) / "back.wav"
        most, sr0, ch0 = probe(src)
        ffmpeg(*local(src), "-map", "0:a:0", "-vn", "-t", f"{most:.3f}", "-ar", str(sr0), "-ac", str(ch0),
               "-c:a", "pcm_f32le", str(wav_in))
        x, sr = load(wav_in)
        before = measure_all(x, sr)
        versions, plan, delivered = [], _plan(name), None
        if before["lufs"] is None:
            raise SystemExit(f"{src}: no loudness to measure (silent, or under 0.4 s)")
        kept = {}  # version -> (encoded file copy, engine output) so the best version can be delivered
        for it in (1, 2, 3):
            text = program(name, before, plan[0], plan[1])
            prog, y, info, res, dmg = _render(name, text, x, sr)
            save(wav_out, y, sr)
            ffmpeg(*local(wav_out), *s["encode"][1:], str(dst))
            ffmpeg(*local(dst), "-c:a", "pcm_f32le", str(back))  # what the listener's player decodes
            d, dsr = load(back)
            delivered = measure_all(d, dsr)
            v = verdict(name, delivered)
            versions.append({"version": it, "program": text, "margin_db": round(plan[0], 2),
                             "shift_db": round(plan[1], 2), "info": info,
                             "checks": _jsonable(res), "damage": dmg, "after_engine": measure_all(y, sr),
                             "delivered": delivered, "verdict": v})
            copy = Path(tmp) / f"v{it}{ext}"
            shutil.copy(dst, copy)
            kept[it] = (copy, y)
            node = info.get("main.level", {})
            reached = node.get("lufs") is not None and abs(node["lufs"] - float(
                re.search(r"target_lufs: (-?[\d.]+)", text).group(1))) <= 0.2
            versions[-1]["level_reached"] = reached
            got = delivered["rms_dbfs"] if s["kind"] == "rms" else delivered["lufs"]
            off = got is not None and abs(got - s["target"]) > ENCODER_TOL and reached
            if all(v["rule"].values()) and all(c["pass"] for c in versions[-1]["checks"]) and not dmg and not off:
                break  # passes and sits within 0.1 of the target: nothing left to correct
            plan = _plan(name, delivered, plan, reached)
        # deliver the latest version that passes everything (a correction round must never make it worse)
        good = [w for w in versions if all(w["verdict"]["rule"].values())
                and all(c["pass"] for c in w["checks"]) and not w["damage"]]
        last = good[-1] if good else versions[-1]
        shutil.copy(kept[last["version"]][0], dst)
        y = kept[last["version"]][1]
    stem.with_suffix(".zz").write_text(last["program"], encoding="utf-8")
    short = last["info"].get("main.level", {}).get("short_by_lu") or 0.0
    warnings = [f"needs limiting to meet {name}: the peaks leave room for {last['info']['main.level']['gain_db']:+.1f} dB "
                f"of gain, {short:.1f} LU short of the target; not applied"] if short > ENCODER_TOL else []
    from zumzum import record
    rec = record.make(src, last["program"], last["checks"], last["damage"], dst, s["encode"][1:],
                      delivered=last["delivered"], nodes=last["info"], extra={"spec": {"name": name, "rule": s["source"],
                                                                   "verdict": last["verdict"]}})
    digest = record.attach(dst, rec)
    ok = all(last["verdict"]["rule"].values()) and all(c["pass"] for c in last["checks"]) and not last["damage"]
    result = {"spec": name, "rule": s["source"], "input": src.name, "output": dst.name, "sr": sr,
              "channels": int(x.shape[0]), "seconds": round(x.shape[1] / sr, 1), "before": before,
              "versions": versions, "delivered": last["delivered"], "delivered_version": last["version"],
              "warnings": warnings, "record_sha256": digest, "pass": bool(ok)}
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if sheet:
        from zumzum.sheets import sheet as draw
        title = f"{src.stem[:60]} -> {name} ({'PASS' if ok else 'FAIL'})" + (
            f"  needs limiting ({short:.1f} LU short)" if warnings else "")
        draw(stem.parent / (stem.name + "_sheet.png"), x, y, sr, title, last["checks"], str(stem.with_suffix(".zz").name),
             fmax=8000)
    return result


def main():
    ap = argparse.ArgumentParser(description="repair a recording to a delivery rule set and prove it")
    ap.add_argument("--spec", choices=sorted(SPECS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("input", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.list or not a.spec:
        for k, s in SPECS.items():
            print(f"{k:15} {s['source']}")
        return
    r = fix(a.spec, a.input, a.out_dir, sheet=not a.no_sheet, overwrite=a.overwrite)
    b, d = r["before"], r["delivered"]
    print(f"{r['input']} -> {r['output']}: {b['lufs']} LUFS / {b['true_peak_dbtp']} dBTP / RMS {b['rms_dbfs']} / "
          f"floor {b['noise_floor_dbfs']}  ->  {d['lufs']} LUFS / {d['true_peak_dbtp']} dBTP / RMS {d['rms_dbfs']} / "
          f"floor {d['noise_floor_dbfs']}  ({len(r['versions'])} version(s))  {'PASS' if r['pass'] else 'FAIL'}")
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
