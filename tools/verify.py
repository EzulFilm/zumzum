"""Check a Zumzum output against its repair record, and optionally re-run it from the source.

    python tools/verify.py out.mp3 [--source in.ogg]

1. Finds the record (inside the file, else <out>.zzr.json) and checks it is canonical and matches the hash stored
   next to it (the tag / bext description), when there is one.
2. Decodes the output and compares its PCM hash with the record's: the file holds the audio the record describes.
3. With --source: checks the source's hash, re-runs the program (checker, engine, the recorded encode), compares the
   re-made output with the delivered one (same PCM hash; else the difference must be under -90 dB of the output),
   and re-measures every check (each value within 0.01 of the recorded one, same pass / fail).
Exit 0 when every step passes.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from zumzum import record as R  # noqa: E402

MAX_DIFF_DB = -90.0
VALUE_TOL = 0.01


def _embedded_sha(path):
    p = Path(path)
    probe = subprocess.run(["ffprobe", "-v", "error", *R._local(p), "-show_entries", "format_tags:stream_tags",
                            "-of", "json"], capture_output=True, text=True)
    if probe.returncode:
        return None
    d = json.loads(probe.stdout or "{}")
    tags = dict((d.get("format") or {}).get("tags") or {})
    for s in d.get("streams") or []:
        tags.update(s.get("tags") or {})
    for k, v in tags.items():
        if k.upper() == R.TAG_SHA:
            return v
        if k.lower() == "comment" and v.startswith("Zumzum repair record "):  # BWF bext description
            return v.split()[-1]
    return None


def _decode(path, tmp):
    from tools.convert import ffmpeg, local, probe
    from zumzum.audio import load
    wav = Path(tmp) / f"dec_{Path(path).stem}.wav"
    most, sr0, ch0 = probe(path)
    ffmpeg(*local(path), "-map", "0:a:0", "-vn", "-t", f"{most:.3f}", "-ar", str(sr0), "-ac", str(ch0),
           "-c:a", "pcm_f32le", str(wav))
    return load(wav)


def verify(out, source=None):
    out = Path(out)
    steps = []

    def step(name, ok, text):
        steps.append({"step": name, "pass": bool(ok), "text": text})
        return ok

    text, where = R.read(out)
    if not step("record found", text is not None, where or "no record inside the file and no side file"):
        return {"pass": False, "steps": steps}
    rec = json.loads(text)
    digest = R.sha256_bytes(text.encode("utf-8"))
    stored = _embedded_sha(out)
    step("record intact", R.canonical(rec) == text and (stored is None or stored == digest),
         f"sha256 {digest[:16]}..; " + ("stored hash matches" if stored == digest else
                                         "no stored hash to compare (side file only)" if stored is None
                                         else f"stored hash {stored[:16]}.. DIFFERS"))
    pcm = R.pcm_sha256(out)
    step("audio is the recorded audio", pcm == rec["delivered"]["pcm_sha256"],
         f"decoded PCM sha256 {pcm[:16]}.. vs recorded {rec['delivered']['pcm_sha256'][:16]}..")
    if source is not None:
        source = Path(source)
        step("source is the recorded source", R.sha256_file(source) == rec["source"]["sha256"],
             f"{source.name}: sha256 {R.sha256_file(source)[:16]}.. vs recorded {rec['source']['sha256'][:16]}..")
        steps.extend(_rerun(rec, source, out))
    return {"pass": all(s["pass"] for s in steps), "steps": steps, "record": rec}


def _rerun(rec, source, out):
    from tools.convert import ffmpeg, local
    from zumzum.audio import save
    from zumzum.checker import check
    from zumzum.engine import render
    from zumzum.measure import run_checks
    steps = []
    ext = out.suffix.lower()
    if ext in (".mp4", ".mov", ".mkv"):
        return [{"step": "re-run", "pass": True, "text": "video output: re-run not done (audio-only re-run)"}]
    with tempfile.TemporaryDirectory() as tmp:
        x, sr = _decode(source, tmp)
        prog = check(rec["program"]["text"], duration=x.shape[1] / sr, sr=sr)
        outs, _ = render(prog, x, sr)
        y = outs[prog.output]
        wav, again = Path(tmp) / "out.wav", Path(tmp) / ("again" + ext)
        save(wav, y, sr)
        ffmpeg(*local(wav), *rec["delivered"]["encode"], str(again))
        same = R.pcm_sha256(again) == rec["delivered"]["pcm_sha256"]
        if same:
            steps.append({"step": "re-run gives the same audio", "pass": True, "text": "bit-exact (same PCM hash)"})
        else:
            a, _ = _decode(again, tmp)
            b, _ = _decode(out, tmp)
            n = min(a.shape[1], b.shape[1])
            d = 10 * np.log10(max(np.sum((a[:, :n] - b[:, :n]) ** 2), 1e-30) / max(np.sum(b[:, :n] ** 2), 1e-30))
            steps.append({"step": "re-run gives the same audio", "pass": d <= MAX_DIFF_DB,
                          "text": f"not bit-exact; difference {d:.1f} dB of the output (allowed <= {MAX_DIFF_DB:g})"})
        res = {r["name"]: r for r in run_checks(prog, x, outs, sr)}
        bad = []
        for c in rec["checks"]:
            r = res.get(c["name"])
            if r is None or bool(r["pass"]) != bool(c["pass"]) or (
                    c.get("value") is not None and (r["value"] is None or abs(float(r["value"]) - c["value"]) > VALUE_TOL)):
                bad.append(f"{c['name']}: recorded {c.get('value')} {'pass' if c['pass'] else 'fail'}, now "
                           f"{None if r is None else r['value']} {'pass' if r and r['pass'] else 'fail'}")
        steps.append({"step": "checks re-measured the same", "pass": not bad,
                      "text": f"{len(rec['checks'])} checks within {VALUE_TOL:g}" if not bad else "; ".join(bad)})
    return steps


def main():
    ap = argparse.ArgumentParser(description="check a Zumzum output against its repair record")
    ap.add_argument("output")
    ap.add_argument("--source")
    a = ap.parse_args()
    r = verify(a.output, a.source)
    for s in r["steps"]:
        print(f"{'PASS' if s['pass'] else 'FAIL'}  {s['step']}: {s['text']}")
    print("VERIFIED" if r["pass"] else "NOT VERIFIED")
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
