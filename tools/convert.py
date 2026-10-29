"""Run a Zumzum program on a media file: mp3 in, mp3 out (or wav, flac, m4a; a video keeps its picture untouched).

    python tools/convert.py program.zz input.mp3 output.mp3 [--sheet sheet.png]

ffmpeg (on PATH, not bundled) decodes the sound to 32-bit float WAV at its own rate and channels. Zumzum checks the
program against that sound's length and rate, renders it, and measures its checks and the damage gate. ffmpeg then
encodes the output by its extension: .mp3 as MP3 (LAME, 192 kb/s), .m4a as AAC 192 kb/s, .flac as FLAC, .wav as
16-bit PCM; for .mp4 / .mov / .mkv the input's video stream is copied bit for bit and the sound is AAC.
Results go to <output>.json.
"""

import argparse
import json
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

AUDIO = {".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"], ".m4a": ["-c:a", "aac", "-b:a", "192k"],
         ".flac": ["-c:a", "flac"], ".wav": ["-c:a", "pcm_s16le"]}
VIDEO = (".mp4", ".mov", ".mkv")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def run(program, src, dst, sheet_path=None, title=None):
    src, dst = Path(src), Path(dst)
    ext = dst.suffix.lower()
    if ext not in AUDIO and ext not in VIDEO:
        raise SystemExit(f"unknown output type {ext} (audio: {', '.join(AUDIO)}; video: {', '.join(VIDEO)})")
    with tempfile.TemporaryDirectory() as tmp:
        wav_in, wav_out = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        ffmpeg("-i", str(src), "-vn", "-c:a", "pcm_f32le", str(wav_in))
        x, sr = load(wav_in)
        prog = check(Path(program).read_text(encoding="utf-8"), duration=x.shape[1] / sr, sr=sr)
        outs, info = render(prog, x, sr)
        y = outs[prog.output]
        res = run_checks(prog, x, outs, sr)
        dmg = damage(x, y)
        save(wav_out, y, sr)
        if ext in VIDEO:
            ffmpeg("-i", str(src), "-i", str(wav_out), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                   "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(dst))
        else:
            ffmpeg("-i", str(wav_out), *AUDIO[ext], str(dst))
    if sheet_path:
        from zumzum.sheets import sheet
        sheet(sheet_path, x, y, sr, title or prog.name, res, str(program), fmax=8000)
    result = {"program": str(program), "input": src.name, "output": dst.name, "sr": sr, "channels": x.shape[0],
              "checks": [{k: (bool(v) if k == "pass" else float(v) if k == "value" else v) for k, v in r.items()}
                         for r in res], "damage": dmg, "info": info}
    dst.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, x, y, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--sheet")
    a = ap.parse_args()
    r, *_ = run(a.program, a.input, a.output, a.sheet)
    for c in r["checks"]:
        print(f"{'PASS' if c['pass'] else 'FAIL'} {c['name']}: {c['text']}")
    for d in r["damage"]:
        print(f"DAMAGE {d}")
    sys.exit(0 if all(c["pass"] for c in r["checks"]) and not r["damage"] else 1)


if __name__ == "__main__":
    main()
