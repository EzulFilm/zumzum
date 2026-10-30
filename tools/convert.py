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
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zumzum.audio import check_shape, load, save  # noqa: E402
from zumzum.checker import check  # noqa: E402
from zumzum.engine import render  # noqa: E402
from zumzum.measure import damage, run_checks  # noqa: E402
from zumzum.ops import LIMITS  # noqa: E402

AUDIO = {".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"], ".m4a": ["-c:a", "aac", "-b:a", "192k"],
         ".flac": ["-c:a", "flac"], ".wav": ["-c:a", "pcm_s16le"]}
VIDEO = (".mp4", ".mov", ".mkv")
FORMATS = "wav,flac,mp3,ogg,mov,mp4,matroska,webm,aac"  # input demuxers ffmpeg may use


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def local(path):
    """ffmpeg input options for a local file only: no network or playlist protocols, and only the demuxers of the
    formats Zumzum converts (an .m3u8 or concat file could otherwise make ffmpeg fetch URLs or read other files)."""
    return ["-protocol_whitelist", "file", "-format_whitelist", FORMATS, "-i", "file:" + str(Path(path).resolve())]


def probe(src):
    """-> (the most seconds ffmpeg may decode, rate, channels) for the first audio stream (the one run() maps).
    Refuses an input whose header claims an absurd rate, channel count or length, or gives no usable rate or channel
    count, before ffmpeg decodes anything. A header without a duration is allowed: the returned limit bounds the
    decode by itself."""
    out = subprocess.run(["ffprobe", "-v", "error", *local(src),
                          "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels:format=duration",
                          "-of", "json"], check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    if not d.get("streams"):
        raise SystemExit(f"{src}: no audio stream")
    s = d["streams"][0]
    try:
        sr, ch = int(s["sample_rate"]), int(s["channels"])
        if ch < 1:
            raise ValueError(f"{ch} channels")
        dur = d.get("format", {}).get("duration")
        dur = float(dur) if dur not in (None, "", "N/A") else 0.0
        if not math.isfinite(dur) or dur < 0:
            raise ValueError(f"duration {dur} is not usable")
        check_shape(sr, ch, int(dur * sr))
    except (KeyError, ValueError, OverflowError) as e:
        raise SystemExit(f"{src}: refused ({e})") from None
    # one second over what the limits allow, so too long an input is refused by load(), never silently cut
    return min(LIMITS["duration_s"], LIMITS["samples"] / (ch * sr)) + 1, sr, ch


def run(program, src, dst, sheet_path=None, title=None):
    src, dst = Path(src), Path(dst)
    ext = dst.suffix.lower()
    if ext not in AUDIO and ext not in VIDEO:
        raise SystemExit(f"unknown output type {ext} (audio: {', '.join(AUDIO)}; video: {', '.join(VIDEO)})")
    with tempfile.TemporaryDirectory() as tmp:
        wav_in, wav_out = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        most, sr0, ch0 = probe(src)  # the decode is bounded by the limits even if the header lied about its length
        ffmpeg(*local(src), "-map", "0:a:0", "-vn", "-t", f"{most:.3f}", "-ar", str(sr0), "-ac", str(ch0),
               "-c:a", "pcm_f32le", str(wav_in))
        x, sr = load(wav_in)
        prog = check(Path(program).read_text(encoding="utf-8"), duration=x.shape[1] / sr, sr=sr)
        outs, info = render(prog, x, sr)
        y = outs[prog.output]
        res = run_checks(prog, x, outs, sr)
        dmg = damage(x, y)
        save(wav_out, y, sr)
        if ext in VIDEO:
            ffmpeg(*local(src), *local(wav_out), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                   "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(dst))
        else:
            ffmpeg(*local(wav_out), *AUDIO[ext], str(dst))
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
