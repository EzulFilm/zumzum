"""Voice clean-up preset: hum removal and a low cut only where its checks find steady hum or rumble, then one
loudness gain. WAV or FLAC in, WAV out, worked in blocks.

    python tools/clean_up.py input.wav output.wav [--target -16] [--ceiling -1]

Prints the choice (each candidate, whether it does something, how far over its budget, kept or not), the checks and
the damage gate. The exit code is 0 only when every check passes and the gate finds nothing.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zumzum.preset import clean_up  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--target", type=float, default=-16.0, help="loudness target, LUFS")
    ap.add_argument("--ceiling", type=float, default=-1.0, help="true-peak ceiling, dBTP")
    a = ap.parse_args()
    if Path(a.output).exists() or Path(a.output).resolve() == Path(a.input).resolve():
        print(f"refused: {a.output} exists (the input is never overwritten)")
        return 2
    r = clean_up(a.input, a.output, a.target, a.ceiling)
    print(r["program"])
    print(json.dumps(r["choice"], indent=1))
    for c in r["checks"]:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['name']}: {c['text']}")
    for d in r["damage"]:
        print(f"DAMAGE  {d}")
    return 0 if all(c["pass"] for c in r["checks"]) and not r["damage"] else 1


if __name__ == "__main__":
    sys.exit(main())
