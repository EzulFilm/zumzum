"""Run a simple Zumzum program from WAV / FLAC file to file in blocks: faster, and with little memory.

    python tools/fast.py program.zz input.wav output.wav

For programs with one layer of dehum and gain nodes (gain on a band selection or none) and at most one loudness node
at the end; anything else is refused with the reason (use tools/convert.py). The output equals the reference
engine's to far below anything audible. The program's checks (loudness, true_peak, loudness_range, rms, hum_drop,
mains_hum_drop) and the damage gate are measured from the files, also in blocks; for other checks, the sheet and the
repair record, use tools/convert.py.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from zumzum.checker import check  # noqa: E402
from zumzum.errors import CheckError  # noqa: E402
from zumzum.stream import Unsupported, run  # noqa: E402
from zumzum.stream_checks import file_checks  # noqa: E402


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    prog_path, src, dst = argv
    if Path(dst).exists() or Path(dst).resolve() == Path(src).resolve():
        print(f"refused: {dst} exists (the input is never overwritten)")
        return 2
    i = sf.info(src)
    try:
        prog = check(Path(prog_path).read_text(encoding="utf-8"), duration=i.frames / i.samplerate, sr=i.samplerate)
        info = run(prog, src, dst, subtype="FLOAT" if i.subtype in ("FLOAT", "DOUBLE") else i.subtype)
    except CheckError as e:
        print(f"program refused: {e}")
        return 1
    except Unsupported as e:
        print(f"not a program for the fast path ({e}); use tools/convert.py")
        return 1
    print(json.dumps(info, indent=1))
    try:
        checks, problems = file_checks(prog, src, dst)
    except Unsupported as e:
        print(f"checks not measured here ({e}); use tools/convert.py")
        return 0
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['name']}: {c['text']}")
    for d in problems:
        print(f"DAMAGE  {d}")
    return 0 if all(c["pass"] for c in checks) and not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
