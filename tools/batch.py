"""Batch mode: one rule set or one program over every audio file in a folder, one repair record per output.

    python tools/batch.py --spec audiobook-acx in_dir/ out_dir/ [--jobs 4]
    python tools/batch.py --program fix.zz --ext .wav in_dir/ out_dir/ [--jobs 4]

Writes out_dir/batch.json (one row per input: pass / fail, the output, its record hash, or the error) and
out_dir/batch.png (a summary sheet). A file that fails never stops the batch. A file whose existing output already
has a passing record for the same source bytes and the same rule set / program is skipped (re-running a batch only
does what changed). 
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUDIO_IN = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}


def inputs(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in AUDIO_IN)


def _done(src, out, key, value):
    """An existing passing record for these source bytes and this rule set / program."""
    from zumzum import record
    if not out.exists():
        return None
    text, _ = record.read(out)
    if text is None:
        return None
    rec = json.loads(text)
    same = rec["source"]["sha256"] == record.sha256_file(src)
    if key == "spec":
        same = same and (rec.get("spec") or {}).get("name") == value and all(
            (rec.get("spec") or {}).get("verdict", {}).get("test", {"x": False}).values())
    else:
        same = same and rec["program"]["sha256"] == record.sha256_bytes(Path(value).read_text("utf-8").encode())
    ok = same and all(c["pass"] for c in rec["checks"]) and not rec["damage"]
    return record.sha256_bytes(text.encode("utf-8")) if ok else None


def one(task):
    mode, value, ext, src, out_dir = task
    src, out_dir = Path(src), Path(out_dir)
    row = {"input": src.name}
    try:
        if mode == "spec":
            from zumzum.specs import SPECS
            out = out_dir / (src.stem + SPECS[value]["encode"][0])
        else:
            out = out_dir / (src.stem + ext)
        skip = _done(src, out, mode, value)
        if skip:
            return {**row, "output": out.name, "pass": True, "skipped": True, "record_sha256": skip}
        if mode == "spec":
            from tools.fix import fix
            r = fix(value, src, out_dir, overwrite=True)
            return {**row, "output": r["output"], "pass": r["pass"], "record_sha256": r["record_sha256"],
                    "delivered": r["delivered"]}
        from tools.convert import run
        r, *_ = run(value, src, out, out_dir / (src.stem + "_sheet.png"), overwrite=True)
        ok = all(c["pass"] for c in r["checks"]) and not r["damage"]
        return {**row, "output": out.name, "pass": ok, "record_sha256": r["record_sha256"],
                "failed": [c["name"] for c in r["checks"] if not c["pass"]] + [f"damage: {d}" for d in r["damage"]]}
    except BaseException as e:  # noqa: BLE001 - reported as a FAIL row; the batch goes on
        return {**row, "pass": False, "error": f"{type(e).__name__}: {e}"[:300]}


def batch(in_dir, out_dir, spec=None, program=None, ext=".wav", jobs=4):
    if (spec is None) == (program is None):
        raise ValueError("give exactly one of spec or program")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode, value = ("spec", spec) if spec else ("program", str(Path(program).resolve()))
    tasks = [(mode, value, ext, str(p), str(out_dir)) for p in inputs(in_dir)]
    if jobs <= 1:
        rows = [one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(jobs) as ex:
            rows = list(ex.map(one, tasks))
    summary = {"mode": mode, "value": spec or Path(program).name, "files": len(rows),
               "pass": sum(r["pass"] for r in rows), "rows": rows}
    (out_dir / "batch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sheet(summary, out_dir / "batch.png")
    return summary


def sheet(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = summary["rows"]
    cells = [[r["input"][:48], r.get("output", "")[:40], "skipped (record matches)" if r.get("skipped") else
              (r.get("error") or ", ".join(r.get("failed", [])) or "")[:60],
              (r.get("record_sha256") or "")[:16], "PASS" if r["pass"] else "FAIL"] for r in rows]
    fig, ax = plt.subplots(figsize=(15, 1.4 + 0.3 * max(1, len(rows))))
    ax.axis("off")
    fig.suptitle(f"Zumzum batch ({summary['mode']} {summary['value']}): {summary['pass']} / {summary['files']} PASS",
                 x=0.01, ha="left", fontsize=12)
    if cells:
        t = ax.table(cellText=cells, colLabels=["input", "output", "note", "record sha256", "verdict"],
                     loc="upper left", cellLoc="left", colWidths=[.3, .25, .27, .1, .06])
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        for (i, j), c in t.get_celld().items():
            if i == 0:
                c.set_facecolor("#dde3ea")
            elif j == 4:
                c.set_facecolor("#d8efd8" if rows[i - 1]["pass"] else "#f6d9d2")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="a rule set or a program over every audio file in a folder")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec")
    g.add_argument("--program")
    ap.add_argument("--ext", default=".wav", help="output type for --program (.wav .flac .mp3 .m4a)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    s = batch(a.in_dir, a.out_dir, a.spec, a.program, a.ext, a.jobs)
    print(f"{s['pass']} / {s['files']} PASS; {Path(a.out_dir) / 'batch.json'}")
    sys.exit(0 if s["pass"] == s["files"] else 1)


if __name__ == "__main__":
    main()
