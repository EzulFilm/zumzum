"""Test: `fix --spec` on 10 public recordings per rule set; a sheet per file and a summary sheet per rule.

    python tools/spec_run.py [--jobs 8] [spec ...]     # results in out/spec/<spec>/, summary out/spec/<spec>.png

PASS per file: the delivered file meets the rule set's published numbers (level within its tolerance, true peak at or
under its limit, ACX's noise floor), every check in the program passes and the damage gate is clean.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = {"podcast-apple": "podcast", "audiobook-acx": "audiobook", "ebu-r128": "broadcast", "atsc-a85": "broadcast"}
OUT = ROOT / "out" / "spec"


def one(args):
    spec, path = args
    from tools.fix import fix
    try:
        return fix(spec, path, OUT / spec, overwrite=True)
    except BaseException as e:  # noqa: BLE001 - one bad file must not stop the other 39; it is reported as a FAIL
        return {"spec": spec, "input": Path(path).name, "pass": False, "error": f"{type(e).__name__}: {e}"}


def summary_sheet(spec, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from zumzum.specs import SPECS
    s = SPECS[spec]
    rms = s["kind"] == "rms"
    cols = ["recording", "s", "before: LUFS / TP / RMS / floor", "delivered: " + ("RMS" if rms else "LUFS"),
            "TP dBTP", "noise floor" if rms else "LRA LU", "short by LU", "checks", "damage", "versions", "verdict"]
    cells, colours = [], []
    for r in rows:
        if "error" in r:
            cells.append([r["input"][:44], "", r["error"][:60], "", "", "", "", "", "", "", "FAIL"])
            colours.append("#f6d9d2")
            continue
        b, d, last = r["before"], r["delivered"], r["versions"][-1]
        fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
        cells.append([r["input"][:44], f"{r['seconds']:.0f}",
                      f"{fmt(b['lufs'])} / {fmt(b['true_peak_dbtp'])} / {fmt(b['rms_dbfs'])} / {fmt(b['noise_floor_dbfs'])}",
                      fmt(d["rms_dbfs"] if rms else d["lufs"]), fmt(d["true_peak_dbtp"]),
                      fmt(d["noise_floor_dbfs"] if rms else d["lra_lu"]),
                      fmt(last["info"].get("main.level", {}).get("short_by_lu")),
                      f"{sum(c['pass'] for c in last['checks'])}/{len(last['checks'])}",
                      "clean" if not last["damage"] else "DAMAGE", str(len(r["versions"])), "PASS" if r["pass"] else "FAIL"])
        colours.append("#d8efd8" if r["pass"] else "#f6d9d2")
    n_pass = sum(r["pass"] for r in rows)
    fig, ax = plt.subplots(figsize=(17, 1.6 + 0.42 * len(rows)))
    ax.axis("off")
    want = (f"RMS {s['range'][0]:g}..{s['range'][1]:g} dBFS, noise floor <= {s['noise_floor']:g} dBFS" if rms
            else f"{s['target']:g} LUFS +-{s['tol']:g}")
    fig.suptitle(f"Zumzum fix --spec {spec}: {n_pass} / {len(rows)} meet the targets   ({s['title']}: {want}, true peak <= "
                 f"{s['true_peak']:g} dBTP, measured on the delivered {s['encode'][0][1:].upper()})",
                 x=0.01, ha="left", fontsize=12)
    t = ax.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="left",
                 colWidths=[.21, .04, .19, .08, .06, .07, .06, .05, .05, .08, .06])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dde3ea")
            cell.set_text_props(weight="bold")
        elif c == len(cols) - 1:
            cell.set_facecolor(colours[r - 1])
        elif c == 6 and "warnings" in rows[r - 1] and rows[r - 1]["warnings"]:
            cell.set_facecolor("#f7e3b5")  # the peaks left too little room: needs limiting
    ax.text(0, -0.02, "Amber 'short by LU': the peaks left too little room for the gain; the file needs limiting, which "
            f"fix does not apply. Targets: {s['source']}. Every delivered file is decoded and measured again.",
            transform=ax.transAxes, fontsize=8, va="top")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(specs, jobs=8):
    from tools.spec_data import fetch
    files, _ = fetch()
    tasks = [(sp, p) for sp in specs for p in files[RUNS[sp]].values()]
    with ProcessPoolExecutor(jobs) as ex:
        results = list(ex.map(one, tasks))
    for sp in specs:
        rows = sorted([r for r in results if r["spec"] == sp], key=lambda r: r["input"])
        (OUT / f"{sp}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        summary_sheet(sp, rows, OUT / f"{sp}.png")
        print(f"{sp}: {sum(r['pass'] for r in rows)} / {len(rows)} PASS")
        for r in rows:
            if not r["pass"]:
                print("   FAIL", r["input"], r.get("error") or r.get("warnings") or r["versions"][-1]["verdict"]["rule"],
                      r.get("versions", [{}])[-1].get("damage"))


if __name__ == "__main__":
    args = sys.argv[1:]
    jobs = 8
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        del args[i:i + 2]
    main(args or list(RUNS), jobs)
