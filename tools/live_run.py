"""Run the editing loop on the five example inputs with a person's note each (the model never sees the example
programs). Writes out/llm/<driver>/<name>/ (every version's program, reply, checks; the final clip and sheet) and a
summary JSON; the committed record is docs/live_runs/.

    ZUMZUM_DRIVER=openweight ZUMZUM_LLM_URL=http://127.0.0.1:8091/v1 python tools/live_run.py [name ...]
    ZUMZUM_DRIVER=rules python tools/live_run.py
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from zumzum.audio import save  # noqa: E402
from zumzum.llm import edit_loop, make_provider  # noqa: E402
from zumzum.sheets import sheet  # noqa: E402
from tools.fetch_audio import fetch  # noqa: E402
from tools.make_examples import build  # noqa: E402

NOTES = {
    "hum_voice": "There is a mains hum under the voice. Remove it, keep the voice as it is.",
    "cylinder_denoise": "Less hiss please, keep the words clear.",
    "record_declick": "Take out the clicks and crackle, change nothing else in the music.",
    "phone_heal": "Someone dials a phone at about 0:09. Remove the dial tones, keep the voice.",
    "drums_down": "The drums are too loud. Bring the drums down about 12 dB and leave the piano alone.",
}


def main(names):
    driver = os.environ.get("ZUMZUM_DRIVER", "rules")
    provider = make_provider(driver)
    files = fetch()
    out = ROOT / "out" / "llm" / driver
    rows = []
    for name in names:
        x, sr, source, view = build(name, files)
        if np.max(np.abs(x)) > 0.95:
            x = x * 0.95 / np.max(np.abs(x))
        print(f"{name}: {NOTES[name]}", flush=True)
        t0 = time.time()
        r = edit_loop(NOTES[name], x, sr, provider, max_iters=3, log=lambda s: print(s, flush=True))
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for v in r["versions"]:
            (d / f"v{v['iter']}.zz").write_text(v["ran"], encoding="utf-8")
            (d / f"v{v['iter']}_reply.txt").write_text(v["reply"], encoding="utf-8")
        last = next((v for v in reversed(r["versions"]) if "output" in v), None)
        if last is not None:
            save(d / "before.wav", x, sr)
            save(d / "after.wav", last["output"], sr)
            sheet(d / "sheet.png", x, last["output"], sr, f"{name} - model loop ({driver})", last["checks"],
                  f"out/llm/{driver}/{name}/v{last['iter']}.zz", fmax=view["fmax"], zoom=view["zoom"])
        row = {"example": name, "note": NOTES[name], "done": r["done"], "iterations": len(r["versions"]),
               "seconds": round(time.time() - t0, 1),
               "versions": [{k: v.get(k) for k in ("iter", "program", "message", "problems", "checks", "damage",
                                                  "info", "usage", "diff", "done")} for v in r["versions"]]}
        (d / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        print(f"  -> {'DONE' if r['done'] else 'not done'} after {len(r['versions'])} version(s), {row['seconds']} s",
              flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or list(NOTES))
