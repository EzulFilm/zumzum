"""README figure docs/img/readme_fix_spec.png: the delivered files of tools/spec_run.py against each rule's targets.

    python tools/spec_run.py                 # 10 public recordings per rule set (results in out/spec/)
    python tools/make_delivery_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RULES = [("podcast-apple", "Apple Podcasts\n-16 LUFS", "lufs", -16, -1), ("audiobook-acx", "ACX audiobook\nRMS -20 dBFS", "rms_dbfs", -20, -3),
         ("ebu-r128", "EBU R 128\n-23 LUFS", "lufs", -23, -1), ("atsc-a85", "ATSC A/85\n-24 LKFS", "lufs", -24, -2)]


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9))
    rng = np.random.default_rng(0)
    done = total = 0
    for i, (name, _, key, target, tp) in enumerate(RULES):
        rows = [r for r in json.loads((ROOT / "out" / "spec" / f"{name}.json").read_text()) if "delivered" in r]
        total += len(rows)
        done += sum(r["pass"] for r in rows)
        x = i + rng.uniform(-0.18, 0.18, len(rows))
        a.scatter(x, [r["before"][key] - target for r in rows], s=22, color="#9aa0a6", label="before" if i == 0 else None)
        a.scatter(x, [r["delivered"][key] - target for r in rows], s=26, color="#4c78a8",
                  label="delivered file" if i == 0 else None)
        b.scatter(x, [r["delivered"]["true_peak_dbtp"] - tp for r in rows], s=26, color="#4c78a8")
    a.axhspan(-0.5, 0.5, color="#54a24b", alpha=0.18, label="within 0.5 of the target")
    a.set_ylabel("level minus the target (dB)")
    a.set_title("Level of each delivered file", loc="left", fontsize=10)
    b.axhline(0, color="#c33", lw=1)
    b.set_ylabel("true peak minus the limit (dB)")
    b.set_title("True peak of each delivered file (the red line is the limit)", loc="left", fontsize=10)
    for ax in (a, b):
        ax.set_xticks(range(4))
        ax.set_xticklabels([r[1] for r in RULES], fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
    a.legend(fontsize=8, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "img" / "readme_fix_spec.png", dpi=110)
    print(f"{done} of {total} delivered files meet their targets")


if __name__ == "__main__":
    main()
