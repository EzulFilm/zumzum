"""Download the example audio listed in examples/sources.json into cache/audio/, checking each file's SHA-256.

    python tools/fetch_audio.py          # download what is missing, refuse any file whose hash differs
    python tools/fetch_audio.py --pin    # record the hashes of files that have none yet (first download only)

Downloaded files are data: they are decoded by libsndfile (soundfile) and never executed.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "examples" / "sources.json"
CACHE = ROOT / "cache" / "audio"
UA = "Zumzum/0.1 (example audio fetch; mail@ezul.ai)"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path_of(key, entry):
    return CACHE / (key + Path(entry["url"].split("?")[0]).suffix)


def fetch(pin=False):
    data = json.loads(SOURCES.read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    changed = False
    for key, e in data["files"].items():
        p = path_of(key, e)
        if not p.exists():
            req = urllib.request.Request(e["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                p.write_bytes(r.read())
            print(f"downloaded {key} ({p.stat().st_size} bytes)")
        h = sha256(p)
        if not e["sha256"]:
            if not pin:
                raise SystemExit(f"{key}: no pinned SHA-256 (run with --pin once, then commit sources.json)")
            e["sha256"], changed = h, True
            print(f"pinned {key}: {h}")
        elif h != e["sha256"]:
            p.unlink()
            raise SystemExit(f"{key}: SHA-256 mismatch ({h} != {e['sha256']}); file removed")
    if changed:
        SOURCES.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {k: path_of(k, e) for k, e in data["files"].items()}


if __name__ == "__main__":
    for k, p in fetch(pin="--pin" in sys.argv).items():
        print(k, p.relative_to(ROOT))
