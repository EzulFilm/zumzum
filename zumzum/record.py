"""Repair record (docs/spec.md section 11): what was done to a file, carried inside it and beside it.

The record is one JSON object in canonical form (UTF-8, sorted keys, no spaces), so it has one hash. It goes into the
output file (WAV: a `bext` chunk naming the program hash plus a private RIFF chunk `zzrc` holding the record; MP3:
ID3v2 TXXX; FLAC / Ogg: Vorbis comments; M4A / MP4: a metadata item) and always into the side file
`<output>.zzr.json`, because hosts and players strip tags. Embedding re-muxes with `-c copy`: the audio is not
re-encoded, and `pcm_sha256` (a hash of the decoded 32-bit float samples) stays the same.
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

VERSION = 1
CHUNK = b"zzrc"
TAG = "ZUMZUM_RECORD"
TAG_SHA = "ZUMZUM_RECORD_SHA256"
ROOT = Path(__file__).resolve().parents[1]


def canonical(rec):
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _local(path):
    from tools.convert import local
    return local(path)


def pcm_sha256(path):
    """SHA-256 of the first audio stream decoded to 32-bit float, interleaved, at its own rate and channels."""
    p = subprocess.run(["ffmpeg", "-v", "error", *_local(path), "-map", "0:a:0", "-f", "f32le", "-c:a", "pcm_f32le",
                        "-"], check=True, capture_output=True)
    return sha256_bytes(p.stdout)


def tool_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _plain(v):
    """JSON-safe copy (numpy numbers to Python numbers)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "item"):
        return v.item()
    return v


def make(source, program_text, checks, damage, delivered_path, encode, delivered=None, models=(), extra=None,
         nodes=None):
    """The record for one output. `encode` is the ffmpeg codec arguments the output was made with (a re-run uses
    them); `delivered` = numbers measured on the delivered file."""
    source = Path(source)
    rec = {"zumzum_record": VERSION,
           "tool": {"name": "zumzum", "version": "0.1", "commit": tool_commit()},
           "source": {"name": source.name, "sha256": sha256_file(source), "pcm_sha256": pcm_sha256(source)},
           "program": {"text": program_text, "sha256": sha256_bytes(program_text.encode("utf-8"))},
           "models": list(models),
           "checks": [{k: c[k] for k in ("name", "check", "pass", "value", "text") if k in c} for c in checks],
           "damage": list(damage),
           "nodes": _plain(nodes or {}),  # what each node measured
           "delivered": {"name": Path(delivered_path).name, "encode": list(encode),
                         "pcm_sha256": pcm_sha256(delivered_path), **(delivered or {})},
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if extra:
        rec.update(extra)
    return rec


def side_path(path):
    return Path(str(path) + ".zzr.json")


def attach(path, rec):
    """Write the record into the file (re-mux, audio untouched) and beside it. Returns the record's SHA-256."""
    path = Path(path)
    text = canonical(rec)
    digest = sha256_bytes(text.encode("utf-8"))
    side_path(path).write_text(text, encoding="utf-8")
    ext = path.suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / ("x" + ext)
        meta = []
        if ext == ".wav":
            history = f"T=zumzum 0.1 program={rec['program']['sha256'][:16]} record={digest[:16]}"
            meta = ["-write_bext", "1", "-metadata", f"description=Zumzum repair record {digest}",
                    "-metadata", f"coding_history={history}", "-metadata", "originator=Zumzum"]
        else:
            meta = ["-metadata", f"{TAG}={text}", "-metadata", f"{TAG_SHA}={digest}"]
            if ext in (".m4a", ".mp4", ".mov"):
                meta += ["-movflags", "use_metadata_tags"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *_local(path), "-map", "0", "-c", "copy", *meta, str(out)],
                       check=True)
        if ext == ".wav":
            _append_chunk(out, CHUNK, text.encode("utf-8"))
        shutil.move(str(out), str(path))
    return digest


def _append_chunk(path, cid, data):
    """Append one RIFF chunk (padded to an even length) and fix the RIFF size. RF64 files are refused (> 4 GB)."""
    with open(path, "r+b") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{path}: not a RIFF / WAVE file (RF64 is not supported for the record chunk)")
        f.seek(0, os.SEEK_END)
        f.write(cid + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b""))
        size = f.tell() - 8
        f.seek(4)
        f.write(struct.pack("<I", size))


def _read_chunk(path, cid):
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return None
        while True:
            h = f.read(8)
            if len(h) < 8:
                return None
            n = struct.unpack("<I", h[4:])[0]
            if h[:4] == cid:
                return f.read(n)
            f.seek(n + (n % 2), os.SEEK_CUR)


def read(path):
    """-> (record text, where) from the file itself, else the side file; (None, None) when neither has one."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        data = _read_chunk(path, CHUNK)
        if data is not None:
            return data.decode("utf-8"), "inside the file (RIFF chunk zzrc)"
    else:
        p = subprocess.run(["ffprobe", "-v", "error", *_local(path), "-show_entries", "format_tags:stream_tags",
                            "-of", "json"], capture_output=True, text=True)
        if p.returncode == 0:
            d = json.loads(p.stdout or "{}")
            tags = dict((d.get("format") or {}).get("tags") or {})
            for s in d.get("streams") or []:
                tags.update(s.get("tags") or {})
            for k, v in tags.items():
                if k.upper() == TAG:
                    return v, "inside the file (tag ZUMZUM_RECORD)"
    side = side_path(path)
    if side.exists():
        return side.read_text(encoding="utf-8"), "side file " + side.name
    return None, None
