"""Repair record: written inside the file and beside it, verified, re-run bit-exact."""

import shutil
import subprocess

import numpy as np
import pytest

from zumzum import record as R
from zumzum.audio import save

pytestmark = pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")

PROGRAM = "zumzum 0.1\nnode g = gain(db: -3)\nlayer main = g\ncheck p = peak(max_dbfs: -1)\n"


@pytest.fixture
def made(tmp_path):
    x = 0.3 * np.sin(2 * np.pi * 330 * np.arange(44100) / 44100)[None, :]
    save(tmp_path / "in.wav", x, 44100)
    (tmp_path / "p.zz").write_text(PROGRAM)
    return tmp_path


@pytest.mark.parametrize("ext", [".wav", ".mp3", ".flac", ".m4a"])
def test_record_goes_inside_and_beside_and_verifies(made, ext):
    from tools.convert import run
    from tools.verify import verify
    out = made / f"out_{ext[1:]}{ext}"
    r, *_ = run(made / "p.zz", made / "in.wav", out)
    text, where = R.read(out)
    assert "inside" in where and text == R.side_path(out).read_text()
    assert R.sha256_bytes(text.encode()) == r["record_sha256"]
    v = verify(out, made / "in.wav")
    assert v["pass"], v["steps"]
    assert any(s["text"].startswith("bit-exact") for s in v["steps"])


def test_a_changed_sample_or_source_fails_verify(made):
    import soundfile as sf
    from tools.convert import run
    from tools.verify import verify
    out = made / "o.wav"
    run(made / "p.zz", made / "in.wav", out)
    y, sr = sf.read(str(out), always_2d=True)
    y[100, 0] += 4 / 32768
    bad = made / "bad.wav"
    sf.write(str(bad), y, sr, subtype="PCM_16")
    shutil.copy(R.side_path(out), R.side_path(bad))
    steps = {s["step"]: s["pass"] for s in verify(bad)["steps"]}
    assert steps["record found"] and not steps["audio is the recorded audio"]
    other = made / "other.wav"
    save(other, 0.1 * np.ones((1, 44100)), 44100)
    steps = {s["step"]: s["pass"] for s in verify(out, other)["steps"]}
    assert not steps["source is the recorded source"]


def test_stripped_tags_fall_back_to_the_side_file(made):
    from tools.convert import run
    from tools.verify import verify
    out = made / "o.mp3"
    run(made / "p.zz", made / "in.wav", out)
    st = made / "stripped.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-map_metadata", "-1", "-c", "copy", str(st)], check=True)
    assert R.read(st) == (None, None)
    shutil.copy(R.side_path(out), R.side_path(st))
    text, where = R.read(st)
    assert where.startswith("side file") and verify(st)["pass"]
