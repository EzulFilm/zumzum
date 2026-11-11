"""The fast path (zumzum/stream.py) against the reference engine on short synthetic files."""

import numpy as np
import pytest
import soundfile as sf

from zumzum import loudness
from zumzum.checker import check
from zumzum.engine import render
from zumzum.stream import Meter, Unsupported, plan, run

SR = 48000
CHAIN = """zumzum 0.1
sel low = band(f_hz: 0..75, feather_hz: 15)
node h50 = dehum(f0_hz: 50, harmonics: 16, depth_db: 50)
node h60 = dehum(f0_hz: 60, harmonics: 16, depth_db: 50)
node r = gain(db: -30) at low
layer main = h50 -> h60 -> r
"""


def _voice_like(seconds=7, channels=1, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    x = []
    for c in range(channels):
        v = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / k for k, f in enumerate((180, 367, 742, 1510, 2890), 1))
        v = v * (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * (1.3 + c) * t))) * 0.1 + 0.003 * rng.standard_normal(len(t))
        v += 0.02 * sum(np.sin(2 * np.pi * 50.07 * k * t) / k for k in range(1, 9)) + 0.05 * np.sin(2 * np.pi * 31 * t)
        x.append(v)
    return np.array(x)


@pytest.mark.parametrize("channels", [1, 2])
def test_fast_path_equals_the_reference_within_50_db(tmp_path, channels):
    x = _voice_like(channels=channels)
    sf.write(tmp_path / "a.wav", x.T, SR, subtype="DOUBLE")
    prog = check(CHAIN, duration=x.shape[1] / SR, sr=SR)
    ref = render(prog, x, SR)[0]["main"]
    info = run(prog, tmp_path / "a.wav", tmp_path / "b.wav", subtype="DOUBLE")
    y = sf.read(tmp_path / "b.wav", always_2d=True)[0].T
    assert y.shape == ref.shape
    diff = 10 * np.log10(np.mean((y - ref) ** 2) / np.mean(ref ** 2))
    assert diff <= -50, diff
    assert set(info) == {"stage0", "stage1"}


def test_meter_equals_the_whole_file_measurements():
    x = _voice_like(seconds=9, channels=2, seed=3)
    m = Meter(SR, 2)
    for a in range(0, x.shape[1], 30011):  # uneven blocks
        m.feed(x[:, a:a + 30011])
    lu, tp = m.result()
    assert abs(lu - loudness.integrated(x, SR)) < 1e-6
    assert abs(tp - loudness.true_peak(x, SR)) < 1e-6


def test_with_a_loudness_node_it_equals_the_reference(tmp_path):
    x = _voice_like()
    sf.write(tmp_path / "a.wav", x.T, SR, subtype="DOUBLE")
    text = CHAIN.replace("layer main = h50 -> h60 -> r", "node lift = loudness(target_lufs: -16, max_true_peak_dbtp: -1)\n"
                         "layer main = h50 -> h60 -> r -> lift")
    prog = check(text, duration=x.shape[1] / SR, sr=SR)
    ref = render(prog, x, SR)[0]["main"]
    info = run(prog, tmp_path / "a.wav", tmp_path / "b.wav", subtype="DOUBLE")["loudness"]
    y = sf.read(tmp_path / "b.wav", always_2d=True)[0].T
    assert 10 * np.log10(np.mean((y - ref) ** 2) / np.mean(ref ** 2)) <= -50
    assert abs(loudness.integrated(y, SR) - info["lufs"]) < 0.02
    assert loudness.true_peak(y, SR) <= -1 + 0.01


def test_other_programs_are_refused():
    with pytest.raises(Unsupported):
        plan(check("zumzum 0.1\nnode c = declick()\nlayer main = c\n", duration=5, sr=SR))


def test_a_program_with_only_a_loudness_node(tmp_path):
    x = _voice_like(channels=2)
    sf.write(tmp_path / "a.wav", x.T, SR, subtype="DOUBLE")
    prog = check("zumzum 0.1\nnode lift = loudness(target_lufs: -24, max_true_peak_dbtp: -1)\nlayer main = lift\n",
                 duration=x.shape[1] / SR, sr=SR)
    ref = render(prog, x, SR)[0]["main"]
    run(prog, tmp_path / "a.wav", tmp_path / "b.wav", subtype="DOUBLE")
    y = sf.read(tmp_path / "b.wav", always_2d=True)[0].T
    assert 10 * np.log10(np.mean((y - ref) ** 2) / np.mean(ref ** 2)) <= -50
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.wav", "b.wav"]  # no temporary file left behind
