"""Block-wise checks and damage gate (zumzum/stream_checks.py) against the whole-file ones (measure.run_checks and
measure.damage). Written before the code. Same result means: every check has the same pass / fail and its value
within 0.02 (dB or LU), and the damage gate gives the same list of problems (same kinds, in the same order)."""

import re

import numpy as np
import pytest
import soundfile as sf

from zumzum.checker import check
from zumzum.engine import render
from zumzum.measure import damage, run_checks

SR = 48000
PROGRAM = """zumzum 0.1
sel low = band(f_hz: 0..75, feather_hz: 15)
node h50 = dehum(f0_hz: 50, harmonics: 16, depth_db: 50)
node h60 = dehum(f0_hz: 60, harmonics: 16, depth_db: 50)
node r = gain(db: -30) at low
node lift = loudness(target_lufs: -18, max_true_peak_dbtp: -1)
layer main = h50 -> h60 -> r -> lift
check level = loudness(target_lufs: -18, tol_lu: 1)
check peaks = true_peak(max_dbtp: -1)
check range = loudness_range(max_lu: 20)
check body = rms(min_dbfs: -40, max_dbfs: -10)
check hum = hum_drop(f0_hz: 50, harmonics: 10, min_db: 6)
check mains = mains_hum_drop(harmonics: 10, min_db: 6)
"""


def _voice_like(seconds, channels, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    out = []
    for c in range(channels):
        v = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / k for k, f in enumerate((180, 367, 742, 1510, 2890), 1))
        v = v * (0.2 + 0.8 * np.abs(np.sin(2 * np.pi * (0.4 + 0.3 * c) * t))) * 0.1 + 0.003 * rng.standard_normal(len(t))
        v += 0.02 * sum(np.sin(2 * np.pi * 50.07 * k * t) / k for k in range(1, 9)) + 0.05 * np.sin(2 * np.pi * 31 * t)
        out.append(v)
    return np.array(out)


def _kinds(problems):
    return [re.sub(r"[-+]?\d+(\.\d+)?", "#", p) for p in problems]


def _same(whole, streamed):
    assert [c["name"] for c in whole] == [c["name"] for c in streamed]
    for a, b in zip(whole, streamed):
        assert a["pass"] == b["pass"], (a, b)
        if a.get("value") is None:
            assert b.get("value") is None
        else:
            assert abs(a["value"] - b["value"]) <= 0.02, (a, b)


@pytest.mark.parametrize("seconds,channels,seed", [(12, 1, 1), (9, 2, 2), (40, 1, 3)])
def test_block_checks_equal_the_whole_file_checks(tmp_path, seconds, channels, seed):
    from zumzum.stream import run
    from zumzum.stream_checks import file_checks
    x = _voice_like(seconds, channels, seed)
    sf.write(tmp_path / "in.wav", x.T, SR, subtype="DOUBLE")
    prog = check(PROGRAM, duration=x.shape[1] / SR, sr=SR)
    run(prog, tmp_path / "in.wav", tmp_path / "out.wav", subtype="DOUBLE")
    y = sf.read(tmp_path / "out.wav", always_2d=True)[0].T
    whole_checks, whole_damage = run_checks(prog, x, {"main": y}, SR), damage(x, y, prog, SR)
    streamed_checks, streamed_damage = file_checks(prog, tmp_path / "in.wav", tmp_path / "out.wav")
    _same(whole_checks, streamed_checks)
    assert _kinds(whole_damage) == _kinds(streamed_damage)


def test_the_gate_finds_the_same_problems(tmp_path):
    from zumzum.stream_checks import file_checks
    x = _voice_like(6, 2, 4)
    prog = check(PROGRAM, duration=x.shape[1] / SR, sr=SR)
    sf.write(tmp_path / "in.wav", x.T, SR, subtype="DOUBLE")
    for name, y in (("too loud", x * 40), ("silent", x * 1e-4), ("clipped", np.clip(x * 30, -1.2, 1.2))):
        sf.write(tmp_path / "out.wav", y.T, SR, subtype="DOUBLE")
        whole = damage(x, y, prog, SR)
        assert whole, name
        assert _kinds(whole) == _kinds(file_checks(prog, tmp_path / "in.wav", tmp_path / "out.wav")[1]), name
        _same(run_checks(prog, x, {"main": y}, SR), file_checks(prog, tmp_path / "in.wav", tmp_path / "out.wav")[0])


def test_unsupported_checks_are_refused(tmp_path):
    from zumzum.stream_checks import Unsupported, file_checks
    x = _voice_like(3, 1, 5)
    sf.write(tmp_path / "in.wav", x.T, SR, subtype="DOUBLE")
    prog = check("zumzum 0.1\nnode g = gain(db: -3)\nlayer main = g\ncheck c = clicks(max: 3)\n", duration=3, sr=SR)
    with pytest.raises(Unsupported):
        file_checks(prog, tmp_path / "in.wav", tmp_path / "in.wav")


def test_render_matches_the_reference_for_this_program():
    """Guard for the test itself: the program renders the same through the reference engine (no exception)."""
    x = _voice_like(3, 1, 6)
    prog = check(PROGRAM, duration=3, sr=SR)
    assert render(prog, x, SR)[0]["main"].shape == x.shape
