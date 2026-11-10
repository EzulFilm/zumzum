"""Loudness and true peak against the published definitions (BS.1770-4, EBU Tech 3341 test signals,
built here from their descriptions), and the loudness op and checks."""

import numpy as np
import pytest

from zumzum import loudness as L
from zumzum.checker import check
from zumzum.engine import render
from zumzum.measure import run_checks

SR = 48000


def tone(db, s, sr=SR, f=1000.0):
    t = np.arange(int(round(s * sr))) / sr
    return 10 ** (db / 20) * np.sin(2 * np.pi * f * t)


def stereo(*parts):
    x = np.concatenate(parts)
    return np.stack([x, x])


def test_k_weighting_rebuilt_at_48k_is_the_published_one():
    for (b, a), shelf in ((L.SHELF_48K, True), (L.HIGHPASS_48K, False)):
        b2, a2 = L._at_rate(b, a, 48000, shelf)
        assert np.allclose(b2, b, atol=1e-12) and np.allclose(a2, a, atol=1e-12)
    b44, _ = L._at_rate(*L.SHELF_48K, 44100, shelf=True)  # the shelf keeps its +4 dB top at any rate
    assert abs(20 * np.log10((b44[0] - b44[1] + b44[2]) / (1 - _[1] + _[2])) - 4.0) < 0.01


@pytest.mark.parametrize("sr", [48000, 44100])
def test_tech3341_tones_and_gating(sr):
    # Tech 3341 cases 1-5 (expected -23 / -33 / -23 / -23 / -23 LUFS, +-0.1 LU)
    assert abs(L.integrated(stereo(tone(-23, 20, sr)), sr) + 23) <= 0.1
    assert abs(L.integrated(stereo(tone(-33, 20, sr)), sr) + 33) <= 0.1
    assert abs(L.integrated(stereo(tone(-36, 10, sr), tone(-23, 60, sr), tone(-36, 10, sr)), sr) + 23) <= 0.1
    assert abs(L.integrated(stereo(tone(-72, 10, sr), tone(-36, 10, sr), tone(-23, 60, sr), tone(-36, 10, sr),
                                   tone(-72, 10, sr)), sr) + 23) <= 0.1
    assert abs(L.integrated(stereo(tone(-26, 20, sr), tone(-20, 20.1, sr), tone(-26, 20, sr)), sr) + 23) <= 0.1
    assert L.integrated(np.zeros((2, sr)), sr) is None


def test_true_peak_between_samples():
    # fs/4 at 45 degrees: every sample is at 0.707 (-3.01 dBFS) but the wave peaks at 1.0 (0 dBTP); Tech 3341
    # allows +0.2 / -0.4 dB. Faded in and out so the edges do not ring.
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * SR / 4 * t + np.pi / 4) * np.clip(np.minimum(t, 1 - t) / 0.05, 0, 1)
    assert abs(20 * np.log10(np.abs(x).max()) + 3.01) < 0.01
    assert -0.4 <= L.true_peak(x, SR) <= 0.2


def test_loudness_op_reaches_the_target_when_the_peaks_leave_room():
    x = stereo(tone(-30, 10))  # crest 3 dB: plenty of room under the ceiling
    prog = check("zumzum 0.1\nnode n = loudness(target_lufs: -16, max_true_peak_dbtp: -1)\nlayer m = n\n"
                 "check l = loudness(target_lufs: -16, tol_lu: 0.5)\ncheck t = true_peak(max_dbtp: -1)\n")
    outs, info = render(prog, x, SR)
    res = {r["name"]: r for r in run_checks(prog, x, outs, SR)}
    assert res["l"]["pass"] and abs(res["l"]["value"] + 16) < 0.1, res["l"]["text"]
    assert res["t"]["pass"] and info["m.n"]["short_by_lu"] == 0


def test_loudness_op_is_one_gain_capped_by_the_true_peak_ceiling():
    rng = np.random.default_rng(1)
    x = 0.05 * rng.standard_normal((2, 10 * SR))
    x[:, 4 * SR:4 * SR + 100] = 0.9  # a peak that leaves almost no room
    prog = check("zumzum 0.1\nnode n = loudness(target_lufs: -16, max_true_peak_dbtp: -1)\nlayer m = n\n"
                 "check l = loudness(target_lufs: -16, tol_lu: 0.5)\ncheck t = true_peak(max_dbtp: -1)\n")
    outs, info = render(prog, x, SR)
    res = {r["name"]: r for r in run_checks(prog, x, outs, SR)}
    y, inf = outs["m"], info["m.n"]
    g = y[0, 1000] / x[0, 1000]
    assert np.allclose(y, x * g)  # one gain for every sample of every channel: nothing limited or compressed
    assert res["t"]["pass"] and abs(res["t"]["value"] + 1) < 0.05, res["t"]["text"]  # the peak sits at the ceiling
    assert not res["l"]["pass"] and inf["short_by_lu"] > 3  # short of the target, and it says by how much


def test_loudness_op_leaves_silence():
    prog = check("zumzum 0.1\nnode n = loudness(target_lufs: -16)\nlayer m = n\n")
    outs, _ = render(prog, np.zeros((1, SR)), SR)
    assert not outs["m"].any()


def test_loudness_range_of_steps():
    # Tech 3342-style: 20 s at -20 then 20 s at -30 LUFS reads 10 LU (+-1)
    x = stereo(tone(-20, 20), tone(-30, 20))
    assert abs(L.loudness_range(x, SR) - 10) <= 1
    assert L.loudness_range(np.zeros((1, 2 * SR)), SR) is None


