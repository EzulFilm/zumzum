"""Display spectrograms and sheets (review output only)."""

import numpy as np

from zumzum import sheets

SR = 44100


def _tone(hz, dur=1.0, amp=1.0):
    t = np.arange(int(dur * SR)) / SR
    return amp * np.sin(2 * np.pi * hz * t)


def test_pretty_spec_shape_and_tone_peak():
    x = _tone(1000) + 1e-4 * np.random.default_rng(0).standard_normal(SR)
    S, f, t = sheets.pretty_spec(x, SR)
    assert S.shape == (len(f), len(t)) and np.isfinite(S).all()
    assert f[0] == 30 and np.all(np.diff(f) > 0)
    prof = S[:, len(t) // 2]
    assert abs(f[np.argmax(prof)] - 1000) < 30
    assert abs(prof.max()) < 3  # medium-window band: a full-scale sine reads about 0 dBFS


def test_band_weights_sum_to_one():
    w = sheets._band_weights(np.geomspace(20, 20000, 200))
    assert np.allclose(w.sum(axis=1), 1) and (w >= 0).all()
    assert w[0, 0] == 1 and w[-1, 2] == 1


def test_short_window_keeps_a_click_sharp():
    x = 1e-4 * np.random.default_rng(1).standard_normal(SR)
    x[SR // 2] = 1.0
    S, f, t = sheets.pretty_spec(x, SR)
    hi = S[f > 5000].mean(axis=0)
    peak = np.argmax(hi)
    assert abs(t[peak] - 0.5) < 0.01
    assert hi[peak] - hi[np.abs(t - 0.5) > 0.03].max() > 20  # sharp in time, not smeared over 100 ms


def test_resample_coarse_keeps_narrow_peak():
    src = np.arange(1000.0)
    D = np.full((1000, 1), -100.0)
    D[501] = 0.0
    out = sheets._resample(D, src, np.arange(0.0, 1000.0, 50.0))
    assert out.max() > -100 + 10  # a single-bin line is averaged into its cell, not skipped


def test_stereo_and_short_input():
    x = np.stack([_tone(440, 0.05), -_tone(440, 0.05)])  # opposite channels must not cancel
    S, f, t = sheets.pretty_spec(x, SR)
    assert np.isfinite(S).all() and S.max() > -10


def test_levels_bottom_tracks_noise_floor():
    rng = np.random.default_rng(2)
    S = -80 + rng.standard_normal((100, 400))
    S[40:42] = -10
    vmin, vmax = sheets.levels(S, range_db=120)
    assert -12 < vmax < -8
    assert -100 < vmin < -95  # floor (about -81) minus the 18 dB margin, not 120 below the top


def test_sheet_writes_png_without_metadata(tmp_path):
    rng = np.random.default_rng(3)
    before = 0.3 * _tone(220, 2.0) + 0.01 * rng.standard_normal(2 * SR)
    after = 0.3 * _tone(220, 2.0)
    checks = [{"name": "a_$x$", "pass": True, "text": "costs $5 and $6"}]  # dollar signs stay plain text
    p = tmp_path / "s.png"
    sheets.sheet(p, before, after, SR, "t $x^2$", checks, "p.zz", fmax=8000, zoom=(0.5, 1.5, 0, 1000))
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # nothing beyond matplotlib's own "Software" entry (no title, program or path in the file)
    assert data.count(b"tEXt") == data.count(b"tEXtSoftware") <= 1
    assert b"iTXt" not in data and b"zTXt" not in data
