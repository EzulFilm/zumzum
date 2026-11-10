"""fix --spec: the rms / noise_floor checks and the four delivery rule sets."""

import numpy as np
import pytest

from zumzum import loudness
from zumzum.checker import check
from zumzum.engine import render
from zumzum.measure import noise_floor_dbfs, rms_dbfs, run_checks
from zumzum.specs import SPECS, measure_all, program, verdict

SR = 48000


def talk(seconds=20, peaky=False):
    """A voice-like signal with pauses over a -84 dBFS hiss: three harmonics in 0.7 s phrases. Its peaks sit about
    6 dB over its level, so one gain reaches every target. peaky=True adds a spike every 0.25 s."""
    rng = np.random.default_rng(0)
    n = seconds * SR
    t = np.arange(n) / SR
    env = np.clip(4 * np.sin(2 * np.pi * t / 1.0), 0, 1)  # 0.5 s on, 0.5 s pause, soft edges
    v = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in (1, 2, 3))
    x = 0.05 * env * v + 10 ** (-84 / 20) * rng.standard_normal(n)
    if peaky:
        x[::SR // 4] = 0.95
    return x[None, :]


def test_rms_reads_a_full_scale_sine_at_minus_3():
    x = np.sin(2 * np.pi * 997 * np.arange(SR) / SR)[None, :]
    assert rms_dbfs(x) == pytest.approx(-3.01, abs=0.01)


def test_noise_floor_skips_digital_silence_and_finds_the_quietest_span():
    rng = np.random.default_rng(1)
    x = 0.1 * rng.standard_normal(4 * SR)
    x[SR:2 * SR] = 0.0  # a pad of digital silence: not a floor
    x[3 * SR:int(3.6 * SR)] = 0.001 * rng.standard_normal(int(0.6 * SR))  # -60 dBFS room tone
    assert noise_floor_dbfs(x[None, :], SR) == pytest.approx(-60, abs=0.5)
    assert noise_floor_dbfs(np.zeros((1, SR)), SR) is None


def test_the_new_checks_go_through_the_checker_and_report():
    x = talk(6)
    prog = check("zumzum 0.1\nlayer main = pass\ncheck r = rms(min_dbfs: -40, max_dbfs: -10)\n"
                 "check f = noise_floor(max_dbfs: -60)\n", duration=6, sr=SR)
    outs, _ = render(prog, x, SR)
    res = {r["name"]: r for r in run_checks(prog, x, outs, SR)}
    assert res["r"]["pass"] and "RMS" in res["r"]["text"]
    assert res["f"]["pass"] and res["f"]["value"] < -65


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("name", sorted(SPECS))
def test_fix_meets_each_rule_on_the_delivered_file(name, tmp_path):
    """End to end: program, engine, checks, damage gate, encode (MP3 / WAV), decode and measure again."""
    from zumzum.audio import save
    from tools.fix import fix
    src = tmp_path / "talk.wav"
    save(src, talk(), SR)
    r = fix(name, src, tmp_path / "out", sheet=False)
    last = r["versions"][-1]
    assert r["pass"], (last["verdict"], [c["text"] for c in last["checks"]], last["damage"])
    assert (tmp_path / "out" / "talk.zz").read_text().startswith("zumzum 0.1")
    key = "rms_dbfs" if SPECS[name]["kind"] == "rms" else "lufs"
    assert abs(r["delivered"][key] - SPECS[name]["target"]) <= 0.15 and not r["warnings"], r["delivered"]


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_file_that_needs_limiting_is_reported_not_limited(tmp_path):
    from zumzum.audio import save
    from tools.fix import fix
    save(tmp_path / "spiky.wav", talk(peaky=True), SR)
    r = fix("podcast-apple", tmp_path / "spiky.wav", tmp_path / "out", sheet=False)
    assert not r["pass"] and r["warnings"] and "needs limiting" in r["warnings"][0]
    assert r["delivered"]["true_peak_dbtp"] <= -1.0  # the peak limit is still kept


def test_verdict_tells_the_rule_from_the_stricter_test():
    m = {"lufs": -23.8, "true_peak_dbtp": -1.2, "rms_dbfs": -26, "noise_floor_dbfs": -80, "lra_lu": 5}
    v = verdict("ebu-r128", m)
    assert not v["rule"]["loudness"] and not v["test"]["loudness_within_0.5"]
    v = verdict("atsc-a85", dict(m, lufs=-25.5, true_peak_dbtp=-2.5))
    assert v["rule"]["loudness"] and not v["test"]["loudness_within_0.5"]  # inside A/85's +-2, outside the test's 0.5
