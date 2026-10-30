import numpy as np
import pytest

from zumzum.checker import check
from zumzum.engine import detect_clicks, render
from zumzum.measure import damage, run_checks
from zumzum.stft import istft, stft

SR = 48000


def tone(f, dur=2.0, amp=0.1, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return amp * np.sin(2 * np.pi * f * t)


def speechlike(dur=3.0, seed=0, sr=SR):
    """Band-limited noise bursts, amplitude modulated like syllables (no recording needed)."""
    from scipy import signal
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = signal.sosfilt(signal.butter(4, [300, 3400], "bandpass", fs=sr, output="sos"), rng.standard_normal(n))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(n) / sr) ** 2
    return 0.1 * x / np.std(x) * env


@pytest.mark.parametrize("size,overlap", [(256, 2), (1024, 4), (2048, 4), (4096, 8), (16384, 4)])
@pytest.mark.parametrize("n", [1, 1000, 48000, 48001])
def test_stft_perfect_reconstruction(size, overlap, n):
    x = np.random.default_rng(n).standard_normal(n)
    hop = size // overlap
    y = istft(stft(x, size, hop), size, hop, n)
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-12


def run(src, x, sr=SR):
    prog = check(src, duration=x.shape[-1] / sr, sr=sr)
    outs, info = render(prog, x, sr)
    return prog, outs, info, run_checks(prog, np.atleast_2d(x), outs, sr)


def test_identity_layer_is_exact():
    x = speechlike()
    _, outs, _, _ = run("zumzum 0.1\nnode g = gain(db: 0)\nlayer main = g\n", x)
    assert np.max(np.abs(outs["main"][0] - x)) < 1e-12
    _, outs, _, _ = run("zumzum 0.1\nlayer main = pass\n", x)
    assert np.array_equal(outs["main"][0], x)


def test_gain_in_box_only():
    x = speechlike()
    src = """zumzum 0.1
sel b = box(t_s: 1..2, f_hz: 500..1500)
sel rest = not b
node g = gain(db: -30) at b
layer main = g
check down = drop(at: b, min_db: 25)
check rest_same = untouched(at: rest, max_db: -30)
"""
    _, _, _, res = run(src, x)
    assert all(r["pass"] for r in res), res


def test_dehum_removes_60hz_and_keeps_speech():
    x = speechlike(dur=4)
    hum = sum(0.02 / k * np.sin(2 * np.pi * 60.1 * k * np.arange(len(x)) / SR + k) for k in range(1, 9))
    src = """zumzum 0.1
sel voice = band(f_hz: 300..3400)
node h = dehum(f0_hz: 60, harmonics: 12)
layer main = h
check hum = hum_drop(f0_hz: 60, harmonics: 8, min_db: 30)
check speech = keep(at: voice, tol_db: 1)
"""
    _, _, info, res = run(src, x + hum)
    assert all(r["pass"] for r in res), res
    assert abs(info["main.h"]["f0_hz"] - 60.1) < 0.05


def test_denoise_lowers_noise_floor():
    rng = np.random.default_rng(3)
    x = speechlike(dur=4)
    x[: SR] = 0  # first second: speech pause
    noisy = x + 0.01 * rng.standard_normal(len(x))
    for mode, need in (("mmse", 12), ("wiener", 12), ("subtract", 8)):
        src = f"""zumzum 0.1
sel pause = time(t_s: 0.1..0.9)
node d = denoise(mode: {mode}, profile_t_s: 0..1, reduce_db: 20)
layer main = d
check quiet = drop(at: pause, min_db: {need})
"""
        _, _, _, res = run(src, noisy)
        assert all(r["pass"] for r in res), (mode, res)


def test_declick_repairs_clicks():
    rng = np.random.default_rng(5)
    x = tone(440, 2.0, 0.3) + tone(660, 2.0, 0.1) + 0.001 * rng.standard_normal(2 * SR)
    clean = x.copy()
    pos = rng.integers(1000, len(x) - 1000, 20)
    for p in pos:
        x[p:p + 8] += rng.choice([-1, 1]) * 0.4 * np.hanning(8)
    assert len(detect_clicks(x, SR, 10)) >= 20
    _, outs, info, res = run("zumzum 0.1\nnode c = declick()\nlayer main = c\ncheck n = clicks(max: 1)\n", x)
    assert all(r["pass"] for r in res), res
    err_before = np.sqrt(np.mean((x - clean) ** 2))
    err_after = np.sqrt(np.mean((outs["main"][0] - clean) ** 2))
    assert err_after < 0.05 * err_before


def test_heal_removes_tone_burst():
    x = speechlike(dur=3)
    burst = np.zeros_like(x)
    i0, i1 = int(1.2 * SR), int(1.6 * SR)
    burst[i0:i1] = tone(1800, 0.4, 0.2)[: i1 - i0] * np.hanning(i1 - i0)
    src = """zumzum 0.1
sel ring = box(t_s: 1.15..1.65, f_hz: 1700..1900)
sel rest = not ring
node h = heal() at ring
layer main = h
check gone = drop(at: ring, min_db: 15)
check others = untouched(at: rest, max_db: -30)
"""
    _, _, _, res = run(src, x + burst)
    assert all(r["pass"] for r in res), res


def test_split_hp_parts_sum_close_to_input():
    rng = np.random.default_rng(1)
    x = tone(220, 2.0, 0.2)
    for p in range(10):
        i = int((0.1 + 0.18 * p) * SR)
        x[i:i + 200] += 0.5 * rng.standard_normal(200) * np.hanning(200)
    src = """zumzum 0.1
node h = split_hp(part: harmonic)
node p = split_hp(part: percussive)
layer harm = h
layer perc = p
output harm
"""
    _, outs, _, _ = run(src, x)
    s = outs["harm"][0] + outs["perc"][0]
    assert np.sqrt(np.mean((s - x) ** 2)) < 0.35 * np.sqrt(np.mean(x ** 2))
    # the 220 Hz tone lives in the harmonic part: its spectral line is far stronger there than in the percussive part
    f = np.fft.rfftfreq(len(x), 1 / SR)
    line = np.abs(f - 220) < 3
    e_h = np.sum(np.abs(np.fft.rfft(outs["harm"][0]))[line] ** 2)
    e_p = np.sum(np.abs(np.fft.rfft(outs["perc"][0]))[line] ** 2)
    assert e_h > 30 * e_p, (e_h, e_p)


def test_damage_gate():
    x = speechlike()
    assert damage(x, x) == []
    assert any("clipping" in d for d in damage(x, x * 20))
    assert any("quieter" in d for d in damage(x, x * 1e-4))


def test_bypass_mix_and_per_node_effect():
    x = speechlike()
    src = """zumzum 0.1
sel b = box(t_s: 1..2, f_hz: 500..1500)
node a = gain(db: -20) at b
node z = gain(db: -60) off
layer x = a -> z
layer dry = pass
output mix(x: 0, dry: -120)
"""
    prog = check(src)
    outs, info = render(prog, x, SR)
    assert info["x.z"] == {"off": True}
    assert abs(info["x.a"]["inside_db"] + 20) < 0.01 and abs(info["x.a"]["outside_db"]) < 0.01
    assert np.allclose(outs["mix"], outs["x"] + 1e-6 * outs["dry"])


def test_denoise_noise_sample_by_selection_and_gate():
    rng = np.random.default_rng(4)
    x = speechlike(dur=4)
    x[: SR // 2] = 0
    x[2 * SR: int(2.5 * SR)] = 0
    noisy = x + 0.01 * rng.standard_normal(len(x))
    src = """zumzum 0.1
sel noise = time(t_s: 0.05..0.45) or time(t_s: 2.05..2.45)
sel pause = time(t_s: 2.1..2.4)
node d = denoise(profile: noise, reduce_db: 30, gate_db: 10)
layer main = d
check quiet = drop(at: pause, min_db: 20)
"""
    prog, _, _, res = run(src, noisy)
    assert not prog.warnings, prog.warnings
    assert all(r["pass"] for r in res), res


def test_declick_selection_follows_the_analysis_grid():
    from zumzum.engine import time_weight
    sel = ("call", "time", {"t_s": (0.2, 0.3), "feather_ms": 30.0})
    n = SR
    a = time_weight(sel, n, SR, 2048, 512)
    b = time_weight(sel, n, SR, 4096, 2048)  # a coarser hop: different frame times
    assert np.max(np.abs(a - b)) > 0.05  # the grid matters, so the engine must use the program's
    x = speechlike(dur=1)
    src = "zumzum 0.1\nanalysis stft(size: 4096, overlap: 8)\nsel s = time(t_s: 0.2..0.3, feather_ms: 30)\n" \
          "node c = declick() at s mix 0.5\nlayer main = c\n"
    prog = check(src)
    outs, _ = render(prog, x, SR)
    w = time_weight(prog.ops["c"]["at"], n, SR, prog.size, prog.hop)
    outside = w == 0
    assert np.array_equal(outs["main"][0][outside], x[outside])  # nothing changes where the selection is 0


def test_gate_follows_a_noise_step_per_frame():
    rng = np.random.default_rng(9)
    n = 6 * SR
    noise = rng.standard_normal(n) * np.where(np.arange(n) < 3 * SR, 0.001, 0.01)  # noise only, 20 dB step at 3 s
    src = "zumzum 0.1\nnode d = denoise(reduce_db: 0, track_s: 0.3, gate_db: 12, gate_release_ms: 10)\nlayer main = d\n"
    prog = check(src)
    outs, _ = render(prog, noise, SR)
    y = outs["main"][0]
    seg = slice(int(5 * SR), int(5.8 * SR))  # well after the step and after the 0.3 s tracker window
    drop = 10 * np.log10(np.mean(noise[seg] ** 2) / np.mean(y[seg] ** 2))
    assert drop > 9, drop  # the gate closes on the louder noise too (about 12 dB)


def test_damage_gate_catches_a_quiet_input_erased():
    x = 1e-5 * speechlike(dur=1) / 0.1
    assert any("quieter" in d for d in damage(x, x * 1e-6))
    assert damage(np.zeros(100), np.zeros(100)) == []  # digital silence has nothing to erase


def test_hum_drop_on_very_short_audio():
    src = "zumzum 0.1\nlayer main = pass\ncheck h = hum_drop(f0_hz: 60, min_db: 10)\n"
    x = np.array([0.1])
    prog = check(src)
    outs, _ = render(prog, x, SR)
    res = run_checks(prog, np.atleast_2d(x), outs, SR)
    assert res[0]["value"] == 0.0 and not res[0]["pass"] and "not measurable" in res[0]["text"]


def test_audio_limits_refuse_bad_input(tmp_path, monkeypatch):
    """Absurd sample rates, too many channels, too long, too many samples in all, or non-finite samples are refused
    before any work, wherever the audio comes in (load, check, render, the editing loop)."""
    import soundfile as sf
    from zumzum import llm
    from zumzum.audio import load, validate
    from zumzum.errors import CheckError
    from zumzum.ops import LIMITS
    for sr in (2147483647, float("nan"), 4000):
        with pytest.raises(CheckError, match="sample rate"):
            check("zumzum 0.1\nlayer main = pass\n", sr=sr)
    for d in (4000, -1, float("nan")):
        with pytest.raises(CheckError, match="s long"):
            check("zumzum 0.1\nlayer main = pass\n", duration=d)
    check("zumzum 0.1\nlayer main = pass\n", duration=3600, sr=8000)
    check("zumzum 0.1\nlayer main = pass\n", duration=1, sr=192000)  # the limits themselves are allowed
    x = speechlike(dur=0.5)
    for bad, sr, msg in ((x, 2147483647, "sample rate"), (np.zeros((9, 100)), SR, "channels"),
                         (np.array([0.0, np.inf]), SR, "non-finite"), (np.array([np.nan]), SR, "non-finite")):
        with pytest.raises(ValueError, match=msg):
            validate(bad, sr)
    prog = check("zumzum 0.1\nnode g = gain(db: 0)\nlayer main = g\n")
    with pytest.raises(ValueError, match="non-finite"):
        render(prog, np.array([0.0, np.inf, 0.0]), SR)
    with pytest.raises(ValueError, match="sample rate"):
        render(prog, x, 2147483647)
    with pytest.raises(ValueError, match="sample rate"):  # the loop refuses before the model sees anything
        llm.edit_loop("less hiss", x, 2147483647, provider=None)
    sf.write(tmp_path / "ok.wav", x, SR)
    assert load(tmp_path / "ok.wav")[1] == SR
    sf.write(tmp_path / "slow.wav", x[:4000], 4000)  # below 8 kHz
    with pytest.raises(ValueError, match="sample rate"):
        load(tmp_path / "slow.wav")
    sf.write(tmp_path / "nine.wav", np.zeros((100, 9)), SR)
    with pytest.raises(ValueError, match="channels"):
        load(tmp_path / "nine.wav")
    monkeypatch.setitem(LIMITS, "duration_s", 0.25)  # a long file, without writing an hour of audio
    with pytest.raises(ValueError, match="at most 0.25 s"):
        load(tmp_path / "ok.wav")
    with pytest.raises(ValueError, match="at most 0.25 s"):  # also when a length is asked for, before reading
        load(tmp_path / "ok.wav", dur_s=1e9)
    assert load(tmp_path / "ok.wav", start_s=0.3, dur_s=0.1)[0].shape[1] == int(0.1 * SR)
    monkeypatch.setitem(LIMITS, "duration_s", 3600)
    monkeypatch.setitem(LIMITS, "samples", 1000)  # the total budget, across channels
    sf.write(tmp_path / "wide.wav", np.zeros((600, 2)), SR)
    with pytest.raises(ValueError, match="samples in all"):
        load(tmp_path / "wide.wav")


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
def test_convert_reads_local_audio_but_not_playlists(tmp_path):
    import subprocess
    import soundfile as sf
    from tools.convert import local, probe
    sf.write(tmp_path / "in.wav", speechlike(dur=0.5), SR)
    plain = lambda src: subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "null", "-"],
                                       capture_output=True, text=True)
    guarded = lambda src: subprocess.run(["ffmpeg", "-v", "error", "-y", *local(src), "-f", "null", "-"],
                                         capture_output=True, text=True)
    assert guarded(tmp_path / "in.wav").returncode == 0
    most, sr, ch = probe(tmp_path / "in.wav")
    assert 0.5 < most <= 3601 and (sr, ch) == (SR, 1)
    # a valid playlist of one MPEG-TS segment: plain ffmpeg plays it (the control), the guarded input refuses it
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_path / "in.wav"), "-c:a", "aac", "-f", "mpegts",
                    str(tmp_path / "seg.ts")], check=True)
    (tmp_path / "list.m3u8").write_text("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n#EXTINF:0.5,\n"
                                        "seg.ts\n#EXT-X-ENDLIST\n")
    assert plain(tmp_path / "list.m3u8").returncode == 0
    r = guarded(tmp_path / "list.m3u8")
    assert r.returncode != 0 and "whitelist" in r.stderr, r.stderr
