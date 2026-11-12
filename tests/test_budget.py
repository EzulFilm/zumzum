"""The damage gate's change budget for hum and rumble, and the voice clean-up preset that uses it."""

import numpy as np

from zumzum.checker import check

SR = 44100


def _voice(seconds=6, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)  # a gliding voice: its harmonics move
    ph = 2 * np.pi * np.cumsum(f0) / SR
    v = sum(np.sin(k * ph) / k for k in range(1, 25)) * (0.2 + 0.8 * np.abs(np.sin(2 * np.pi * 1.1 * t)))
    return 0.05 * v + 0.001 * rng.standard_normal(len(t))


def test_budget_lets_a_dehum_through_where_there_is_hum_and_stops_one_where_there_is_none():
    from zumzum.measure import budget_problems
    t = np.arange(SR * 6) / SR
    hum = sum(0.02 / k * np.sin(2 * np.pi * 50 * k * t) for k in range(1, 12))  # dehum: +4.7 dB; at 0.006 it
    # loses 5.4 dB (the gliding voice's harmonics weigh more than the hum), and the rule rightly stops it
    prog = check("zumzum 0.1\nnode h = dehum(f0_hz: 50, harmonics: 16, depth_db: 50)\nlayer main = h\n", duration=6, sr=SR)
    assert budget_problems(prog, (_voice() + hum)[None], SR) == []
    problems = budget_problems(prog, _voice()[None], SR)
    assert problems and "took programme" in problems[0]


def test_budget_lets_a_low_cut_through_on_rumble_and_stops_it_on_steady_bass_free_music():
    from scipy import signal
    from zumzum.measure import budget_problems
    rng = np.random.default_rng(8)
    t = np.arange(SR * 6) / SR
    rumble = signal.sosfilt(signal.butter(4, 70, "lowpass", fs=SR, output="sos"), rng.standard_normal(len(t))) * 0.03
    prog = check("zumzum 0.1\nsel low = band(f_hz: 0..75, feather_hz: 15)\nnode r = gain(db: -30) at low\nlayer main = r\n",
                 duration=6, sr=SR)
    assert budget_problems(prog, (_voice() + rumble)[None], SR) == []
    bass = 0.05 * np.sin(2 * np.pi * 55 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)  # a bass note that comes and goes
    assert budget_problems(prog, (_voice() + bass)[None], SR)


def test_preset_keeps_only_what_the_input_needs(tmp_path):
    import soundfile as sf
    from scipy import signal
    from zumzum.preset import clean_up
    rng = np.random.default_rng(9)
    t = np.arange(SR * 8) / SR
    v = _voice(8)
    hum60 = sum(0.02 / k * np.sin(2 * np.pi * 60 * k * t) for k in range(1, 12))
    rumble = signal.sosfilt(signal.butter(4, 70, "lowpass", fs=SR, output="sos"), rng.standard_normal(len(t))) * 0.1
    for x, want in ((v, set()), (v + hum60, {"h60"}), (v + rumble, {"r"})):  # rumble: >= 1 % of the energy
        sf.write(tmp_path / "in.wav", x, SR, subtype="FLOAT")
        r = clean_up(tmp_path / "in.wav", tmp_path / "out.wav")
        assert {c["node"] for c in r["choice"] if c["kept"]} == want, r["choice"]
        assert r["damage"] == [] and "loudness(" in r["program"]
        (tmp_path / "out.wav").unlink()
