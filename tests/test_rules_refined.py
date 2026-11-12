"""Note-to-program refinements: rumble said as a drone or boom underneath, narrow boxes for keypad tones, a phone in
other words, one mains series when one clearly leads, and a gentler denoise that only runs when asked."""

import json

import numpy as np

from zumzum.analysis import analyze, mains_guess
from zumzum.checker import check
from zumzum.llm import RulesProvider, extract
from zumzum.notes import note_goals

SR = 44100


def _goals(note):
    return [g["goal"] for g in note_goals(note.lower(), {"tonal_events": [], "hum": None, "clipped_pct": 0})[2]]


def test_rumble_said_as_a_drone_or_boom_underneath():
    for note in ("There's a boomy basement-like drone below the voice; tame it without thinning the narrator.",
                 "An air-conditioning unit drones below the reading.",
                 "A low mechanical drone sits under the sustained piano notes."):
        assert "rumble" in _goals(note), note
    for note in ("The bagpipe drone under the melody is lovely, keep it.", "a drone shot of the city at night",
                 "the singer's low drone is part of the piece", "the organ's low drone is beautiful"):
        assert "rumble" not in _goals(note), note


def test_keypad_tone_boxes_are_narrow_and_other_tones_keep_60_hz():
    ev = [{"t_s": [6.2, 6.33], "f_hz": 1336.0, "above_db": 30}, {"t_s": [6.2, 6.33], "f_hz": 770.0, "above_db": 28}]
    sels = note_goals("someone dials a phone at about 0:06, remove the keypad tones", {"tonal_events": ev})[0]
    box = next(s for s in sels if s.startswith("sel note_sel_tones"))
    assert "f_hz: 1304..1368" in box and "f_hz: 746.5..793.5" in box  # 1.5 % + 12 Hz either side
    sels = note_goals("there is a whistle tone around 0:06, take it out", {"tonal_events": ev})[0]
    assert "f_hz: 1276..1396" in next(s for s in sels if s.startswith("sel note_sel_tones"))


def test_mains_guess_names_the_series_that_is_there():
    t = np.arange(SR * 6) / SR
    rng = np.random.default_rng(3)
    for f0 in (50.0, 60.0):
        x = 0.05 * rng.standard_normal(len(t)) + sum(0.004 / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 10))
        g = mains_guess(x[None], SR)
        assert abs(g["f0_hz"] - f0) < 1.0 and g["lead_db"] > 6, (f0, g)


def test_rules_driver_writes_one_dehum_node_when_one_series_clearly_leads():
    t = np.arange(SR * 6) / SR
    rng = np.random.default_rng(4)
    x = 0.05 * rng.standard_normal(len(t)) + sum(0.004 / k * np.sin(2 * np.pi * 60 * k * t) for k in range(1, 10))
    a = analyze(x[None], SR)
    a["hum"] = None  # as when no hum line clears the analysis' 6 dB in every quarter
    prog = extract(RulesProvider().complete("", f"Note: remove the mains buzz\nAnalysis: x\nAnalysis JSON: {json.dumps(a)}"))[0]
    assert prog.count("dehum(") == 1 and "f0_hz: 60" in prog
    assert check(prog, duration=6, sr=SR)


def test_a_phone_in_other_words_keeps_only_telephone_frequencies():
    """"telephone button beeps" and "number-pad beeps" are a phone too: only telephone frequencies are healed."""
    ev = [{"t_s": [5.0, 5.2], "f_hz": 8800.0, "above_db": 40}, {"t_s": [5.0, 5.2], "f_hz": 1336.0, "above_db": 30}]
    for note in ("There are telephone button beeps around 0:05. Remove them.",
                 "You'll hear number-pad beeps at about 0:05. Get rid of them.",
                 "Someone presses the keypad around 0:05, remove the beeps",
                 "push-button beeps around 0:05, please remove them",
                 "Push-button tones interrupt the reading at about 0:05. Remove the tones, keep the voice."):
        sels = note_goals(note, {"tonal_events": ev})[0]
        box = next(s for s in sels if s.startswith("sel note_sel_tones"))
        assert box.count("box(") == 1 and "f_hz: 1304..1368" in box, (note, box)  # only the telephone frequency


def test_tone_without_a_phone_or_button_is_still_timbre():
    assert "tones" not in _goals("the tone of the piano is lovely, keep it")
    assert "tones" not in _goals("the tones of the cello are warm")


def test_quietest_frame_noise_estimate_is_capped_under_500_hz():
    from zumzum.engine import noise_psd
    from zumzum.stft import bin_freqs, frame_times, stft
    t = np.arange(SR * 6) / SR
    rng = np.random.default_rng(5)
    on = (np.sin(2 * np.pi * 0.4 * t) > 0).astype(float)  # bass and treble take turns
    bass = 0.2 * np.sin(2 * np.pi * 110 * t) * on  # the quietest whole frames hold this bass
    treble = 0.5 * np.sin(2 * np.pi * 3000 * t) * (1 - on)
    x = bass + treble + 0.005 * rng.standard_normal(len(t))
    P = np.abs(stft(x, 2048, 512)) ** 2
    f, tt = bin_freqs(2048, SR), frame_times(P.shape[1], 2048, 512, SR)
    N = noise_psd(P, tt, None, f)
    b = np.argmin(np.abs(f - 110))
    plain = P[:, np.argsort(P.sum(axis=0))[:max(5, int(0.1 * P.shape[1]))]].mean(axis=1)
    assert N[b] < plain[b] and np.allclose(N[f >= 500], plain[f >= 500])


def test_rules_denoise_only_when_asked_and_gently():
    import json
    from zumzum.llm import RulesProvider, extract
    a = {"peak_dbfs": -6, "hum": None, "quiet_spans_s": [[1.0, 1.8]], "clicks": 0, "tonal_events": [], "clipped_pct": 0}
    ask = lambda note: extract(RulesProvider().complete("", f"Note: {note}\nAnalysis: x\nAnalysis JSON: {json.dumps(a)}"))[0]  # noqa: E731
    assert "denoise" not in ask("an old 1920s recording, please restore it")
    assert "denoise(mode: mmse, reduce_db: 8, noise_db: -3)" in ask("there is hiss on this old recording, remove the hiss")
