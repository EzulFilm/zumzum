"""Note rules: phone tones, phrases not single words, hum at an unknown mains frequency; the file driver."""

import numpy as np
import pytest

from zumzum.analysis import analyze
from zumzum.llm import FileProvider, NeedAnswer
from zumzum.notes import note_goals

SR = 44100


def test_file_provider_writes_the_prompt_then_reads_the_answer(tmp_path):
    p = FileProvider(tmp_path / "x")
    with pytest.raises(NeedAnswer):
        p.complete("sys", "user 1")
    assert "user 1" in (tmp_path / "x" / "v1_prompt.txt").read_text()
    (tmp_path / "x" / "v1_answer.txt").write_text("```zumzum\nzumzum 0.1\nlayer main = pass\n```\n")
    q = FileProvider(tmp_path / "x")  # a re-run starts again from turn 1
    assert "layer main = pass" in q.complete("sys", "user 1")


EVENTS = [{"t_s": [9.0, 9.25], "f_hz": 441.4, "above_db": 27},     # dial tone (440)
          {"t_s": [9.28, 9.46], "f_hz": 1475.0, "above_db": 37},   # keypad column 1477
          {"t_s": [10.76, 10.9], "f_hz": 269.2, "above_db": 40},   # a held vowel's harmonic
          {"t_s": [11.06, 11.2], "f_hz": 602.9, "above_db": 34},   # 2.8 % off 620: voice
          {"t_s": [12.67, 12.81], "f_hz": 452.2, "above_db": 37}]  # 2.8 % off 440: voice


def test_a_phone_note_keeps_only_telephone_frequencies():
    sels, _, goals = note_goals("Someone dials a phone at about 0:09. Remove the dial tones.", {"tonal_events": EVENTS})
    sel = next(s for s in sels if s.startswith("sel note_sel_tones"))
    assert "1415..1535" in sel and "381.4..501.4" in sel  # 1475 and 441.4, +-60 Hz
    assert "209.2" not in sel and "542.9" not in sel and "392.2" not in sel
    assert goals[0]["goal"] == "tones" and "2 telephone-frequency" in goals[0]["why"]


def test_a_phone_note_with_only_voice_tones_is_unmeasurable_not_done():
    _, _, goals = note_goals("remove the phone ringing at 0:10", {"tonal_events": EVENTS[2:]})
    assert goals[0]["status"] == "unmeasurable"


def test_other_tone_notes_keep_every_event():
    sels, _, _ = note_goals("remove the whistle at 0:09", {"tonal_events": EVENTS})
    sel = next(s for s in sels if s.startswith("sel note_sel_tones"))
    assert "209.2" in sel and "1415..1535" in sel


def test_analysis_lists_more_than_16_tonal_events():
    rng = np.random.default_rng(0)
    n = 20 * SR
    x = 0.003 * rng.standard_normal(n)
    t = np.arange(int(0.3 * SR)) / SR
    for k in range(24):  # 24 separate 0.3 s tones, ranked by level; the weakest ones are what a time window may need
        i = int((0.5 + 0.8 * k) * SR)
        x[i:i + t.size] += 0.02 * (1 + k / 8) * np.sin(2 * np.pi * (500 + 97 * k) * t)
    ev = analyze(x[None, :], SR)["tonal_events"]
    assert len(ev) > 16


def _filled(after_fn):
    from zumzum.checker import check
    from zumzum.measure import run_checks
    rng = np.random.default_rng(3)
    x = 0.05 * rng.standard_normal(4 * SR)  # a steady bed (the "voice" under the tone)
    t = np.arange(x.size) / SR
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t) * ((t > 2.0) & (t < 2.3))
    before = (x + tone)[None, :]
    prog = check("zumzum 0.1\nsel b = box(t_s: 2.0..2.3, f_hz: 900..1100)\nlayer main = pass\n"
                 "check f = filled(at: b, max_below_db: 6)\n", duration=4, sr=SR)
    return run_checks(prog, before, {"main": after_fn(x, tone)[None, :]}, SR)[0]


def test_filled_passes_a_tone_removed_down_to_its_surroundings():
    r = _filled(lambda bed, tone: bed)  # the tone gone, the bed under it kept
    assert r["pass"] and abs(r["value"]) < 2, r["text"]


def test_filled_fails_a_heal_that_also_took_what_was_under_the_tone():
    from scipy.signal import stft, istft
    def carve(bed, tone):
        f, t, Z = stft(bed, SR, nperseg=2048)
        Z[np.ix_((f > 900) & (f < 1100), (t > 2.0) & (t < 2.3))] *= 0.01  # the box emptied
        return istft(Z, SR, nperseg=2048)[1][:bed.size]
    r = _filled(carve)
    assert not r["pass"] and r["value"] > 6, r["text"]


def test_phone_note_with_keep_voice_adds_the_filled_check_and_drums_band_starts_at_2k():
    _, checks, goals = note_goals("Someone dials a phone at about 0:09. Remove the dial tones, keep the voice.",
                                  {"tonal_events": EVENTS})
    assert "check note_voice_in_tones = filled(at: note_sel_tones, max_below_db: 6)" in checks
    assert "note_voice_in_tones" in next(g for g in goals if g["goal"] == "tones")["checks"]
    sels, _, _ = note_goals("Bring the drums down about 12 dB", {})
    assert "sel note_sel_cymbals = band(f_hz: 2000..16000)" in sels


def test_filled_with_max_above_fails_a_tone_left_in():
    r = _filled(lambda bed, tone: bed + tone)  # nothing removed: the tone still stands over its surroundings
    assert r["pass"]  # max_above_db defaults to 200: only the "under" side is checked
    from zumzum.checker import check
    from zumzum.measure import run_checks
    rng = np.random.default_rng(3)
    x = 0.05 * rng.standard_normal(4 * SR)
    t = np.arange(x.size) / SR
    before = (x + 0.3 * np.sin(2 * np.pi * 1000 * t) * ((t > 2.0) & (t < 2.3)))[None, :]
    prog = check("zumzum 0.1\nsel b = box(t_s: 2.0..2.3, f_hz: 900..1100)\nlayer main = pass\n"
                 "check f = filled(at: b, max_above_db: 3, max_below_db: 200)\n", duration=4, sr=SR)
    res = run_checks(prog, before, {"main": before}, SR)[0]
    assert not res["pass"] and res["value"] < -3, res["text"]


def test_no_hum_found_measures_the_mains_series_present():
    _, checks, goals = note_goals("There's an electrical buzz from the mains. Please get rid of it.", {"hum": None})
    assert "check note_hum = mains_hum_drop(harmonics: 10, min_db: 20)" in checks
    assert "stronger in the input" in goals[0]["why"]
    _, checks, _ = note_goals("remove the hum", {"hum": {"f0_hz": 49.97}})
    assert [c for c in checks if "hum_drop" in c] == ["check note_hum = hum_drop(f0_hz: 50, harmonics: 10, min_db: 20)"]
    _, checks, _ = note_goals("remove the 60 Hz hum", {"hum": None})
    assert [c for c in checks if "hum_drop" in c] == ["check note_hum = hum_drop(f0_hz: 60, harmonics: 10, min_db: 20)"]


def test_mains_hum_drop_measures_the_series_that_is_there():
    from zumzum.checker import check
    from zumzum.engine import render
    from zumzum.measure import run_checks
    rng = np.random.default_rng(2)
    t = np.arange(4 * SR) / SR
    x = (0.05 * rng.standard_normal(t.size) + sum(0.02 / k * np.sin(2 * np.pi * 50 * k * t) for k in range(1, 11)))[None]
    for f0, ok in ((50, True), (60, False)):
        p = check(f"zumzum 0.1\nnode h = dehum(f0_hz: {f0}, harmonics: 12, depth_db: 50)\nlayer main = h\n"
                  "check m = mains_hum_drop(harmonics: 10, min_db: 20)\n", duration=4, sr=SR)
        r = run_checks(p, x, render(p, x, SR)[0], SR)[0]
        assert abs(r["f0_hz"] - 50) < 1 and r["pass"] == ok, r["text"]


def test_phrases_not_single_words_buzz_and_tone():
    goals = lambda note: [g["goal"] for g in note_goals(note, {"tonal_events": EVENTS, "hum": None})[2]]  # noqa: E731
    assert "hum" not in goals("the guitar has a brittle buzz on the loud notes")
    assert "hum" in goals("there is an electrical buzz under everything")
    assert "hum" in goals("please remove the hum")
    assert "tones" not in goals("the tone of the voice is harsh, warm it up")
    assert "tones" in goals("someone dials a phone at 0:09, remove the dial tones")
    assert "tones" in goals("there are beeps at 0:10")


def test_ringing_out_is_music_not_a_tone():
    goals = lambda note: [g["goal"] for g in note_goals(note, {"tonal_events": EVENTS, "hum": None})[2]]  # noqa: E731
    assert "tones" not in goals("loud piano passages distort instead of ringing out naturally")
    assert "tones" in goals("a phone is ringing at 0:10, please remove it")
    assert "tones" in goals("there are beeps around 0:05")


def test_hum_text_says_which_way_it_passed():
    from zumzum.measure import _hum_text
    assert _hum_text("hum", 31.0, 30, True) == "hum went down 31.0 dB (needed >= 30 dB)"
    t = _hum_text("hum", 25.1, 30, True)
    assert "no longer above the neighbouring bins, which passes" in t and "needed" not in t
    assert "still stands above" in _hum_text("hum", 5.0, 30, False)
