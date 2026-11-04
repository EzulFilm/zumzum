"""The 2026-11-04 review: each note goal takes its own dB amount, example downloads and model-server replies are
bounded, and the rules driver reads a note over several lines."""

import numpy as np
import pytest

SR = 48000


def test_each_note_goal_takes_its_own_amount():
    from zumzum.notes import note_goals
    a = {"quiet_spans_s": [[1.0, 2.0]], "hum": {"f0_hz": 60.0}}
    _, checks, _ = note_goals("reduce the hum by 12 dB and the hiss by 6 dB", a)
    assert any("hum_drop" in c and "min_db: 12" in c for c in checks)
    assert any("note_hiss" in c and "min_db: 6" in c for c in checks)
    _, checks, _ = note_goals("drums down about 9 dB", a)  # one amount still applies as before
    assert any("target_db: -9" in c for c in checks)


def test_fetch_refuses_an_oversized_download(tmp_path, monkeypatch):
    import json
    from tools import fetch_audio as F

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b"x" * n  # an endless body

    src = tmp_path / "sources.json"
    src.write_text(json.dumps({"files": {"big": {"url": "https://example.org/big.wav", "sha256": "0" * 64}}}))
    monkeypatch.setattr(F, "SOURCES", src)
    monkeypatch.setattr(F, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(F, "MAX_BYTES", 1000)
    monkeypatch.setattr(F.urllib.request, "urlopen", lambda req, timeout: Reply())
    with pytest.raises(SystemExit, match="more than 1000 bytes"):
        F.fetch()
    assert not (tmp_path / "cache" / "big.wav").exists()


def test_rules_driver_reads_a_note_over_several_lines():
    from zumzum.analysis import analyze
    from zumzum.llm import RulesProvider, _user_prompt
    a = analyze(np.atleast_2d(0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)), SR)
    user = _user_prompt("Please clean this recording.\nThere is loud hum throughout.", a)
    assert "dehum" in RulesProvider().complete("", user)


def test_model_server_reply_is_bounded(monkeypatch):
    import urllib.request
    from zumzum import llm

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b" " * n

    monkeypatch.setattr(llm, "MAX_REPLY", 1000)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: Reply())
    with pytest.raises(ValueError, match="over 1000 bytes"):
        llm.OpenWeightProvider(url="http://127.0.0.1:9/v1").complete("s", "u")




def test_sheet_waveform_fits_a_louder_after(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from zumzum import sheets
    sr = 16000
    b = 0.01 * np.random.default_rng(4).standard_normal(sr)
    figs = []
    monkeypatch.setattr(plt, "close", lambda fig: figs.append(fig))
    sheets.sheet(tmp_path / "s.png", b, 50 * b, sr, "t", [], "p")
    lo, hi = figs[0].axes[0].get_ylim()
    assert hi >= np.abs(50 * b).max() and lo <= -np.abs(50 * b).max()
