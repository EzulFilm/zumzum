import numpy as np

from zumzum.checker import check
from zumzum.llm import SYSTEM, RulesProvider, ScriptedProvider, edit_loop, extract
from zumzum.notes import merge, note_checks

from .test_engine import SR, speechlike


def hummy():
    x = speechlike(dur=4)
    t = np.arange(len(x)) / SR
    return x + sum(0.02 / k * np.sin(2 * np.pi * 60 * k * t) for k in range(1, 9))


def test_reference_lists_every_call():
    for name in ("box(", "harmonics(", "dehum(", "heal(", "split_hp(", "hum_drop(", "untouched("):
        assert name in SYSTEM


def test_extract():
    prog, msg = extract("blah\n```zumzum\nzumzum 0.1\nlayer m = pass\n```\nmessage: nothing to do\n")
    assert prog == "zumzum 0.1\nlayer m = pass\n" and msg == "nothing to do"


def test_note_checks_merge_and_check():
    sels, checks = note_checks("remove the hum, keep the voice as it is", {"hum": {"f0_hz": 60.1}})
    src = merge("zumzum 0.1\nlayer main = pass\ncheck note_peak = peak(max_dbfs: 0)\n", sels, checks)
    p = check(src)
    names = [c["name"] for c in p.checks]
    assert names.count("note_peak") == 1 and "note_hum" in names and "note_voice" in names


def test_loop_revises_after_a_refusal_and_a_failed_check():
    bad = "```zumzum\nzumzum 0.1\nnode h = dehum(f0_hz: 600)\nlayer main = h\n```\nmessage: oops"
    weak = "```zumzum\nzumzum 0.1\nnode h = dehum(f0_hz: 60, harmonics: 2)\nlayer main = h\n```\nmessage: two"
    good = "```zumzum\nzumzum 0.1\nnode h = dehum(f0_hz: 60, harmonics: 12)\nlayer main = h\n```\nmessage: twelve"
    r = edit_loop("remove the hum", hummy(), SR, ScriptedProvider([bad, weak, good]), max_iters=3,
                  log=lambda *_: None)
    v = r["versions"]
    assert "outside" in v[0]["problems"][0]
    assert not v[1]["done"] and v[2]["done"] and r["done"]
    assert "harmonics: 12" in v[2]["diff"]


def test_rules_driver_finishes_the_hum_note():
    r = edit_loop("remove the hum, keep the voice as it is", hummy(), SR, RulesProvider(), max_iters=1,
                  log=lambda *_: None)
    assert r["done"], r["versions"][-1].get("checks")


def test_merge_drops_model_names_in_the_reserved_prefix():
    sels, checks = note_checks("remove the hum", {"hum": {"f0_hz": 60.0}})
    model = ("zumzum 0.1\nsel note_x = all()\nnode note_hum = gain(db: 0)\nlayer note_l = pass\n"
             "node g = gain(db: 0)\nlayer main = g\ncheck note_peak = peak()\n")
    src = merge(model, sels, checks)
    for kind in ("sel note_x", "node note_hum", "layer note_l"):
        assert kind not in src
    check(src)  # no name collisions left


def test_rules_revision_keeps_the_previous_program():
    prev = "zumzum 0.1\nnode keep = gain(db: -3)\nlayer main = keep\n"
    user = ("Note: remove the hum\nAnalysis: x\nAnalysis JSON: {\"hum\": {\"f0_hz\": 60}}\n\nYour previous program:\n"
            "```zumzum\n" + prev + "```\n\nWhat happened:\nFAIL x\n")
    prog, _ = extract(RulesProvider().complete("", user))
    p = check(prog)
    assert "keep" in p.ops and "h" in p.ops
    assert p.layers == [("main", ["keep", "h"])]
    prog2, _ = extract(RulesProvider().complete("", user.replace(prev, prog)))
    assert check(prog2).layers == [("main", ["keep", "h"])]  # a second revision adds nothing twice


def test_feedback_names_each_nodes_effect():
    from zumzum.llm import node_effects
    text = node_effects({"main.h": {"inside_db": -24.1, "outside_db": 0.0}, "main.z": {"off": True}})
    assert "main.h: inside its region -24.1 dB, outside +0.0 dB" in text and "main.z: off" in text


def _revise(prev):
    user = ("Note: remove the hum\nAnalysis: x\nAnalysis JSON: {\"hum\": {\"f0_hz\": 60}}\n\nYour previous program:\n"
            "```zumzum\n" + prev + "```\n\nWhat happened:\nFAIL x\n")
    prog, _ = extract(RulesProvider().complete("", user))
    return check(prog)


def test_rules_revision_edits_the_layer_that_is_the_output():
    p = _revise("zumzum 0.1\nnode keep = gain(db: -3)\nlayer other = keep\n")  # one layer, not named main
    assert p.layers == [("other", ["keep", "h"])] and p.output == "other"
    p = _revise("zumzum 0.1\nnode a = gain(db: -3)\nnode b = gain(db: -6)\nlayer x = a\nlayer other = b\noutput other\n")
    assert dict(p.layers) == {"x": ["a"], "other": ["b", "h"]} and p.output == "other"
    p = _revise("zumzum 0.1\nnode a = gain(db: -3)\nlayer x = a\nlayer y = pass\noutput mix(x: 0, y: -6)\n")
    assert dict(p.layers) == {"x": ["a", "h"], "y": ["h"]} and p.output == "mix"


def test_unsupported_request_is_never_done():
    model = "```zumzum\nzumzum 0.1\nlayer main = pass\ncheck vocals_gone = drop(at: everything, min_db: 20)\n" \
            "sel everything = all()\n```\nmessage: x"
    r = edit_loop("remove the vocals", hummy(), SR, ScriptedProvider([model]), max_iters=1, log=lambda *_: None)
    assert r["status"] == "unsupported" and not r["done"]


def test_a_failing_author_check_keeps_a_version_from_done():
    good = "```zumzum\nzumzum 0.1\nsel a = all()\nnode h = dehum(f0_hz: 60, harmonics: 12)\nlayer main = h\n" \
           "check mine = drop(at: a, min_db: 60)\n```\nmessage: x"
    r = edit_loop("remove the hum", hummy(), SR, ScriptedProvider([good]), max_iters=1, log=lambda *_: None)
    v = r["versions"][-1]
    assert any(c["name"] == "note_hum" and c["pass"] for c in v["checks"])
    assert v["status"] == "not_done" and not r["done"]


def test_rules_match_whole_words_only():
    from zumzum.notes import note_goals
    for note in ("Bring the drums down", "during the song", "a string quartet", "the dialogue is clear, keep it"):
        _, _, goals = note_goals(note, {})
        assert "tones" not in [g["goal"] for g in goals], note
    _, _, goals = note_goals("remove the phone ringing at 0:09", {})
    assert [g["goal"] for g in goals] == ["tones"] and goals[0]["status"] == "unmeasurable"


def test_every_shipped_note_has_the_goals_it_asks_for():
    from tools.live_run import NOTES
    from zumzum.notes import note_goals
    a = {"hum": {"f0_hz": 60.0}, "quiet_spans_s": [[1.0, 1.5]], "clicks": 30,
         "tonal_events": [{"t_s": [9.3, 9.5], "f_hz": 1209.0, "above_db": 30}]}
    want = {"hum_voice": ["hum", "keep voice"], "cylinder_denoise": ["noise", "keep voice"],
            "record_declick": ["clicks", "keep music"], "phone_heal": ["tones", "keep voice"],
            "drums_down": ["drums", "keep music"]}
    for name, note in NOTES.items():
        _, checks, goals = note_goals(note, a)
        assert sorted(g["goal"] for g in goals) == sorted(want[name]), (name, goals)
        check("zumzum 0.1\nlayer main = pass\n" + "\n".join(_sel_lines(note, a) + checks) + "\n")


def _sel_lines(note, a):
    from zumzum.notes import note_goals
    return note_goals(note, a)[0]


def test_requested_amount_becomes_a_two_sided_check():
    from zumzum.notes import note_goals
    _, checks, _ = note_goals("Bring the drums down about 12 dB", {})
    assert "check note_drums = change(at: note_sel_cymbals, target_db: -12, tol_db: 3)" in checks
    _, checks, _ = note_goals("reduce the hum by 3 dB", {"hum": {"f0_hz": 60}})
    assert "min_db: 3)" in [c for c in checks if "note_hum" in c][0]


def test_rules_revision_keeps_multi_line_calls_intact():
    from pathlib import Path
    prev = (Path(__file__).resolve().parents[1] / "examples" / "cylinder_denoise.zz").read_text(encoding="utf-8")
    user = ("Note: remove the hum\nAnalysis: x\nAnalysis JSON: {\"hum\": {\"f0_hz\": 60}}\n\nYour previous program:\n"
            "```zumzum\n" + prev + "```\n\nWhat happened:\nFAIL x\n")
    prog, _ = extract(RulesProvider().complete("", user))
    p = check(prog)
    assert p.ops["d"]["params"]["gate_db"] == 8.0 and "h" in p.ops
    assert p.layers == [("main", ["d", "hs", "h"])]
