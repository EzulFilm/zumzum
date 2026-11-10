"""The attenuate node: the plain clamp min(1, |S| / |X|) per cell and the projection rounds; the attenuated check."""

import numpy as np

from zumzum.checker import check
from zumzum.engine import render
from zumzum.measure import run_checks

SR = 16000
CHECK = "check att = attenuated(max_over_db: 1, floor_db: -40, max_excess_db: -30)\n"


def sig():
    rng = np.random.default_rng(0)
    t = np.arange(3 * SR) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(t.size))[None, :]


def run(text, x):
    p = check(text, duration=x.shape[1] / SR, sr=SR)
    outs, info = render(p, x, SR)
    return outs, info, run_checks(p, x, outs, SR)


def test_gain_never_over_one_and_a_louder_estimate_gives_back_the_input():
    x = sig()
    outs, info, res = run("zumzum 0.1\nnode g = gain(db: 6)\nnode a = attenuate(est: est)\nlayer est = g\n"
                          "layer main = a\noutput main\n" + CHECK, x)
    assert info["main.a"]["max_gain"] <= 1.0
    assert np.max(np.abs(outs["main"] - x)) < 1e-9  # every cell capped at the input: the input comes back
    assert res[0]["pass"]


def test_a_denoise_estimate_only_turns_down():
    x = sig()
    outs, info, res = run("zumzum 0.1\nnode d = denoise(mode: mmse, reduce_db: 12)\nnode a = attenuate(est: est)\n"
                          "layer est = d\nlayer main = a\noutput main\n" + CHECK, x)
    assert info["main.a"]["max_gain"] <= 1.0 and info["main.a"]["kept_db"] < 0
    assert res[0]["pass"], res[0]["text"]


def test_the_check_fails_a_program_that_adds_sound():
    _, _, res = run("zumzum 0.1\nnode g = gain(db: 3)\nlayer main = g\n" + CHECK, sig())
    assert not res[0]["pass"] and res[0]["value"] > 2.9
