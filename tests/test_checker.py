import random
from pathlib import Path

import pytest

from zumzum.checker import check
from zumzum.errors import CheckError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.zz"))
OK = "zumzum 0.1\nnode g = gain(db: -6)\nlayer main = g\n"


def problems(src, **kw):
    with pytest.raises(CheckError) as e:
        check(src, **kw)
    return "\n".join(e.value.problems)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_check(path):
    prog = check(path.read_text(encoding="utf-8"))
    assert prog.layers and prog.checks


def test_minimal():
    p = check(OK)
    assert p.output == "main" and p.ops["g"]["params"]["db"] == -6.0 and p.size == 2048


@pytest.mark.parametrize("src,msg", [
    ("zumzum 0.2\nlayer m = pass\n", "reads zumzum 0.1"),
    ("zumzum 0.1\nnode g = boost(db: 3)\nlayer m = g\n", "unknown op 'boost'"),
    ("zumzum 0.1\nnode g = gain(db: 30)\nlayer m = g\n", "outside [-120, 24]"),
    ("zumzum 0.1\nnode g = gain()\nlayer m = g\n", "needs db"),
    ("zumzum 0.1\nnode g = gain(db: 1, db: 2)\nlayer m = g\n", "given twice"),
    ("zumzum 0.1\nnode g = gain(gain: 1)\nlayer m = g\n", "no parameter 'gain'"),
    ("zumzum 0.1\nnode h = heal()\nlayer m = h\n", "needs 'at"),
    ("zumzum 0.1\nsel b = band(f_hz: 1..2)\nnode h = heal(at: b)\nlayer m = h\n", "goes after the call"),
    ("zumzum 0.1\nnode g = gain(db: 1) at nope\nlayer m = g\n", "no sel named 'nope'"),
    ("zumzum 0.1\nsel a = b\nsel b = all()\nlayer m = pass\n", "no sel named 'b' before"),
    ("zumzum 0.1\nsel b = band(f_hz: 5..1)\nlayer m = pass\n", "lo < hi"),
    ("zumzum 0.1\nnode g = gain(db: 1)\nnode g = gain(db: 2)\nlayer m = g\n", "already used"),
    ("zumzum 0.1\nnode g = gain(db: 1)\nlayer m = h\n", "no node named 'h'"),
    ("zumzum 0.1\nlayer a = pass\nlayer b = pass\n", "say what the output is"),
    ("zumzum 0.1\n", "at least one layer"),
    ("zumzum 0.1\nnode g = gain(db: 1) mix 2\nlayer m = g\n", "mix 2 is outside"),
    ("zumzum 0.1\nnode d = denoise(mode: magic)\nlayer m = d\n", "must be one of mmse, logmmse, wiener, subtract"),
    ("zumzum 0.1\nlayer m = pass\ncheck c = drop(at: x, min_db: 3)\n", "no sel named 'x'"),
    ("zumzum 0.1\nlayer m = pass\ncheck c = peak() on other\n", "no layer named 'other'"),
    ("zumzum 0.1\nlayer m = pass\nexec \"rm -rf /\"\n", "unknown statement 'exec'"),
    ("zumzum 0.1\nname \"a\"\nname \"b\"\nlayer m = pass\n", "more than once"),
    ("zumzum 0.1\nnode pass = gain(db: 1)\n", "reserved word"),
    ("zumzum 0.1\nnode g = gain(db: 1e999)\nlayer m = g\n", "out of range"),
])
def test_refusals(src, msg):
    assert msg in problems(src)


def test_spans_against_the_audio():
    src = "zumzum 0.1\nsel s = time(t_s: 50..60)\nsel b = band(f_hz: 30000..40000)\nlayer m = pass\n"
    msg = problems(src, duration=10.0, sr=48000)
    assert "after the end of the audio" in msg and "above the Nyquist" in msg


def test_all_problems_reported_at_once():
    src = "zumzum 0.1\nnode a = gain(db: 99)\nnode b = dehum(f0_hz: 1)\nlayer m = a -> b -> c\n"
    msg = problems(src)
    assert "outside [-120, 24]" in msg and "f0_hz" in msg and "no node named 'c'" in msg


def test_limits():
    many = "zumzum 0.1\n" + "".join(f"node g{i} = gain(db: 0)\n" for i in range(40)) + "layer m = g0\n"
    assert "more than 32 node" in problems(many)
    deep = "zumzum 0.1\nsel s = " + "not " * 20 + "all()\nlayer m = pass\n"
    assert "nested more than 16" in problems(deep)
    assert "32 KiB" in problems("zumzum 0.1\n" + "#" * 40000)


def test_fuzz_never_crashes():
    """Mutated programs and random text only ever give a checked program or a CheckError."""
    rng = random.Random(7)
    seeds = [p.read_text(encoding="utf-8") for p in EXAMPLES] + [OK]
    alphabet = "abcdefghijklmnopqrstuvwxyz_0123456789 .,:()=-><#\"\n\t"
    for i in range(1500):
        s = list(rng.choice(seeds))
        for _ in range(rng.randint(1, 6)):
            k = rng.randrange(len(s))
            op = rng.random()
            if op < 0.4:
                s[k] = rng.choice(alphabet)
            elif op < 0.7:
                del s[k]
            else:
                s.insert(k, rng.choice(alphabet))
        text = "".join(s) if i % 10 else "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200)))
        try:
            check(text)
        except CheckError:
            pass


def test_nodes_off_and_output_mix():
    src = """zumzum 0.1
node a = gain(db: -6)
node b = gain(db: -20) off
layer x = a -> b
layer y = pass
output mix(x: 0, y: -6)
"""
    p = check(src)
    assert p.ops["b"]["off"] and not p.ops["a"]["off"]
    assert p.output == "mix" and p.output_mix == {"x": 0.0, "y": -6.0}
    assert "no layer named 'z'" in problems("zumzum 0.1\nlayer x = pass\noutput mix(z: 0)\n")
    assert "gain in dB" in problems("zumzum 0.1\nlayer x = pass\noutput mix(x: 99)\n")


def test_time_ops_take_time_only_selections():
    msg = problems("zumzum 0.1\nsel b = band(f_hz: 1000..4000)\nnode c = declick() at b\nlayer m = c\n")
    assert "time-only" in msg
    check("zumzum 0.1\nsel t = time(t_s: 1..2)\nnode c = declick() at t\nlayer m = c\n")


def test_change_check():
    from zumzum.engine import render
    from zumzum.measure import run_checks
    import numpy as np
    x = np.random.default_rng(1).standard_normal(48000) * 0.1
    p = check("zumzum 0.1\nsel a = all()\nnode g = gain(db: -12)\nlayer m = g\n"
              "check near = change(at: a, target_db: -12, tol_db: 1)\ncheck far = change(at: a, target_db: -3)\n")
    outs, _ = render(p, x, 48000)
    r = {c["name"]: c for c in run_checks(p, np.atleast_2d(x), outs, 48000)}
    assert r["near"]["pass"] and not r["far"]["pass"]
