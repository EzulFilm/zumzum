"""The editing loop: note + analysis numbers -> program (by a model or a rules driver) -> checker -> reference engine
-> measured checks -> revision. The model never hears the audio, and nothing it writes is executed: its reply is
text that goes through the checker like any other input.

Drivers (ZUMZUM_DRIVER): `openweight` (an OpenAI-compatible /chat/completions server, e.g. llama.cpp's
llama-server on 127.0.0.1, at ZUMZUM_LLM_URL), `rules` (deterministic, no model), or a ScriptedProvider in tests.
There is no paid-API driver; any OpenAI-compatible server can be used through `openweight`.
"""

import difflib
import json
import os
import re
import time

import numpy as np

from .lang import statements

from .analysis import analyze, summary
from .audio import validate
from .checker import check
from .engine import render
from .errors import CheckError
from .measure import damage, run_checks
from .notes import merge, note_goals, request_status
from .ops import CHECKS, OPS, OPTIONAL, SELS


def _param_text(key, p):
    kind, lo, hi, default, text = p
    if kind == "enum":
        shape = "|".join(str(c) for c in lo)
    elif kind == "range":
        shape = f"lo..hi in [{lo:g}, {hi:g}]"
    elif kind == "sel":
        shape = "a sel name"
    else:
        shape = f"{'int' if kind == 'int' else 'num'} {lo:g}..{hi:g}"
    d = " required" if default is None else " optional" if default == OPTIONAL else f" ={default}"
    return f"{key}: {shape}{d}"


def language_reference():
    out = ["Language: zumzum 0.1. One statement per line. Units are in parameter names (_s, _ms, _hz, _db).",
           "  zumzum 0.1",
           '  name "..."',
           "  analysis stft(size: 2048)                      # optional; FFT size for the whole program",
           "  sel NAME = CALL | sel and sel | sel or sel | not sel | (sel) | NAME   # a time-frequency region",
           "  node NAME = OP(...) [at SEL] [mix 0..1] [off]   # one op, optionally inside a region; off = bypass",
           "  layer main = NODE1 -> NODE2                    # nodes in order, from the input (or: layer main = pass)",
           "  output main | output mix(a: 0, b: -6)          # several layers: name one, or mix them (gain dB)",
           "  check NAME = CALL                              # a measurement on before / after",
           "Selections:"]
    for k, v in SELS.items():
        out.append(f"  {k}({', '.join(_param_text(a, p) for a, p in v['params'].items())})  - {v['about']}")
    out.append("Ops:")
    for k, v in OPS.items():
        at = " [needs at]" if v.get("needs_at") else ""
        out.append(f"  {k}({', '.join(_param_text(a, p) for a, p in v['params'].items())}){at}  - {v['about']}")
    out.append("Checks:")
    for k, v in CHECKS.items():
        out.append(f"  {k}({', '.join(_param_text(a, p) for a, p in v['params'].items())})  - {v['about']}")
    return "\n".join(out)


EXAMPLE = """```zumzum
zumzum 0.1
name "Hum under a voice"
sel voice = band(f_hz: 300..3400)
node h = dehum(f0_hz: 60, harmonics: 12, depth_db: 40)
layer main = h
check voice_kept = keep(at: voice, tol_db: 1)
```
message: cut the 60 Hz hum and 12 harmonics, voice band checked."""

SYSTEM = ("You edit audio by writing short programs in a small language. You never hear the audio: you get a "
          "person's note and numbers measured on the audio. Answer with exactly one ```zumzum code block holding the "
          "whole program, then one line 'message: ...' saying what you did in plain words. Use only the calls "
          "listed. Times are seconds from the start of the file.\n\n" + language_reference() +
          "\n\nExample answer:\n" + EXAMPLE)


def extract(text):
    m = re.search(r"```(?:zumzum|zz)?\s*\n(.*?)```", text, re.S)
    prog = (m.group(1) if m else text).strip() + "\n"
    msg = re.search(r"message:\s*(.+)", text[m.end():] if m else "")
    return prog, (msg.group(1).strip() if msg else "")


class OpenWeightProvider:
    """llama.cpp server (or any OpenAI-compatible endpoint). Standard library only; greedy decoding (temperature 0,
    fixed seed) for repeatable answers."""

    def __init__(self, url=None, model=None, max_tokens=900, timeout=None):
        self.url = (url or os.environ.get("ZUMZUM_LLM_URL") or "").rstrip("/")
        if not self.url:
            raise ValueError("the open-weight driver needs ZUMZUM_LLM_URL (e.g. http://127.0.0.1:8091/v1)")
        self.model = model or os.environ.get("ZUMZUM_MODEL") or "qwen2.5-7b-instruct-q4_k_m"
        self.timeout = float(timeout or os.environ.get("ZUMZUM_LLM_TIMEOUT") or 1800)
        self.max_tokens, self.last_usage = max_tokens, {}

    def complete(self, system, user):
        import urllib.request
        body = json.dumps({"model": self.model, "temperature": 0, "seed": 1, "max_tokens": self.max_tokens,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        u = data.get("usage") or {}
        self.last_usage = {"model": data.get("model") or self.model, "input_tokens": u.get("prompt_tokens", 0),
                           "output_tokens": u.get("completion_tokens", 0),
                           "seconds": round(time.monotonic() - t0, 1)}
        return data["choices"][0]["message"]["content"] or ""


class RulesProvider:
    """No model: keyword rules over the note and the analysis numbers. A floor for the loop and a test fixture."""
    last_usage = {"model": "rules"}

    def complete(self, system, user):
        note = re.search(r"Note: (.*)", user).group(1).lower()
        a = json.loads(re.search(r"Analysis JSON: (.*)", user).group(1))
        ops = []
        if "hum" in note or "buzz" in note:
            f0 = (a.get("hum") or {}).get("f0_hz", 60)
            ops.append(f"node h = dehum(f0_hz: {round(f0)}, harmonics: 16, depth_db: 50)")
        if "click" in note or "crackle" in note:
            ops.append("node c = declick()")
        if "hiss" in note or "noise" in note:
            ops.append("node d = denoise(mode: mmse, reduce_db: 12)")
        if "drum" in note:
            ops.append("node p = split_hp(part: harmonic, rest_db: -12, time_ms: 800, freq_hz: 1000)")
        if not ops:
            ops.append("node g = gain(db: 0)")
        prev = re.search(r"Your previous program:\n```zumzum\n(.*?)```", user, re.S)
        if prev:  # a revision: keep everything already there, add only the rule nodes it does not have yet
            text = prev.group(1).rstrip()
            lines = text.splitlines()
            stmts = statements(text)
            declared = {re.match(r"\s*\w+\s+(\w+)", lines[a]).group(1) for w, a, b in stmts
                        if w in ("sel", "node", "layer", "check") and re.match(r"\s*\w+\s+(\w+)", lines[a])}
            new = [o for o in ops if o.split()[1] not in declared]
            names = [o.split()[1] for o in new]
            lines = _add_to_output(lines, names)
            after = max([b for w, a, b in stmts if w == "node"] or [0])  # after the last line of the last node
            prog = "\n".join(lines[:after + 1] + new + lines[after + 1:]) + "\n"
            return f"```zumzum\n{prog}```\nmessage: rules driver kept the previous program, added: {', '.join(names) or 'nothing'}"
        names = [o.split()[1] for o in ops]
        prog = "zumzum 0.1\n" + "\n".join(ops) + f"\nlayer main = {' -> '.join(names)}\n"
        return f"```zumzum\n{prog}```\nmessage: rules driver: {', '.join(names)}"


def _layers(lines):
    """{layer name: line index} of a program's layer statements."""
    out = {}
    for i, ln in enumerate(lines):
        m = re.match(r"\s*layer\s+(\w+)\s*=", ln)
        if m:
            out[m.group(1)] = i
    return out


def _output_layers(lines, layers):
    """The layers that reach the output: the one `output` names, every layer of an `output mix(...)`, the only layer,
    or none (then a new `main` layer is made)."""
    for ln in lines:
        m = re.match(r"\s*output\s+mix\s*\((.*)\)", ln)
        if m:
            return [k.strip() for k in re.findall(r"(\w+)\s*:", m.group(1)) if k.strip() in layers]
        m = re.match(r"\s*output\s+(\w+)\s*$", ln)
        if m and m.group(1) in layers:
            return [m.group(1)]
    return list(layers) if len(layers) == 1 else []


def _add_to_output(lines, names):
    """Append nodes to the end of every layer that reaches the output, so a revision changes what is heard."""
    if not names:
        return lines
    lines = list(lines)
    layers = _layers(lines)
    targets = _output_layers(lines, layers)
    if not targets:
        if layers:  # several layers and no output: the checker would refuse it anyway; leave the choice visible
            targets = [next(iter(layers))]
        else:
            return lines + [f"layer main = {' -> '.join(names)}"]
    for t in targets:
        i = layers[t]
        head, chain = lines[i].split("=", 1)
        chain = [c.strip() for c in chain.split("->") if c.strip() and c.strip() != "pass"]
        lines[i] = f"{head.rstrip()} = {' -> '.join(chain + names)}"
    return lines


class ScriptedProvider:
    def __init__(self, answers):
        self.answers, self.last_usage = list(answers), {"model": "scripted"}

    def complete(self, system, user):
        return self.answers.pop(0)


def make_provider(driver=None):
    driver = driver or os.environ.get("ZUMZUM_DRIVER") or "rules"
    if driver == "openweight":
        return OpenWeightProvider()
    if driver == "rules":
        return RulesProvider()
    raise ValueError(f"unknown ZUMZUM_DRIVER {driver!r} (openweight, rules)")


def _user_prompt(note, a, feedback=None, previous=None):
    lines = [f"Note: {note}", f"Analysis: {summary(a)}", "Analysis JSON: " + json.dumps(a)]
    if previous:
        lines += ["", "Your previous program:", "```zumzum", previous.rstrip(), "```", "", "What happened:", feedback,
                  "", "Write the whole revised program."]
    return "\n".join(lines)


def node_effects(info):
    """One line per node from the engine's per-node measurements, so a revision knows which node to change."""
    out = []
    for key, inf in info.items():
        if inf.get("off"):
            out.append(f"  {key}: off (bypassed)")
        elif "inside_db" in inf:
            outside = f", outside {inf['outside_db']:+.1f} dB" if "outside_db" in inf else " (it covers everything)"
            out.append(f"  {key}: inside its region {inf['inside_db']:+.1f} dB{outside}")
        else:
            extra = ", ".join(f"{k} {v}" for k, v in inf.items())
            out.append(f"  {key}: {extra or 'no measurement (a time-domain node)'}")
    return "\n".join(out) or "  (no nodes)"


def edit_loop(note, audio, sr, provider, max_iters=3, log=print):
    """Run the loop on one note. Returns {"done", "versions": [...]}; each version keeps the model's text, the
    program that ran (with the note's checks merged in), checker problems or check results, damage and a diff."""
    validate(audio, sr)  # bad input is the caller's problem, never fed to the model as a checker error
    a = analyze(audio, sr)
    nsels, nchecks, goals = note_goals(note, a)
    versions, prev, feedback = [], None, None
    for it in range(1, max_iters + 1):
        reply = provider.complete(SYSTEM, _user_prompt(note, a, feedback, prev))
        src, message = extract(reply)
        full = merge(src, nsels, nchecks)
        v = {"iter": it, "reply": reply, "program": src, "ran": full, "message": message,
             "usage": dict(getattr(provider, "last_usage", {}) or {}),
             "diff": "".join(difflib.unified_diff((prev or "").splitlines(True), src.splitlines(True),
                                                  "previous", "this")) if prev else ""}
        try:
            prog = check(full, duration=np.atleast_2d(audio).shape[1] / sr, sr=sr)
        except CheckError as e:
            v["problems"] = e.problems
            feedback = "The checker refused the program:\n" + "\n".join(e.problems)
            log(f"  iter {it}: checker refused ({len(e.problems)} problems)")
            versions.append(v)
            prev = src
            continue
        outs, info = render(prog, audio, sr)
        res = run_checks(prog, np.atleast_2d(audio), outs, sr)
        dmg = damage(audio, outs[prog.output])
        v.update({"checks": [{k: (bool(x) if k == "pass" else float(x) if k == "value" else x)
                              for k, x in r.items()} for r in res], "damage": dmg, "info": info,
                  "output": outs[prog.output]})
        v["status"] = request_status(goals, res, dmg)
        v["done"] = v["status"] == "done"
        log(f"  iter {it}: {v['status']}: " + ", ".join(f"{r['name']} {'ok' if r['pass'] else 'FAIL'}" for r in res)
            + (f"; damage: {dmg}" if dmg else ""))
        versions.append(v)
        if v["done"] or v["status"] in ("unsupported", "unmeasurable"):
            break  # done, or nothing a revision could change: the note itself is not measurable
        prev = src
        feedback = "\n".join(f"{'PASS' if r['pass'] else 'FAIL'} {r['name']}: {r['text']}" for r in res)
        feedback += "\nWhat each node did (energy after / before, inside and outside its region):\n" + node_effects(info)
        feedback += "".join(f"\nDAMAGE: {d}" for d in dmg)
    last = versions[-1] if versions else {}
    return {"done": bool(last.get("done")), "status": last.get("status", "refused"), "goals": goals,
            "versions": versions, "analysis": a}
