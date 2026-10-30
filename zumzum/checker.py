"""The checker: refuses anything unknown, mistyped, out of range or over a size limit, and returns a normalised
program (defaults filled in, numbers as floats, selections expanded) the engine can trust. A program that passes is
safe to run, whoever wrote it: it has no I/O, no loops, no arithmetic and bounded cost (LIMITS).

All problems are collected, each with its line, so a revision can fix them in one go.
"""

import math
from dataclasses import dataclass, field

from . import VERSION
from . import lang as L
from .errors import CheckError
from .ops import ANALYSIS, CHECKS, LIMITS, OPS, OPTIONAL, SELS


@dataclass
class Checked:
    name: str = ""
    about: str = ""
    size: int = 2048
    overlap: int = 4
    sels: dict = field(default_factory=dict)     # name -> normalised selection
    ops: dict = field(default_factory=dict)      # node name -> {"op", "params", "at", "mix", "off"}
    layers: list = field(default_factory=list)   # [(name, [op names])]
    output: str = ""                             # a layer name, or "mix" (see output_mix)
    output_mix: dict = field(default_factory=dict)  # layer -> gain dB, when the output mixes layers
    checks: list = field(default_factory=list)   # [{"name", "check", "params", "layer"}]
    warnings: list = field(default_factory=list)

    @property
    def hop(self):
        return self.size // self.overlap


class _Ctx:
    def __init__(self, duration, sr):
        self.problems, self.warnings = [], []
        self.duration, self.sr = duration, sr

    def err(self, line, msg):
        self.problems.append(f"line {line}: {msg}")


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _show(v):
    if isinstance(v, L.Ident):
        return v.name
    if isinstance(v, L.Range):
        return f"{v.lo:g}..{v.hi:g}"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


def _value(ctx, where, line, spec, v, sel_names):
    kind, lo, hi, _, _ = spec
    if kind in ("num", "int"):
        if not _is_num(v) or (kind == "int" and not isinstance(v, int)):
            ctx.err(line, f"{where}: expected {'an integer' if kind == 'int' else 'a number'}, got {_show(v)}")
            return None
        if not lo <= v <= hi:
            ctx.err(line, f"{where}: {v:g} is outside [{lo:g}, {hi:g}]")
            return None
        return int(v) if kind == "int" else float(v)
    if kind == "enum":
        choice = v.name if isinstance(v, L.Ident) else v if _is_num(v) else None
        if choice not in lo:
            ctx.err(line, f"{where}: must be one of {', '.join(str(c) for c in lo)}, got {_show(v)}")
            return None
        return choice
    if kind == "range":
        if not isinstance(v, L.Range):
            ctx.err(line, f"{where}: expected a range lo..hi, got {_show(v)}")
            return None
        if not (lo <= v.lo <= hi and lo <= v.hi <= hi):
            ctx.err(line, f"{where}: {_show(v)} is outside [{lo:g}, {hi:g}]")
            return None
        if not v.lo < v.hi:
            ctx.err(line, f"{where}: range {_show(v)} must have lo < hi")
            return None
        return (float(v.lo), float(v.hi))
    if kind == "sel":
        if not isinstance(v, L.Ident):
            ctx.err(line, f"{where}: expected the name of a sel, got {_show(v)}")
            return None
        if v.name not in sel_names:
            ctx.err(line, f"{where}: no sel named {v.name!r}" + _known(sel_names))
            return None
        return v.name
    raise AssertionError(kind)


def _known(names):
    return f" (defined: {', '.join(sorted(names))})" if names else " (none defined)"


def _params(ctx, where, call, table, sel_names=()):
    if call.name not in table:
        ctx.err(call.line, f"{where}: unknown {_kind_word(table)} {call.name!r} (known: {', '.join(table)})")
        return None
    specs = table[call.name]["params"]
    out, ok, seen = {}, True, set()
    for key, v, line in call.args:
        if key in seen:
            ctx.err(line, f"{where}: {call.name}: parameter {key!r} given twice")
            ok = False
            continue
        seen.add(key)
        if key not in specs:
            hint = (f"; the region goes after the call: op ... = {call.name}(...) at <sel>"
                    if key in ("at", "mix") and table is OPS else "")
            ctx.err(line, f"{where}: {call.name} has no parameter {key!r} (it takes: "
                          f"{', '.join(specs) or 'nothing'}){hint}")
            ok = False
            continue
        cv = _value(ctx, f"{where}: {call.name}.{key}", line, specs[key], v, sel_names)
        if cv is None:
            ok = False
        out[key] = cv
    for key, spec in specs.items():
        if key not in out:
            if spec[3] is None:
                ctx.err(call.line, f"{where}: {call.name} needs {key} ({spec[4]})")
                ok = False
            elif spec[3] != OPTIONAL:
                out[key] = spec[3]
            else:
                out[key] = None
    return out if ok else None


def _kind_word(table):
    return {id(SELS): "selection", id(OPS): "op", id(CHECKS): "check", id(ANALYSIS): "analysis"}[id(table)]


def _sel(ctx, where, e, sels, count):
    """Normalise a selection expression; named selections are expanded (they were checked when defined)."""
    if isinstance(e, L.Call):
        count[0] += 1
        p = _params(ctx, where, e, SELS)
        if p is None:
            return None
        _check_span(ctx, where, e.line, e.name, p)
        count[1] += 1
        return ("call", e.name, p)
    if isinstance(e, L.Ref):
        if e.name not in sels:
            ctx.err(e.line, f"{where}: no sel named {e.name!r} before this line" + _known(sels))
            return None
        count[1] += sels[e.name][1]
        return sels[e.name][0]
    if isinstance(e, L.Not):
        a = _sel(ctx, where, e.arg, sels, count)
        return None if a is None else ("not", a)
    parts = [_sel(ctx, where, a, sels, count) for a in e.args]
    if any(a is None for a in parts):
        return None
    return ("and" if isinstance(e, L.And) else "or", tuple(parts))


def _check_span(ctx, where, line, name, p):
    t = p.get("t_s")
    if t and ctx.duration is not None and t[0] >= ctx.duration:
        ctx.err(line, f"{where}: {name}.t_s starts at {t[0]:g} s, after the end of the audio ({ctx.duration:g} s)")
    f = p.get("f_hz")
    if f and ctx.sr is not None and f[0] >= ctx.sr / 2:
        ctx.err(line, f"{where}: {name}.f_hz starts at {f[0]:g} Hz, above the Nyquist frequency ({ctx.sr / 2:g} Hz)")


def check(src, duration=None, sr=None):
    """Check program text. `duration` (s) and `sr` (Hz), when the host knows them, catch spans outside the audio.
    Returns `Checked`; raises CheckError with every problem found."""
    prog = L.parse(src)
    ctx = _Ctx(duration, sr)
    out = Checked()
    if sr is not None and not LIMITS["sr_min"] <= sr <= LIMITS["sr_max"]:  # also refuses NaN
        ctx.err(1, f"the audio's sample rate {sr} Hz is outside {LIMITS['sr_min']}..{LIMITS['sr_max']} Hz")
    if duration is not None and not 0 <= duration <= LIMITS["duration_s"]:  # also refuses NaN
        ctx.err(1, f"the audio is {duration:g} s long; it must be 0..{LIMITS['duration_s']} s")
    if prog.version != VERSION:
        ctx.err(1, f"this checker reads zumzum {VERSION}, the program says {prog.version}")
    for word, line in prog.dup:
        ctx.err(line, f"{word!r} appears more than once")
    out.name = prog.meta.get("name", ("", 0))[0]
    out.about = prog.meta.get("about", ("", 0))[0]
    if prog.analysis is not None:
        p = _params(ctx, "analysis", prog.analysis, ANALYSIS)
        if p:
            out.size, out.overlap = p["size"], p["overlap"]
    for kind, items, limit in (("sel", prog.sels, "sels"), ("node", prog.ops, "ops"), ("layer", prog.layers, "layers"),
                               ("check", prog.checks, "checks")):
        if len(items) > LIMITS[limit]:
            ctx.err(items[LIMITS[limit]][-1], f"more than {LIMITS[limit]} {kind} statements")
    names = {}
    for kind, items in (("sel", prog.sels), ("node", prog.ops), ("layer", prog.layers), ("check", prog.checks)):
        for item in items:
            n, line = item[0], item[-1]
            if n in names:
                ctx.err(line, f"the name {n!r} is already used on line {names[n]}")
            names.setdefault(n, line)

    sels = {}  # name -> (normalised, expanded call count)
    for n, e, line in prog.sels:
        count = [0, 0]
        s = _sel(ctx, f"sel {n}", e, sels, count)
        if count[0] > LIMITS["sel_calls"] or count[1] > LIMITS["sel_calls_expanded"]:
            ctx.err(line, f"sel {n}: more than {LIMITS['sel_calls']} selection calls "
                          f"({LIMITS['sel_calls_expanded']} once named selections are expanded)")
            s = None
        if s is not None:
            sels[n] = (s, count[1])
    out.sels = {n: s for n, (s, _) in sels.items()}

    for n, c, at, mix, line in prog.ops:
        p = _params(ctx, f"node {n}", c, OPS, sel_names=out.sels)
        a = None
        if at is not None:
            count = [0, 0]
            a = _sel(ctx, f"node {n} at", at, sels, count)
            if count[0] > LIMITS["sel_calls"] or count[1] > LIMITS["sel_calls_expanded"]:
                ctx.err(line, f"node {n}: 'at' has too many selection calls")
        elif c.name in OPS and OPS[c.name].get("needs_at"):
            ctx.err(line, f"node {n}: {c.name} needs 'at <selection>' (the region it works on)")
        if a is not None and c.name in OPS and OPS[c.name]["domain"] == "time" and _has_frequency(a):
            ctx.err(line, f"node {n}: {c.name} works on the waveform, so its 'at' must be time-only "
                          "(time(...), and / or / not of those); a band would not limit it")
        if mix is not None and not 0 <= mix <= 1:
            ctx.err(line, f"node {n}: mix {mix:g} is outside [0, 1]")
        if p is not None:
            out.ops[n] = {"op": c.name, "params": p, "at": a, "mix": 1.0 if mix is None else float(mix),
                          "off": n in prog.off}

    uses, used = 0, set()
    op_names = {o[0] for o in prog.ops}
    for n, chain, line in prog.layers:
        for o in chain:
            if o not in op_names:
                ctx.err(line, f"layer {n}: no node named {o!r}" + _known(op_names))
            used.add(o)
        uses += len(chain)
        out.layers.append((n, list(chain)))
    if uses > LIMITS["op_uses"]:
        ctx.err(prog.layers[-1][2], f"more than {LIMITS['op_uses']} node uses over all layers")
    layer_names = [n for n, _, _ in prog.layers]
    if not layer_names:
        ctx.err(1, "a program needs at least one layer (layer main = op1 -> op2)")
    if prog.output is not None and isinstance(prog.output[0], L.Call):
        c, line = prog.output
        if c.name != "mix":
            ctx.err(line, f"output: expected a layer name or mix(layer: dB, ...), got {c.name}(...)")
        seen = set()
        for k, v, aline in c.args:
            if k not in layer_names:
                ctx.err(aline, f"output mix: no layer named {k!r}" + _known(layer_names))
            elif k in seen:
                ctx.err(aline, f"output mix: layer {k!r} given twice")
            elif not _is_num(v) or not -120 <= v <= 24:
                ctx.err(aline, f"output mix: {k} needs a gain in dB in [-120, 24], got {_show(v)}")
            else:
                out.output_mix[k] = float(v)
            seen.add(k)
        if not c.args:
            ctx.err(line, "output mix needs at least one layer")
        out.output = "mix"
    elif prog.output is not None:
        if prog.output[0] not in layer_names:
            ctx.err(prog.output[1], f"output: no layer named {prog.output[0]!r}" + _known(layer_names))
        out.output = prog.output[0]
    elif len(layer_names) == 1:
        out.output = layer_names[0]
    elif layer_names:
        ctx.err(prog.layers[-1][2], "several layers: say what the output is (output <layer>, or output mix(a: 0, b: -6))")

    for n, c, layer, line in prog.checks:
        p = _params(ctx, f"check {n}", c, CHECKS, sel_names=out.sels)
        if layer is not None and layer not in layer_names:
            ctx.err(line, f"check {n}: no layer named {layer!r}" + _known(layer_names))
        if p is not None:
            out.checks.append({"name": n, "check": c.name, "params": p, "layer": layer or out.output})

    for n in op_names - used:
        ctx.warnings.append(f"node {n} is not used by any layer")
    used_sels = _sel_uses(prog)
    for n in set(out.sels) - used_sels:
        ctx.warnings.append(f"sel {n} is not used")
    if ctx.problems:
        raise CheckError(ctx.problems)
    out.warnings = ctx.warnings
    return out


def _has_frequency(s):
    """Does a normalised selection limit frequency anywhere?"""
    if s[0] == "not":
        return _has_frequency(s[1])
    if s[0] in ("and", "or"):
        return any(_has_frequency(x) for x in s[1])
    _, name, p = s
    return name in ("band", "box", "harmonics")


def _sel_uses(prog):
    used = set()

    def walk(e):
        if isinstance(e, L.Ref):
            used.add(e.name)
        elif isinstance(e, L.Not):
            walk(e.arg)
        elif isinstance(e, (L.And, L.Or)):
            for a in e.args:
                walk(a)

    for _, e, _ in prog.sels:
        walk(e)
    for o in prog.ops:
        if o[2] is not None:
            walk(o[2])
        for k, v, _ in o[1].args:
            if isinstance(v, L.Ident):  # a sel-typed parameter (e.g. denoise's profile)
                used.add(v.name)
    for _, c, _, _ in prog.checks:
        for k, v, _ in c.args:
            if k == "at" and isinstance(v, L.Ident):
                used.add(v.name)
    return used
