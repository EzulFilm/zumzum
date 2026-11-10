"""Notes become measurements: a small rules table turns words in a person's note into goals, and each goal into
check lines the loop adds to every version, whatever the model wrote. These goals decide "done"; the model's own
checks run too, and a failing one also keeps a version from being done.

Every rule matches whole words (so "bring" is not "ring", "dialogue" is not "dial"). A requested amount ("down about
12 dB") becomes a two-sided `change` check around that amount; without an amount, a removal becomes a one-sided
`drop`. A note no rule understands is `unsupported`; a goal with nothing to measure against is `unmeasurable`.
Neither can end the loop as done.
"""

import re

# Telephone frequencies: the DTMF keypad pairs (ITU-T Q.23) and the North American call-progress tones (dial 350 + 440,
# ringback 440 + 480, busy 480 + 620). A note about a phone is about these, never about a held voice harmonic.
PHONE_HZ = (350, 440, 480, 620, 697, 770, 852, 941, 1209, 1336, 1477, 1633)
PHONE_TOL = 0.025  # Q.24 allows 1.5 % off nominal; plus the analysis bin (10.8 Hz at 44.1 kHz / 4096)

AMOUNT = re.compile(r"(?:about|around|roughly|by|some)?\s*(\d+(?:\.\d+)?)\s*db\b")


def _w(pattern, n):
    return re.search(r"\b(?:" + pattern + r")\b", n) is not None


def _amount(pattern, n, amount):
    """The dB amount for one goal: the note's only amount, or, when the note gives several ("hum by 12 dB and hiss
    by 6 dB"), the one in the goal word's own clause (up to the next "and", comma or semicolon), else None."""
    if len(AMOUNT.findall(n)) <= 1:
        return amount
    m = re.search(r"\b(?:" + pattern + r")\b", n)
    m = AMOUNT.search(re.split(r"\band\b|[,;]", n[m.end():], maxsplit=1)[0]) if m else None
    return float(m.group(1)) if m else None


def note_goals(note, analysis):
    """-> (sel lines, check lines, goals). Each goal is {"goal", "status": "measured" | "unmeasurable", "checks",
    "why"}; every generated name is prefixed `note_` (a model's names cannot collide: see `merge`)."""
    n = note.lower()
    sels, checks, goals = [], [], []
    m = AMOUNT.search(n)
    amount = float(m.group(1)) if m else None

    def goal(name, lines, why, status="measured"):
        checks.extend(lines)
        goals.append({"goal": name, "status": status, "checks": [ln.split()[1] for ln in lines], "why": why})

    # phrases, not single words ("a brittle buzz" is not mains hum): "buzz" is hum only with a mains / electrical /
    # power / ground / earth context
    mains_ctx = _w(r"mains|electrical|electric|power|ground|earth|grounding|hum", n)
    if _w(r"hum|mains|60 ?hz|50 ?hz", n) or (_w(r"buzz|buzzing", n) and mains_ctx):
        h = analysis.get("hum") or {}
        named = 60.0 if re.search(r"\b60 ?hz\b", n) else 50.0 if re.search(r"\b50 ?hz\b", n) else None
        hum = _amount(r"hum|buzz|mains|60 ?hz|50 ?hz", n, amount)
        need = hum if hum is not None and _w(r"by|down|reduce|lower", n) else 20
        if named is None and not h.get("f0_hz"):
            # no hum found and none named: measure the mains series (50 or 60) that is stronger in the input;
            # assuming 60 lets a 50 Hz hum pass while worse, and both series share harmonics (300, 600 Hz ...)
            goal("hum", [f"check note_hum = mains_hum_drop(harmonics: 10, min_db: {need:g})"],
                 f"no steady hum found: the mains series (50 or 60 Hz) stronger in the input down at least {need:g} dB")
        else:
            f0 = named if named is not None else round(h["f0_hz"])
            goal("hum", [f"check note_hum = hum_drop(f0_hz: {f0:g}, harmonics: 10, min_db: {need:g})"],
                 f"hum at {f0:g} Hz down at least {need:g} dB")
    if _w(r"hiss|noise|noisy|denoise", n):
        if analysis.get("quiet_spans_s"):
            a, b = analysis["quiet_spans_s"][0]
            sels.append(f"sel note_sel_quiet = time(t_s: {a:g}..{b:g})")
            hiss = _amount(r"hiss|noise|noisy|denoise", n, amount)
            need = hiss if hiss is not None else 6
            goal("noise", [f"check note_hiss = drop(at: note_sel_quiet, min_db: {need:g})"],
                 f"noise in the quiet span {a:g}..{b:g} s down at least {need:g} dB")
        else:
            goal("noise", [], "no quiet span to measure the noise in", status="unmeasurable")
    if _w(r"clicks?|crackles?|crackling|pops?", n):
        goal("clicks", [f"check note_clicks = clicks(max: {max(0, analysis.get('clicks', 0) // 3)})"],
             "at most a third of the clicks left")
    if _w(r"drums?|percussion", n) and _w(r"down|quieter|lower|less|too loud|reduce", n):
        # 2-16 kHz: snare and cymbals; a piano holds little up there (drums_down: piano 27-50 dB under the drums per
        # octave above 2 kHz). 5-16 kHz alone lets a program leave the drums' 1-5 kHz in.
        sels.append("sel note_sel_cymbals = band(f_hz: 2000..16000)")
        drums = _amount(r"drums?|percussion", n, amount)
        if drums is not None:
            tol = max(2.0, round(0.25 * drums, 1))
            goal("drums", [f"check note_drums = change(at: note_sel_cymbals, target_db: {-drums:g}, tol_db: {tol:g})"],
                 f"drums (cymbal band) {-drums:g} dB +- {tol:g}")
        else:
            goal("drums", ["check note_drums = drop(at: note_sel_cymbals, min_db: 6)"], "drums down at least 6 dB")
    keep_voice = _w(r"voice|words|speech|dialogue", n) and _w(r"keep|clear|leave|as it is|natural", n)
    # a bare "tone" is timbre ("the tone is harsh"); tones count with a phone / dial / beep / keypad context, or by
    # their own words
    # "ringing out" is music, not a tone; "ringing" needs a phone / bell / beep context
    ring_out = _w(r"rings? out|ringing out", n)
    tone_ctx = _w(r"phones?|dial|dials|dialling|dialing|beeps?|keypad|touch-?tones?|dtmf|bells?|ring tones?", n)
    if _w(r"beeps?|dial|dials|dialling|dialing|whistles?|squeals?|keypad|touch-?tones?|dtmf", n) or (
            _w(r"tones?|rings?|ringing", n) and tone_ctx and not ring_out):
        ev = list(analysis.get("tonal_events") or [])
        mt = re.search(r"\b(\d+):(\d\d)\b", n)
        if mt:  # "at about 0:09": the events within 5 s after that time
            t = int(mt.group(1)) * 60 + int(mt.group(2))
            ev = [e for e in ev if t - 1 <= e["t_s"][0] <= t + 5]
        phone = _w(r"phones?|dial|dials|dialling|dialing|keypad|touch-?tones?|dtmf", n)
        if phone:  # only events at a telephone frequency
            ev = [e for e in ev if any(abs(e["f_hz"] - f) <= PHONE_TOL * f for f in PHONE_HZ)]
        if ev:
            boxes = [f"box(t_s: {max(0, e['t_s'][0] - 0.05):.2f}..{e['t_s'][1] + 0.05:.2f}, "
                     f"f_hz: {max(0, e['f_hz'] - 60):g}..{e['f_hz'] + 60:g})" for e in ev[:16]]  # the checker allows 16 calls in a sel; ev is strongest first
            sels.append("sel note_sel_tones = (" + "\n  or ".join(boxes) + ")")
            if keep_voice:  # box by box: the tone gone (<= 3 dB over its surroundings), the voice in it kept (<= 6 under).
                lines = ["check note_tones = filled(at: note_sel_tones, max_above_db: 3, max_below_db: 200)",
                         "check note_voice_in_tones = filled(at: note_sel_tones, max_below_db: 6)"]
                why = "each no more than 3 dB over its surroundings (tone gone) and 6 dB under them (voice kept)"
            else:
                lines = ["check note_tones = drop(at: note_sel_tones, min_db: 10)"]
                why = "down at least 10 dB"
            goal("tones", lines, f"{len(boxes)} {'telephone-frequency ' if phone else ''}tonal events, {why}")
        else:
            goal("tones", [], "no steady tone found where the note says" + (" at a telephone frequency" if phone else ""),
                 status="unmeasurable")
    if _w(r"voice|words|speech|dialogue", n) and _w(r"keep|clear|leave|as it is|natural", n):
        sels.append("sel note_sel_voice = band(f_hz: 300..3400)")
        goal("keep voice", ["check note_voice = keep(at: note_sel_voice, tol_db: 3)"], "voice band within 3 dB")
    if _w(r"piano|music", n) and _w(r"keep|leave|alone|nothing else", n):
        sels.append("sel note_sel_music = band(f_hz: 200..1000)")
        goal("keep music", ["check note_music = keep(at: note_sel_music, tol_db: 3)"], "music band within 3 dB")
    checks.append("check note_peak = peak(max_dbfs: -0.1)")  # always, but it is a guard, not a goal
    return sels, checks, goals


def note_checks(note, analysis):
    """(sel lines, check lines) only; see `note_goals`."""
    sels, checks, _ = note_goals(note, analysis)
    return sels, checks


def request_status(goals, results, damage):
    """The version's status, from the note's goals, every check result and the damage gate:
    `unsupported` (no rule understood the note), `unmeasurable` (a goal has nothing to measure), `damaged`,
    `not_done` (a goal's check, or one of the model's own checks, failed), or `done`."""
    if not [g for g in goals if g["goal"] not in ("keep voice", "keep music")]:
        return "unsupported"
    if any(g["status"] == "unmeasurable" for g in goals):
        return "unmeasurable"
    if damage:
        return "damaged"
    if not all(r["pass"] for r in results):
        return "not_done"
    return "done"


def merge(program, sels, checks):
    """Append the note's lines to a program. `note_` is reserved for generated names: any sel, node, layer or check
    of the model's that declares one is dropped first, since all four kinds share one name space."""
    from .lang import statements
    from .errors import CheckError
    lines = program.rstrip().splitlines()
    try:
        drop = set()
        for word, a, b in statements(program):
            if word in ("sel", "node", "layer", "check") and re.match(rf"\s*{word}\s+note_", lines[a]):
                drop.update(range(a, b + 1))  # the whole statement, however many lines it spans
        keep = [ln for i, ln in enumerate(lines) if i not in drop]
    except CheckError:  # not even tokenizable: leave it for the checker to report
        keep = lines
    return "\n".join(keep + [""] + sels + checks) + "\n"
