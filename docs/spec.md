# Zumzum 0.1 language specification (draft, 2026-10-28)

Status: prototype. Files use the extension `.zz`. **This document is normative**: it defines what a
program means. The reference engine (`zumzum/engine.py`) implements it, the checker (`zumzum/checker.py`) enforces it,
and the parameter table (`zumzum/ops.py`) is the one source of names, types, units, ranges and defaults. Where the code
and this text disagree, one of them is wrong. A change to either goes into both on purpose.

## 1. Model

A program describes one edit of one audio file as a small **node graph**. It has **selections** (`sel`), regions of
the time-frequency plane; **nodes** (`node`), one op each, with an optional region (`at`), a strength (`mix`) and a
bypass (`off`); **layers** (`layer`), serial chains of nodes that each start from the input, which are the graph's
parallel branches; an **output**, either one layer or a mix of layers with a gain each; and **checks** (`check`),
measurements on the before / after pair that say whether the note behind the edit was met. Each node's own effect is
measured as it runs (energy change inside and outside its region), so a revision knows which node to change.

A program has no variables, arithmetic, loops, functions, I/O or randomness. It is data with a grammar, and its cost
is bounded (section 7). The checker treats every program the same way, whoever wrote it: a person, a rules driver or a
language model. A program that passes the checker is safe to render.

Design rules (kept on purpose; they shape what the language can say):
- Selections are written as text in audio units (seconds, hertz, a fundamental and harmonic numbers). Nothing is
  drawn, brushed or painted on a display, no image or pixel coordinate is ever a selection, and an op never returns a
  mask, an image or an opacity.
- A model may look at **views** of the sound (section 8); it answers only in checked text. Edits always happen on
  the sound, never on a picture that is turned back into sound. Shapes stay simple: boxes, bands, and harmonics of a
  fixed fundamental. There are no contours or polygons, and no many-box outlines that trace a shape.
- No automatic note-object detection in polyphonic audio. A harmonic series is given its fundamental.
- `dehum`'s cut depth is a fixed number, never driven by the signal level.
- `declick` fills each gap by one joint least-squares AR solve, not by blending one-sided predictions.
- No trained separation model until there is one whose code, weights and training data all have clean licences.

## 2. Grammar (EBNF)

```
program  = { NL } "zumzum" NUMBER NL { stmt NL | NL } EOF
stmt     = "name" STRING | "about" STRING | "analysis" call
         | "sel" IDENT "=" sel
         | "node" IDENT "=" call { "at" sel | "mix" NUMBER | "off" }
         | "layer" IDENT "=" ( "pass" | IDENT { "->" IDENT } )
         | "output" ( IDENT | "mix" "(" IDENT ":" NUMBER { "," IDENT ":" NUMBER } ")" )
         | "check" IDENT "=" call [ "on" IDENT ]
sel      = sand { "or" sand }
sand     = sunary { "and" sunary }
sunary   = "not" sunary | "(" sel ")" | call | IDENT          (IDENT = an earlier sel)
call     = IDENT "(" [ arg { "," arg } [ "," ] ] ")"
arg      = IDENT ":" value
value    = NUMBER [ ".." NUMBER ] | STRING | IDENT | "true" | "false"
```

Lexical rules: `NUMBER = -?\d+(\.\d+)?([eE][+-]?\d+)?`; `IDENT = [A-Za-z_][A-Za-z0-9_]*` (at most 40 characters);
`STRING` is double-quoted on one line, with the escapes `\"` and `\\` only, at most 400 characters. `#` starts a
comment that runs to the end of the line. A newline ends a statement except inside `( )`, so a long selection is
wrapped in parentheses. Reserved words: `zumzum name about analysis sel node layer output check at mix off on pass and
or not true false`.

Uniqueness: `name`, `about`, `analysis` and `output` appear at most once each. All `sel`, `node`, `layer` and `check`
names share one name space and are unique. A parameter appears at most once in a call, and `at` and `mix` at most
once per node. Each of these is an error, never "the last one wins". A selection may refer only to selections defined
above it, so there are no cycles.

Defaults: `analysis stft(size: 2048, overlap: 4)`. `mix` is 1, in [0, 1]. A node is on unless `off`. With exactly one
layer, that layer is the output; with several, `output` is required: one layer's name, or `mix(a: 0, b: -6)`, which
sums the named layers, each at its gain in dB in [-120, 24]. A check without `on` measures the output (the mix, if
there is one).

## 3. Types, units and ranges

| kind | written | meaning |
|---|---|---|
| num | `0.25` | a finite real in the parameter's [lo, hi] |
| int | `7` | an integer in [lo, hi] |
| enum | `mmse` | one of the listed identifiers (or numbers, for `fft` / `size`) |
| range | `1.2..1.5` | lo < hi, both in the parameter's [lo, hi] |
| sel | `voice` | a reference to a named `sel` statement, wherever a parameter takes one (check `at`, `denoise` `profile`) |

**Units are part of the parameter name:** `_s` seconds, `_ms` milliseconds, `_hz` hertz, `_db` decibels, `_dbfs` dB
relative to full scale (a full-scale sine is 0 dBFS at its peak). A bare number never carries a hidden unit. Time is
measured from the start of the file. Frequency runs from 0 Hz to the Nyquist frequency. When the host gives the
checker the file's duration and sample rate, a span that starts after the end of the file, or a band that starts
above the Nyquist frequency, is an error.

The full parameter table, with ranges and defaults, is in `zumzum/ops.py` (`SELS`, `OPS`, `CHECKS`, `ANALYSIS`).

## 4. Selections

A selection is a weight M(f, t) in [0, 1] on an STFT grid (bins x frames): 1 is fully selected, 0 is untouched.

| call | meaning |
|---|---|
| `all()` | everything |
| `time(t_s: a..b, feather_ms: 0)` | all frequencies inside a time span |
| `band(f_hz: a..b, feather_hz: 0)` | a frequency band over the whole file |
| `box(t_s: a..b, f_hz: a..b, feather_ms: 0, feather_hz: 0)` | a time span and a band together |
| `harmonics(f0_hz, count: 10, width_hz: 6, t_s: optional)` | bands of `width_hz` at k * f0, k = 1..count (each at least one bin wide), over `t_s` or the whole file |

A feather is a raised-cosine fall from 1 to 0 over the given width **outside** the stated edges. A frame belongs to
a span by its centre time, and a bin belongs to a band by its centre frequency. `a and b` = min, `a or b` = max,
`not a` = 1 - a.

## 5. Engine

Audio is float64 (channels, samples) at full scale ±1. Channels are processed independently.

**STFT** (`zumzum/stft.py`): periodic Hann window of `size` samples, hop = size / overlap. The signal is padded with
`size` zeros on both sides, so every sample is covered by full frames. The inverse is a least-squares overlap-add,
dividing by the summed squared window. `istft(stft(x)) == x` to within 1e-12 for every size and overlap (tested).
Frame t is centred at (t * hop - size / 2) / sr seconds.

**Layers.** Each layer starts from the input and applies its nodes in order, skipping nodes marked `off`. `pass` is
the identity. An `output mix(...)` is the sum of the named layers, each times 10^(gain / 20).

**Per-node effect.** For each STFT node, the engine records `inside_db` = 10 log10(E(Z M) / E(X M)) and
`outside_db` = the same with 1 - M (energy after / before the node, inside and outside its region, mean over
channels). A node that changes what it should not shows up there by name.

**STFT ops.** On the op's own grid (its `fft` parameter, else the analysis size, with the analysis overlap):
X = STFT(current signal), Y = op(X), Z = X + mix * M * (Y - X), and the signal becomes ISTFT(Z). Without `at`, M = 1.

**Time ops** (`declick`) take only time-only selections (`time`, and `and` / `or` / `not` of those); a band would not
limit them, so the checker refuses it. y = op(x), then x + mix * w(n) * (y - x), where w(n) is the selection's largest
weight over frequency at each frame of the program's analysis grid (its `size` and hop), interpolated to samples.

| op | what it does |
|---|---|
| `gain(db)` | multiply by 10^(db / 20) |
| `dehum(f0_hz, harmonics: 10, depth_db: 40, width_hz: 3, search_hz: 0.5, fft: 16384)` | refine f0 within ±search_hz (0.01 Hz steps, the f0 whose harmonics hold the most mean power). For each harmonic, the bins within max(width / 2, 2 bins) are cut to the per-frame median level of the bins just beside them, but never by more than depth_db. The depth is fixed. |
| `denoise(mode: mmse, profile: optional, profile_t_s: optional, reduce_db: 15, smooth: 0.98, over: 2, noise_db: 0, track_s: 0, smooth_hz: 0, smooth_ms: 0, gate_db: 0, gate_release_ms: 200, fft: 0)` | **Noise estimate:** per bin, from the cells fully inside the sel named by `profile`, else the span `profile_t_s`, else the quietest 10 % of frames; with `track_s` > 0 it instead follows a changing noise (the running minimum of the time-smoothed power over `track_s`, times 1.5); then raised by `noise_db`. **Gain:** `mmse` Ephraim-Malah (1984) short-time spectral amplitude, `logmmse` Ephraim-Malah (1985) log-spectral amplitude, `wiener` xi / (1 + xi), all with a decision-directed a-priori SNR xi (smoothing `smooth`; xi is floored at -max(25, reduce_db) dB); `subtract` power subtraction with over-subtraction `over`. The gain is clipped to [10^(-reduce_db / 20), 1], then averaged over `smooth_hz` in frequency and `smooth_ms` in time. **Gate:** with `gate_db` > 0, a per-frame posterior SNR in 300-4000 Hz gives a presence from 0 (1 dB) to 1 (6 dB); it opens at once, closes over `gate_release_ms`, opens 20 ms early, and every bin of a frame goes down a further gate_db x (1 - presence). |
| `declick(sensitivity: 10, max_ms: 3, order: 24)` | AR(order) prediction error per 4096-sample block. A click is where the error is more than `sensitivity` times its local robust level (1.4826 x the running median of the absolute error over 10 ms), widened by 2 samples before and order / 4 after. Each click up to `max_ms` long is refilled by one least-squares AR interpolation over both sides (Janssen, Veldhuis & Vries 1986). The model order is at least 3x the gap length (at most 160), fitted on up to 2048 samples each side. An implausible fill (> 1.2x the surrounding peak) falls back to a straight line. A fill is held within ±0.99. Three passes. |
| `heal(direction: time, context_ms: 60, phase: keep, fft: 0) at <sel>` | the region M > 0 is replaced by an estimate from outside it. `time`: per bin, log magnitude interpolated linearly between the mean of `context_ms` before and after each run of selected frames. `freq`: per frame, between the 3 bins below and above. `both`: their average. `phase: keep` keeps the region's own phase. `continue` advances the phase of the frame before the region by its measured instantaneous frequency (for steady tones). `at` is required. |
| `split_hp(part, rest_db: -120, time_ms: 200, freq_hz: 500, fft: 0)` | harmonic / percussive split by median filtering (FitzGerald 2010). H = median along time over time_ms, P = median along frequency over freq_hz, and soft masks H² / (H² + P²) and P² / (H² + P²). Keeps `part`, with the other part at rest_db (-120 removes it). |

## 6. Checks

A check measures the before (program input) / after (a layer, by default the output) pair on the program's analysis
grid. Every result has `pass`, `value` and a one-line `text` that a person, or a model, can act on.

Region checks measure the selection's **core**: cells with weight 1, eroded by one analysis window (2 bins in
frequency and `overlap` frames in time). Spectral leakage across a region's edge is therefore not counted against
the op. If the selection is too small to have a core, all fully selected cells are used, or failing that every
selected cell. **A check selects the sound, not the op's region.** To test that a sound is gone, select an area at
least as wide as the sound itself. A check on only the narrow region an op cut can pass while the sound is still
there. This was found on 2026-10-28, on the phone example.

| check | passes when |
|---|---|
| `drop(at, min_db)` | 10 log10(E_before / E_after) >= min_db over the selection |
| `keep(at, tol_db: 1)` | in every octave band (from 31.25 Hz, plus >= 16 kHz) of the selection that holds sound (within 60 dB of its loudest band), the change is within ±tol_db |
| `untouched(at, max_db: -40)` | energy of (after - before) inside the selection, relative to before, is <= max_db |
| `hum_drop(f0_hz, harmonics: 10, min_db)` | tonal excess at f0 (refined ±1 Hz on before) and its harmonics, i.e. Welch PSD power in ±2 bins above the median of the neighbouring bins, fell by min_db, or is less than 10 % of that neighbouring floor |
| `clicks(max: 0, sensitivity: 10)` | the `declick` detector finds at most `max` clicks in after (summed over channels) |
| `peak(max_dbfs: -0.1)` | the sample peak is at most max_dbfs |
| `change(at, target_db, tol_db: 3)` | the selection's energy changed by target_db, within ±tol_db (after minus before, dB) |
| `musical_noise(at, max_ratio: 1.5)` | **experimental, not validated**: the kurtosis of the noise-only cells' power (each bin divided by its own mean), after / before, is at most max_ratio. It is present in 0.1 but did not flag plain spectral subtraction in a test, so no example relies on it |

`hum_drop` needs at least 4,096 samples. On shorter audio it fails with value 0, and its text says the hum was not
measurable and why.

**Damage gate** (always, `zumzum/measure.py`): non-finite samples, clipping (peak > 0 dBFS), an output more than 6 dB
louder than the input, or an output more than 40 dB quieter (near silence) than any input that is not digital silence,
however quiet. A damaged version can never end the loop as done, whatever its checks say.

Planned (not in 0.1): perceptual checks that follow the three questions listeners are asked about noise reduction
(how much the speech was damaged, how intrusive the remaining noise is, overall quality), using measures with clean
licences (no-reference speech quality estimators, ViSQOL, STOI / ESTOI, SI-SDR, HAAQI for music); onset-sharpness and pre-onset energy checks for
smearing and pre-echo; a validated replacement for the experimental `musical_noise`. Thresholds will be calibrated by
ear on a small listening set before any of them decides "done".

## 7. Limits

Audio: sample rate 8-192 kHz, at most 8 channels, at most one hour, at most 8 x 48000 x 3600 samples in all (an hour
of 7.1 sound at 48 kHz; so 8 channels at 192 kHz fit 15 minutes), finite samples only; anything else is refused when it
is loaded, checked or rendered. The limits bound each array: rendering keeps one copy per layer plus STFT buffers, so
peak memory is several times the input's size.

Source <= 32 KiB; <= 32 sels, 32 nodes, 8 layers, 32 checks; <= 64 node uses over all layers (each node use is at most one
STFT pass, or one declick of three passes); one selection expression <= 16 calls as written and <= 64 once named
selections are expanded, nested <= 16 deep; names <= 40 characters; strings <= 400 characters; number literals <= 40
characters and finite. FFT sizes 256..32768. Every number is range-checked, and every problem is reported with its
line. The host bounds the audio length.

## 8. What a model is given

`zumzum/analysis.py` turns the audio into a few lines of numbers. This is all a model learns about the audio, because
it never hears it: duration, sample rate, channels, peak and RMS; octave-band levels and noise floor; up to three
quiet spans (candidates for a noise profile); a steady low tone at 50 or 60 Hz (mains hum, or a held low note); the
click count; and up to 16 steady tonal events (time span, frequency, level above the usual spectrum). The JSON a model
gets holds all 16 events; the short text summary beside it shows the strongest 8.

**Goals and status.** The loop turns a person's note into goals with whole-word rules (`zumzum/notes.py`). A
requested amount ("down about 12 dB") becomes a two-sided `change` check (target -12, tolerance max(2, 25 %) dB);
without one, a removal is a one-sided `drop`. Each version gets one status:
- `unsupported`: no rule understood the note;
- `unmeasurable`: a goal has nothing to measure against;
- `damaged`: the damage gate fired;
- `not_done`: a goal's check, or one of the model's own checks, failed;
- `done`: every check passed, with no damage.
Only `done` ends the loop as a success. `unsupported` and `unmeasurable` end it at once, since no revision can change
them. "done" means "the objective checks passed", not "it sounds good".

**Names starting with `note_`** are reserved for the checks and selections the loop generates from a person's note. A
model's own `sel`, `node`, `layer` or `check` with such a name is dropped before the program is checked.

**Revision feedback** gives the model, for each version: the checker's problems, or every check result and the damage
gate, plus each node's measured effect (energy after / before, inside and outside its region, or "off").

**Views (planned, design).** A model may also be shown pictures of the sound: waveform and spectrogram with labelled
time and frequency axes and a grid, before / after / change views, and zooms of a region it asks for with a `look`
request (a time span and a band, in seconds and hertz). Views are input only. The answer is always Zumzum text, in
seconds and hertz, that goes through the checker, and the checker refuses pixel coordinates and images. One model
may both look at the views and write the next edit.

## 9. Not in 0.1 (planned)

Time stretch and pitch shift of a region (phase vocoder: Flanagan & Golden 1966, Laroche & Dolson 1999), de-reverb,
a `separate` node that splits a track into layers with a trained model plus a residual layer (so the layers sum back
to the input, checked), NMF unmixing and "a sound like this sample" (Smaragdis 2004), `look` requests for views,
transient selections, a canonical formatter and program hash, keyframes, stereo-aware ops, and GPU / plugin / web
targets tested against this engine.

## 10. Planned: `separate` (design, not in 0.1)

`node n = separate(model: <name>, stem: <stem>)` keeps one stem of a trained separation model. `stem: rest` is the
input minus the sum of the model's other stems, so a model's stems always sum back to the input. Models are plug-ins
(`zumzum/models/<name>.py`). Each declares its stems, sample rate, pinned source, SHA-256 and licence, so the checker
refuses an unknown model or stem by name. The engine runs a model once per (model hash, input hash), with a fixed seed
on CPU, caches the stems, and records the model hash in the results. A new check `sums_to(layers: [...], max_db)`
measures how far the named layers' sum is from the input. A separation is chosen by model and stem name (and a text
prompt for prompted models), never by drawing. No model ships until its code, weights and training data are cleared.
