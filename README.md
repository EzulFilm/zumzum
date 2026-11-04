# Zumzum

**Audio repair written as a short, checked program.** You describe a fix in a few lines of text: where in time and
frequency, what to do there, and how to tell that it worked ("hum down 30 dB", "voice kept within 1 dB"). Zumzum
checks the program before anything runs, renders it with one NumPy reference engine, and measures every check on the
before / after pair. Programs can be written by a person, by simple keyword rules, or by a local AI model, and all
three go through the same checker. It handles hum, hiss, clicks, short tones such as phone dialling, and turning the
drums down under other music. Zumzum 0.1 is a prototype.

![A Zumzum program beside the 1888 wax cylinder it cleans up, before and after](docs/img/readme_program.png)

The program on the left is [examples/cylinder_denoise.zz](examples/cylinder_denoise.zz). It learns the noise from
one pause, takes the hiss in another pause down about 23 dB, and changes the words band by only 0.4 dB.

## A program

```
zumzum 0.1
name "Mains hum under a voice"

sel hum_lines = harmonics(f0_hz: 60, count: 16, width_hz: 50)
sel voice = band(f_hz: 200..4000) and not hum_lines

node h = dehum(f0_hz: 60, harmonics: 16, depth_db: 50)
layer main = h

check hum_gone = hum_drop(f0_hz: 60, harmonics: 16, min_db: 30)
check voice_kept = keep(at: voice, tol_db: 1)
check no_clip = peak(max_dbfs: -0.1)
```

- `sel` names a region: `time(t_s: a..b)`, `band(f_hz: a..b)`, `box(...)` (both), `harmonics(f0_hz: ...)` or
  `all()`, combined with `and`, `or`, `not`. Units are part of every parameter name (`_s`, `_ms`, `_hz`, `_db`,
  `_dbfs`).
- `node` applies one operation, optionally only inside a region (`at`), partly (`mix`) or not at all (`off`).
- `layer` chains nodes, each layer starting from the input. With one layer, that layer is the output (as here).
  With several, `output` picks one or mixes them, each with a gain in dB.
- `check` is a measurement that must pass. Each node's own effect (inside and outside its region) is measured as
  well, so you can see which node did what.

A program has no file access, loops, arithmetic or randomness, and its size and cost are limited. Every name, unit
and range is checked, and each problem is reported with its line number. The full language is in
[docs/spec.md](docs/spec.md); the parameter table, with every range and default, is in
[zumzum/ops.py](zumzum/ops.py).

**Operations:** `gain`, `dehum` (mains hum and its harmonics), `denoise` (MMSE, log-MMSE, Wiener or spectral
subtraction, from a noise profile you point at or from the quietest frames), `declick` (finds clicks and refills
them by autoregressive interpolation; its region can only be a time span), `heal` (fills a region from the sound
around it) and `split_hp` (keeps the harmonic or the percussive part; no trained model).

**Checks:** `drop` (a region got at least N dB quieter), `change` (a region changed by N dB, within a tolerance),
`keep` (every octave of a region stayed within N dB), `untouched` (whatever changed in a region is at least N dB below
it), `hum_drop`, `clicks` (at most N left), `peak`, and `musical_noise` (experimental, not validated). A damage gate
always runs as well: an output with non-finite samples, a peak above 0 dBFS, more than 6 dB louder than the input or
more than 40 dB quieter never counts as done.

## Quick start

You need Python 3.12 or later. Clone this repository and run these from its folder (on Windows, activate with
`venv\Scripts\activate` instead):

```
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                       # the tests; no network, GPU or model needed
python tools/fetch_audio.py               # downloads seven recordings to cache/audio/, checked by SHA-256
python tools/make_examples.py             # renders five examples into out/examples/ (a few minutes)
```

The tests should all pass in under a minute. `fetch_audio.py` needs a network connection the first time.
`make_examples.py` prints each check as PASS or FAIL, and each example's first line says `ALL PASS` when every check
passed and the damage gate is clear. You can render one example by name, for example
`python tools/make_examples.py hum_voice`. Each example folder gets `before.wav`, `after.wav`, `removed.wav` (what
the edit took out), `sheet.png` (waveform, spectrograms before and after, and a map of what changed) and
`results.json` with every check.

## Run a program on your own file

This needs [ffmpeg](https://ffmpeg.org/) on your PATH. It is not included; install it yourself under its own
licence.

```
python tools/convert.py examples/mp3_record.zz input.mp3 output.mp3 --sheet sheet.png
```

The input can be any common audio or video file. The output type follows its extension: `.mp3`, `.m4a`, `.flac` or
`.wav`. For `.mp4`, `.mov` or `.mkv` the picture is copied unchanged and only the sound is replaced. The checks are
printed and saved to `output.json`. The command exits with 0 only when every check passes and the damage gate is
clear, so it can be used in scripts. An existing output, its `.json` or the sheet is not replaced unless you add
`--overwrite`, and the input is never overwritten.

To try the mp3 example on the 1920 record that `fetch_audio.py` downloads, take its first 60 seconds:

```
ffmpeg -t 60 -i cache/audio/mp3_sandman_1920.mp3 -c copy sandman_60s.mp3
python tools/convert.py examples/mp3_record.zz sandman_60s.mp3 sandman_clean.mp3 --sheet sandman_sheet.png
```

Times and frequencies in a program belong to one recording. To use an example on your own file, change its pauses
(`time(...)`, `profile_t_s`) and bands to match your sound.

## The examples

![Before and after spectrograms of every example](docs/img/readme_examples.png)

| example | request | what the checks measured (0.1) | program |
|---|---|---|---|
| hum under a voice | remove the 60 Hz hum, keep the voice | hum no longer above the spectrum around it; voice band within 0.01 dB | [hum_voice.zz](examples/hum_voice.zz) |
| 1888 wax cylinder | less hiss, keep the words clear | hiss in a pause down 23 dB; words within 0.4 dB | [cylinder_denoise.zz](examples/cylinder_denoise.zz) |
| 1917 78 rpm record | take out the clicks, change nothing else | clicks found: 451 before, 70 after (allowed 100); music within 0.01 dB | [record_declick.zz](examples/record_declick.zz) |
| phone dialling over a reading | remove the dial tones and the rumble, keep the voice | tones down 11.5 dB, rumble down 29 dB; the change elsewhere is 66 dB below the sound there (allowed: 30 dB below) | [phone_heal.zz](examples/phone_heal.zz) |
| drums over a piano | drums down about 12 dB, leave the piano | cymbal band down 11.9 dB; piano within 1.5 dB (allowed 3) | [drums_down.zz](examples/drums_down.zz) |
| 1920 acoustic record, **mp3 in, mp3 out** | take out the clicks and the hiss, keep the band | clicks 200 before, 26 after (allowed 60); hiss in the lead-in down 15.6 dB; band within 1.2 dB | [mp3_record.zz](examples/mp3_record.zz) |
| 1920s newsreel, **video in, video out** | less hiss and rumble, keep the voice | no result here: written for one newsreel clip that `fetch_audio.py` does not download; a starting point for your own video | [video_denoise.zz](examples/video_denoise.zz) |

The first five are rendered by `tools/make_examples.py`; the mp3 example runs through `tools/convert.py` as shown
above. A check passing means the measurement passed. It does not mean the result sounds good: listen to it.

**Example audio.** The recordings are public domain or CC0, from Wikimedia Commons. Each file's page, licence,
author and SHA-256 are in [examples/sources.json](examples/sources.json). The audio itself is not stored in this
repository; `tools/fetch_audio.py` downloads it. Where a recording has no fault of its own, the example adds one
(hum, clicks, phone dialling) with a fixed seed, or mixes two recordings (drums over a piano), as
`tools/make_examples.py` describes.

## Asking in words: the editing loop

`tools/live_run.py` runs the five example inputs with a plain-language note each, for example "There is a mains hum
under the voice. Remove it, keep the voice as it is." The note is turned into checks by fixed word rules
([zumzum/notes.py](zumzum/notes.py)), so the goal is measured the same way whoever writes the program:

- hum, buzz, mains (50 or 60 Hz): the hum must drop at least 20 dB, or by the amount you give;
- hiss, noise: the noise in the first quiet span the analysis finds must drop at least 6 dB, or by the amount you
  give (no quiet span: `unmeasurable`);
- clicks, crackle, pops: at most a third of the clicks may remain;
- drums or percussion down: the cymbal band (5-16 kHz) must drop at least 6 dB, or change by the amount you give
  ("about 12 dB");
- tones, beeps, dialling, whistles, optionally at a time ("at about 0:09"): the steady tones found there must drop
  at least 10 dB;
- keep the voice / words (300-3400 Hz), keep the music / piano (200-1000 Hz): that band must stay within 3 dB.

These are simplified; `zumzum/notes.py` has every trigger word and band.

A program is then written by a driver, run, and measured. Failed checks and checker problems go back as text, and
each request gets up to three versions.

```
ZUMZUM_DRIVER=rules python tools/live_run.py hum_voice
ZUMZUM_DRIVER=openweight ZUMZUM_LLM_URL=http://127.0.0.1:8091/v1 python tools/live_run.py
```

Without names it runs all five. In PowerShell, set the variable first: `$env:ZUMZUM_DRIVER = "rules"`. `rules`
needs no model. `openweight` talks to any OpenAI-compatible chat server running a local model (for example
llama.cpp's `llama-server`). The model never hears the sound: it gets the note and a few measured numbers (levels,
noise floor, quiet spans, hum, click count, steady tones), and it answers only with a Zumzum program, which is
checked like any other input. Each version ends as `done`, `not_done`, `damaged`, `unmeasurable` (nothing to
measure the goal against) or `unsupported` (no rule understood the note). Results go to `out/llm/<driver>/`.

## Status and limits

Zumzum 0.1 is a prototype, and the language may change.

- A passing check is an objective measurement, not a listening test. There are no perceptual quality checks yet.
- Regions are simple shapes in seconds and hertz: boxes, bands and harmonics of a given fundamental.
- Channels are processed independently; there are no stereo-aware operations.
- No trained separation models: `split_hp` is a signal-processing split, so "drums down" is approximate.
- `musical_noise` is experimental and no example relies on it.
- Audio limits: 8-192 kHz, up to 8 channels, up to one hour. Rendering holds several copies of the audio in memory.
- One reference engine, in NumPy on the CPU. The STFT / ISTFT round trip is exact to 1e-12 (tested).

Planned: time stretch and pitch shift of a region, de-reverb, separation into layers with trained models (with a
residual layer so nothing is lost), perceptual checks calibrated by listening, views of the sound for models, and
faster engines tested against the reference.

## Contributing

Issues and pull requests are welcome. Please run `python -m pytest -q` before sending a change; the tests also run on
GitHub Actions with Python 3.12 and 3.13. A change to the language goes into [docs/spec.md](docs/spec.md) and
[zumzum/ops.py](zumzum/ops.py) together. Contributions are accepted under the Apache License 2.0.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The example recordings are not part of this repository;
their own licences are listed in [examples/sources.json](examples/sources.json).
