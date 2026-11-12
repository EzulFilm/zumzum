# Changelog

What changed in Zumzum, newest first. The language version stays 0.1 until its first breaking change.

## 2026-11-12

- The rules driver reads more ways of saying it: a phone in other words (telephone, number pad, push-button keys),
  rumble described as a drone or boom underneath (but not a wanted drone such as bagpipes or a singer), and it writes
  hum removal for only one mains series when one clearly stands higher. Keypad and dial tones are healed in narrow
  boxes around each tone, so less of the voice under them is rebuilt.
- We made `denoise` gentler on old recordings: when no noise sample is given, its noise estimate under 500 Hz is no
  higher than each frequency's steady level, so less of the programme's bass is taken for noise; the rules driver
  uses it only when the request asks for hiss or noise removal, and more gently (mmse, 8 dB, noise_db -3).
- We added a change budget to the damage gate: hum and rumble are steady, so a `dehum` node or a low cut (a gain under
  0 dB on a band that ends at or under 200 Hz) may remove only a fixed amount more than the steady hum or low noise it
  targets; removing more is reported as taking the programme. The steady level of each bin is its power averaged over
  0.4 s, then its quietest tenth over the file.
- We added a voice clean-up preset (`tools/clean_up.py`, `zumzum/preset.py`) that chooses hum removal and a low cut
  only where its checks find steady hum or rumble, then applies one loudness gain.
- We changed `dehum` to a fixed notch: each hum line (f0 and its harmonics) is cut by `depth_db` for the whole file,
  the same in every frame, one bin either side of the line, and the default FFT is now 32768. Before, each cut
  followed the level of the neighbouring bins frame by frame. On 20 new test recordings with hum the fixed notch
  repaired more (median +9.8 dB against +6.4 dB) and left none worse than its input (the old cut left 3).
- We removed the `gate_db` and `gate_release_ms` options from `denoise`.

## 2026-11-11

- We made `tools/fast.py` measure the program's loudness, true-peak, loudness-range, RMS and hum checks and the
  damage gate too, from the files and in blocks, with the same results as the whole-file checks.
- Fixed: when the true-peak ceiling, not the target, decided the loudness gain, the result sat exactly on the
  ceiling, so a `true_peak` check at the same ceiling passed or failed by rounding. `loudness` now stays 0.01 dB under
  its ceiling.
- We added a faster path that uses little memory for simple chains (`tools/fast.py`, `zumzum/stream.py`): hum
  removal, a gain on a band and one loudness gain run from file to file in blocks, with the same output as the
  reference engine.
- The damage gate now refuses a version where `declick` was run on a clipped recording and mostly rewrote the
  clipped peaks. We found that this leaves a file further from the original than it was, while every check stayed
  silent. A few clicks that hit the ceiling on their own are not counted as clipping.

## 2026-11-10

- The rules driver (the loop without a model) now takes its nodes from the same phrase rules that set the checks. We
  added a rule for rumble (a fixed low cut under 75 Hz) and one that removes keypad and dial tones by healing the
  boxes around them. When the mains frequency is unknown it writes both the 50 Hz and the 60 Hz hum nodes.
- We added `loudness`: one gain that brings a layer to a loudness target, capped by a true-peak ceiling. It does not
  limit or compress. New checks: `loudness`, `true_peak`, `loudness_range`, `rms`, `noise_floor`.
- We added `tools/fix.py --spec` and `tools/batch.py`: bring a file, or a folder, to the published delivery targets
  of Apple Podcasts, ACX, EBU R 128 or ATSC A/85, and measure the delivered file (targets from their published
  specifications; not affiliated with or endorsed by Apple, Audible (ACX), the EBU or ATSC). A file that would need
  limiting is reported, not fixed.
- We added the repair record: every output carries the program, the check results and hashes of the source and the
  result, inside the file and beside it. `tools/verify.py` checks it and can run the program again.
- We added `attenuate`, which can only turn parts of a recording down, and the `attenuated` check that measures it.
- The rules that turn a request into checks now read phrases, not single words, and a request about a phone only
  counts tones at telephone frequencies. New report-only checks: `mains_hum_drop`, `filled`.
- We added a file driver for the editing loop (`ZUMZUM_DRIVER=file`): a person or another program writes each
  answer to a file.
- The hum check now says which way it passed. Drawing a sheet for a long file no longer needs tens of gigabytes
  of memory. Time spans and bands in a node's parameters are checked against the file.
- The README is short now; the full walkthrough moved to `docs/guide.md`.

## 2026-11-04

- Review fixes: each goal in a request takes its own amount, downloads and model replies are bounded, requests may
  span several lines, `convert.py` refuses to overwrite its outputs unless asked.

## 2026-10-29

- Zumzum 0.1: the language, the checker, the NumPy reference engine and six operations; limits on input size and a
  safer conversion of media files.
