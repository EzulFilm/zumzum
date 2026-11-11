# Changelog

What changed in Zumzum, newest first. The language version stays 0.1 until its first breaking change.

## 2026-11-11

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
