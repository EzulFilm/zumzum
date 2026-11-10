"""Delivery rule sets for `fix --spec` . Each rule set is the published numbers, where they come from,
how the delivered file is encoded, and how a Zumzum program that meets it is written. The program is ordinary Zumzum
(a loudness node and the rule's own requirements as checks), so what
was done to the file can be read and re-run.

Sources (read 2026-11-10):
- Apple Podcasts, "Audio requirements" (podcasters.apple.com/support/audio-requirements): about -16 LKFS +- 1 dB,
  true peak not over -1 dBFS, both per ITU-R BS.1770; level before encoding.
- ACX (Audible), "Audio submission requirements" (help.acx.com): RMS -23..-18 dB, peaks under -3 dB, noise floor
  no higher than -60 dB RMS; MP3 192 kb/s or higher, constant bit rate, 44.1 kHz.
- EBU R 128 (tech.ebu.ch/docs/r/r128.pdf): programme loudness -23.0 LUFS, +- 0.5 LU (+- 1 LU where exact
  normalisation is not practical, e.g. live), maximum true peak -1 dBTP.
- ATSC A/85:2013 (atsc.org): target -24 LKFS, a commonly applied +- 2 dB tolerance, true peak -2 dBTP.
"""

SPECS = {
    "podcast-apple": {
        "title": "Apple Podcasts", "kind": "lufs", "target": -16.0, "tol": 1.0, "true_peak": -1.0,
        "encode": [".mp3", "-c:a", "libmp3lame", "-b:a", "192k"],
        "source": "Apple Podcasts audio requirements: about -16 LKFS +- 1 dB, true peak <= -1 dBFS (BS.1770)"},
    "audiobook-acx": {
        "title": "ACX / Audible audiobook", "kind": "rms", "target": -20.0, "range": (-23.0, -18.0),
        "true_peak": -3.0, "noise_floor": -60.0,
        "encode": [".mp3", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "192k", "-abr", "0"],
        "source": "ACX audio submission requirements: RMS -23..-18 dB, peaks < -3 dB, noise floor <= -60 dB RMS, "
                  "MP3 192 kb/s CBR 44.1 kHz"},
    "ebu-r128": {
        "title": "EBU R 128 broadcast", "kind": "lufs", "target": -23.0, "tol": 0.5, "true_peak": -1.0,
        "encode": [".wav", "-c:a", "pcm_s24le"],
        "source": "EBU R 128: -23.0 LUFS +- 0.5 LU, true peak <= -1 dBTP"},
    "atsc-a85": {
        "title": "ATSC A/85 broadcast (US)", "kind": "lufs", "target": -24.0, "tol": 2.0, "true_peak": -2.0,
        "encode": [".wav", "-c:a", "pcm_s24le"],
        "source": "ATSC A/85:2013: -24 LKFS (+- 2 dB commonly applied), true peak <= -2 dBTP"},
}

TEST_TOL = 0.5  # the test: loudness (ACX: RMS) within +-0.5 of the target on every file, stricter than every rule


def program(name, before, margin_db=0.5, shift_db=0.0):
    """Zumzum source that brings a file to rule set `name`. `before` = measure_all() of the input. The gain's
    ceiling sits margin_db under the rule's true peak (an encoder can add some). ACX has no loudness rule: its RMS
    target becomes a loudness target through the file's own LUFS - RMS difference. shift_db moves the level target
    by what a delivered file missed it by (fix() measures the encoded file). A noise floor over the rule's limit is
    reported, not repaired: this program changes level and peaks only."""
    s = SPECS[name]
    if s["kind"] == "rms":
        target = before["lufs"] + (s["target"] - before["rms_dbfs"])
    else:
        target = s["target"]
    target += shift_db
    ceiling = s["true_peak"] - margin_db
    lines = ["zumzum 0.1", f'name "fix --spec {name}: {s["title"]}"',
             f"node level = loudness(target_lufs: {target:.2f}, max_true_peak_dbtp: {ceiling:.2f})"]
    lines.append("layer main = level")
    if s["kind"] == "rms":
        lo, hi = s["range"]
        lines.append(f"check spec_rms = rms(min_dbfs: {lo:g}, max_dbfs: {hi:g})")
        lines.append(f"check spec_noise_floor = noise_floor(max_dbfs: {s['noise_floor']:g})")
    else:
        lines.append(f"check spec_loudness = loudness(target_lufs: {s['target']:g}, tol_lu: {s['tol']:g})")
    lines.append(f"check spec_true_peak = true_peak(max_dbtp: {s['true_peak']:g})")
    return "\n".join(lines) + "\n"


def measure_all(x, sr):
    """The numbers every rule set is judged on."""
    from . import loudness
    from .measure import noise_floor_dbfs, rms_dbfs
    lu = loudness.integrated(x, sr)
    return {"lufs": None if lu is None else round(lu, 2), "true_peak_dbtp": round(loudness.true_peak(x, sr), 2),
            "rms_dbfs": round(rms_dbfs(x), 2), "noise_floor_dbfs": _r(noise_floor_dbfs(x, sr)),
            "lra_lu": _r(loudness.loudness_range(x, sr))}


def _r(v):
    return None if v is None else round(v, 2)


def verdict(name, m):
    """Per requirement of the rule set (its own tolerances), and the stricter test (+-0.5): {"rule": {...}, "test": {...}}."""
    s = SPECS[name]
    rule, test = {}, {}
    tp_ok = m["true_peak_dbtp"] <= s["true_peak"]
    rule["true_peak"] = test["true_peak"] = tp_ok
    if s["kind"] == "rms":
        lo, hi = s["range"]
        rule["rms"] = lo <= m["rms_dbfs"] <= hi
        test["rms_within_0.5"] = abs(m["rms_dbfs"] - s["target"]) <= TEST_TOL
        rule["noise_floor"] = test["noise_floor"] = (m["noise_floor_dbfs"] is not None
                                                     and m["noise_floor_dbfs"] <= s["noise_floor"])
    else:
        ok = m["lufs"] is not None
        rule["loudness"] = ok and abs(m["lufs"] - s["target"]) <= s["tol"]
        test["loudness_within_0.5"] = ok and abs(m["lufs"] - s["target"]) <= TEST_TOL
    return {"rule": rule, "test": test}
