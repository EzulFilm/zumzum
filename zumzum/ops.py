"""The one table of selections, ops and checks: names, parameters, types, units, ranges, defaults. The checker and the
language reference given to a model are built from it, and docs/spec.md describes it.

A parameter is (kind, lo, hi, default, text). kind: "num" (finite real), "int", "enum" (lo = tuple of choices),
"range" (lo..hi with lo <= hi, both in [lo, hi]), "sel" (name of a `sel` statement). default None = required;
default OPTIONAL = may be left out (the op then says what it does without it).
Units are in the parameter name: _s seconds, _ms milliseconds, _hz hertz, _db decibels, _dbfs dB re full scale.
"""

OPTIONAL = "optional"
FFT_SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
T_MAX = 36000.0      # s: ten hours, the longest time any program may name
F_MAX = 96000.0      # Hz: half of 192 kHz


def num(lo, hi, default, text):
    return ("num", lo, hi, default, text)


def int_(lo, hi, default, text):
    return ("int", lo, hi, default, text)


def enum(choices, default, text):
    return ("enum", tuple(choices), None, default, text)


def rng(lo, hi, default, text):
    return ("range", lo, hi, default, text)


def sel(default, text):
    return ("sel", None, None, default, text)


ANALYSIS = {
    "stft": {
        "about": "analysis grid every STFT op uses unless it sets its own `fft`",
        "params": {
            "size": enum(FFT_SIZES, 2048, "FFT size in samples (Hann window)"),
            "overlap": enum((2, 4, 8), 4, "frames per window length (hop = size / overlap)"),
        },
    },
}

SELS = {
    "all": {"about": "the whole file", "params": {}},
    "time": {"about": "every frequency inside a time span",
             "params": {"t_s": rng(0, T_MAX, None, "start..end in seconds"),
                        "feather_ms": num(0, 2000, 0, "soft edge width in time")}},
    "band": {"about": "a frequency band over the whole file",
             "params": {"f_hz": rng(0, F_MAX, None, "low..high in hertz"),
                        "feather_hz": num(0, 4000, 0, "soft edge width in frequency")}},
    "box": {"about": "a time span and a frequency band together",
            "params": {"t_s": rng(0, T_MAX, None, "start..end in seconds"),
                       "f_hz": rng(0, F_MAX, None, "low..high in hertz"),
                       "feather_ms": num(0, 2000, 0, "soft edge width in time"),
                       "feather_hz": num(0, 4000, 0, "soft edge width in frequency")}},
    "harmonics": {"about": "narrow bands at a fundamental and its integer multiples",
                  "params": {"f0_hz": num(20, 8000, None, "fundamental frequency"),
                             "count": int_(1, 40, 10, "number of harmonics, the fundamental included"),
                             "width_hz": num(0.5, 400, 6, "width of each band"),
                             "t_s": rng(0, T_MAX, OPTIONAL, "start..end in seconds (whole file if left out)")}},
}

OPS = {
    "gain": {"domain": "stft", "about": "raise or lower the selected region",
             "params": {"db": num(-120, 24, None, "gain in dB (negative attenuates)")}},
    "dehum": {"domain": "stft",
              "about": "remove mains hum: narrow cuts at f0 and its harmonics, each cut down to the level of the "
                       "neighbouring bins, never more than depth_db; depth is fixed, not signal-dependent",
              "params": {"f0_hz": num(20, 400, None, "hum fundamental (50 or 60 for mains)"),
                         "harmonics": int_(1, 40, 10, "number of harmonics, the fundamental included"),
                         "depth_db": num(0, 80, 40, "largest cut"),
                         "width_hz": num(0.5, 20, 3, "width of each cut"),
                         "search_hz": num(0, 2, 0.5, "refine f0 within +- this much (one value for the file)"),
                         "fft": enum(FFT_SIZES, 16384, "FFT size for this op")}},
    "denoise": {"domain": "stft",
                "about": "broadband noise reduction from a noise estimate (Ephraim-Malah MMSE or log-MMSE, Wiener or "
                         "spectral subtraction)",
                "params": {"mode": enum(("mmse", "logmmse", "wiener", "subtract"), "mmse", "estimator"),
                           "profile": sel(OPTIONAL, "a sel holding noise only (a box, a time span, several with "
                                                    "or): the noise sample to learn from"),
                           "profile_t_s": rng(0, T_MAX, OPTIONAL,
                                              "a span with noise only; left out = estimated from the quietest frames"),
                           "reduce_db": num(0, 40, 15, "largest reduction (gain floor)"),
                           "smooth": num(0.5, 0.999, 0.98, "decision-directed smoothing of the a-priori SNR"),
                           "over": num(1, 6, 2, "over-subtraction factor (subtract mode only)"),
                           "noise_db": num(-10, 20, 0, "raise (or lower) the noise estimate before the gain is computed"),
                           "track_s": num(0, 10, 0, "follow a changing noise: running minimum over this window (0 = "
                                                     "one fixed noise estimate)"),
                           "smooth_hz": num(0, 500, 0, "average the gain over this width in frequency (less musical "
                                                       "noise; 0 = off)"),
                           "smooth_ms": num(0, 200, 0, "average the gain over this long in time (0 = off)"),
                           "gate_db": num(0, 40, 0, "push noise-only moments down this much more (0 = off)"),
                           "gate_release_ms": num(10, 2000, 200, "how slowly the gate closes after sound stops"),
                           "fft": enum((0,) + FFT_SIZES, 0, "FFT size for this op (0 = the analysis size)")}},
    "declick": {"domain": "time",
                "about": "find clicks from the error of an autoregressive model and refill them by least-squares AR "
                         "interpolation (Janssen 1986); only the time span of `at` is used",
                "params": {"sensitivity": num(3, 50, 10, "threshold over the local error level (lower finds more)"),
                           "max_ms": num(0.05, 20, 3, "longest click to repair"),
                           "order": int_(4, 64, 24, "AR model order")}},
    "heal": {"domain": "stft",
             "about": "replace the selected region with an estimate from its surroundings (left and right in time, "
                      "or below and above in frequency, or both)",
             "params": {"direction": enum(("time", "freq", "both"), "time", "where the estimate comes from"),
                        "context_ms": num(5, 1000, 60, "how much audio on each side the estimate uses"),
                        "phase": enum(("keep", "continue"), "keep",
                                      "keep the region's own phase, or continue the phase from before it (tonal "
                                      "sounds)"),
                        "fft": enum((0,) + FFT_SIZES, 0, "FFT size for this op (0 = the analysis size)")},
             "needs_at": True},
    "split_hp": {"domain": "stft",
                 "about": "keep only the harmonic (sustained) or percussive (transient) part, by median filtering "
                          "(FitzGerald 2010); no trained model",
                 "params": {"part": enum(("harmonic", "percussive"), None, "which part to keep"),
                            "rest_db": num(-120, 0, -120, "level of the other part (-120 removes it, -12 turns it "
                                                          "down 12 dB)"),
                            "time_ms": num(50, 2000, 200, "median length along time"),
                            "freq_hz": num(50, 4000, 500, "median length along frequency"),
                            "fft": enum((0,) + FFT_SIZES, 0, "FFT size for this op (0 = the analysis size)")}},
}

CHECKS = {
    "drop": {"about": "energy in the selection fell by at least min_db",
             "params": {"at": sel(None, "a sel name"), "min_db": num(0, 120, None, "required drop")}},
    "change": {"about": "energy in the selection changed by target_db, within +-tol_db (for a requested amount: "
                        "\"down about 12 dB\" is target -12)",
               "params": {"at": sel(None, "a sel name"), "target_db": num(-120, 60, None, "requested change"),
                          "tol_db": num(0, 40, 3, "how close")}},
    "keep": {"about": "energy in the selection stayed within tol_db, in every octave band of it that holds sound",
             "params": {"at": sel(None, "a sel name"), "tol_db": num(0, 20, 1, "largest change allowed")}},
    "untouched": {"about": "what changed inside the selection is at most max_db relative to what was there",
                  "params": {"at": sel(None, "a sel name"), "max_db": num(-120, 0, -40, "difference / original")}},
    "hum_drop": {"about": "tonal hum at f0 and its harmonics (energy above the neighbouring bins) fell by min_db, or "
                          "is no longer above the neighbouring bins",
                 "params": {"f0_hz": num(20, 400, None, "hum fundamental"),
                            "harmonics": int_(1, 40, 10, "harmonics measured"),
                            "min_db": num(0, 120, None, "required drop")}},
    "clicks": {"about": "at most max clicks are left (same detector as declick)",
               "params": {"max": int_(0, 100000, 0, "clicks allowed"),
                          "sensitivity": num(3, 50, 10, "detector threshold")}},
    "musical_noise": {"about": "EXPERIMENTAL, not validated (it did not flag plain spectral subtraction): the noise left in a noise-only span is not more 'twinkly' than before: kurtosis of its "
                               "spectral power after / before is at most max_ratio (musical noise raises it)",
                      "params": {"at": sel(None, "a sel name: a span with noise only (a pause)"),
                                 "max_ratio": num(1, 10, 1.5, "largest kurtosis ratio")}},
    "peak": {"about": "the sample peak is at most max_dbfs (no clipping)",
             "params": {"max_dbfs": num(-60, 0, -0.1, "highest sample peak")}},
}

LIMITS = {
    "source_bytes": 32 * 1024,
    "sels": 32, "ops": 32, "layers": 8, "checks": 32,
    "op_uses": 64,            # op uses over all layers (each is at most one STFT pass)
    "sel_calls": 16,          # selection calls in one expression as written
    "sel_calls_expanded": 64,  # ... once named selections are expanded
    "nesting": 16,
    "sr_min": 8000, "sr_max": 192000,  # Hz, for any audio a program runs on
    "channels": 8,
    "duration_s": 3600,       # one hour
    "samples": 8 * 48000 * 3600,  # channels x frames in all: an hour of 7.1 film sound at 48 kHz (11 GB as float64)
}
