"""A faster path that uses little memory, for simple chains: file in, file out, in blocks.

Supported: one layer whose chain is STFT nodes of kind `dehum` and `gain` (gain with a `band` selection or none),
then at most one `loudness` node. A stage is a dehum node, or gain nodes, plus the gain nodes right after it with the
same FFT size. The result is meant to equal the reference engine's
to far below anything audible (tests/test_stream.py holds it within -50 dB), not bit for bit:

- the nodes of a stage share ONE analysis and ONE synthesis (the reference goes back to samples between all nodes;
  two dehum nodes are not merged, because merged they would differ from the reference);
- the file is read, worked and written in blocks of frames, stage by stage through temporary files next to the
  output, so memory does not grow with its length;
- the loudness node is ONE gain: min(gain to the target, room under the true-peak ceiling). It measures loudness and
  true peak once, while the repaired signal streams to a temporary file, as the loudness op does: no limiter.

A dehum node's f0 is refined on the mean spectrum of its whole input (as in the reference), so each dehum node costs
one analysis sweep before its stage runs.
"""

import math
import os
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

from . import loudness
from .engine import TP_MARGIN_DB, dehum_gains, dehum_lines, refine_f0, sel_mask
from .stft import bin_freqs, n_frames, window

CHUNK_SAMPLES = 1 << 20  # about this many samples of frames are worked at a time (8 MB per array of doubles)


def _chunk_frames(size):
    return max(8, CHUNK_SAMPLES // size)


class Unsupported(ValueError):
    pass


def plan(prog):
    """-> (stages, loudness node or None) of a supported program; a stage is (fft size, [nodes]). Raises Unsupported."""
    if prog.output == "mix" or len(prog.layers) != 1:
        raise Unsupported("one layer only")
    chain = [prog.ops[n] for n in prog.layers[0][1] if not prog.ops[n]["off"]]
    loud = None
    if chain and chain[-1]["op"] == "loudness":
        loud, chain = chain[-1], chain[:-1]
    stages = []
    for op in chain:
        if op["op"] not in ("dehum", "gain"):
            raise Unsupported(f"op {op['op']} is not in the fast path")
        at = op["at"]
        if at is not None and not (at[0] not in ("not", "and", "or") and at[1] in ("band", "all") and not at[2].get("t_s")):
            raise Unsupported("only a band selection (no time span) in the fast path")
        if op["op"] == "dehum" and at is not None:
            raise Unsupported("dehum at a selection is not in the fast path")
        size = op["params"].get("fft") or prog.size
        if stages and stages[-1][0] == size and op["op"] == "gain":
            stages[-1][1].append(op)
        else:
            stages.append((size, [op]))
    return stages, loud


class Stage:
    """Nodes that share one STFT, fed with blocks of samples (channels, k); gives back finished samples in order."""

    def __init__(self, size, overlap, nch, n, nodes, states):
        self.size, self.ov, self.hop, self.nch, self.n = size, overlap, size // overlap, nch, n
        self.nodes, self.states = nodes, states
        self.w = window(size)
        self.ws = np.array([np.sum(self.w[j::self.hop] ** 2) for j in range(self.hop)])  # summed squared window
        self.total = n_frames(n, size, self.hop)
        self.buf, self.at, self.t = np.zeros((nch, size)), 0, 0  # input in padded samples (the file starts at `size`)
        self.carry = np.zeros((nch, size - self.hop))

    def feed(self, x, last=False):
        self.buf = np.concatenate([self.buf, x] + ([np.zeros((self.nch, self.size + self.hop))] if last else []), axis=1)
        out = []
        while self.t < self.total:
            have = (self.at + self.buf.shape[1] - self.size - self.t * self.hop) // self.hop + 1
            k = min(have, _chunk_frames(self.size), self.total - self.t)
            if k <= 0:
                break
            a = self.t * self.hop - self.at
            fr = np.lib.stride_tricks.sliding_window_view(self.buf[:, a:a + (k - 1) * self.hop + self.size], self.size,
                                                          axis=1)[:, ::self.hop]
            ola = np.zeros((self.nch, (k - 1) * self.hop + self.size))
            ola[:, :self.size - self.hop] += self.carry
            for c in range(self.nch):
                X = np.fft.rfft(fr[c] * self.w, axis=1)
                for j, op in enumerate(self.nodes):
                    X = _apply(X, op, self.states[j][c])
                y = np.fft.irfft(X, n=self.size, axis=1) * self.w
                for j in range(self.ov):  # frames j, j + overlap, ... do not overlap each other: one vector add
                    sub = y[j::self.ov]
                    ola[c, j * self.hop:j * self.hop + sub.shape[0] * self.size] += sub.reshape(-1)
            out.append(self._trim(ola[:, :k * self.hop], self.t * self.hop))
            self.carry = ola[:, k * self.hop:].copy()
            self.t += k
            keep = self.t * self.hop - self.at
            self.buf, self.at = self.buf[:, keep:], self.t * self.hop
        if last:
            out.append(self._trim(self.carry, self.t * self.hop))
        return np.concatenate(out, axis=1) if out else np.zeros((self.nch, 0))

    def _trim(self, y, at):
        lo, hi = max(at, self.size), min(at + y.shape[1], self.size + self.n)
        if hi <= lo:
            return np.zeros((self.nch, 0))
        return y[:, lo - at:hi - at] / self.ws[np.arange(lo, hi) % self.hop]


class Meter:
    """Integrated loudness and true peak of a signal fed in blocks (same definitions and blocking as loudness.py)."""

    def __init__(self, sr, nch):
        self.sr, self.nch = sr, nch
        self.f1 = loudness._at_rate(*loudness.SHELF_48K, sr, shelf=True)
        self.f2 = loudness._at_rate(*loudness.HIGHPASS_48K, sr, shelf=False)
        self.z1, self.z2 = np.zeros((nch, 2)), np.zeros((nch, 2))
        self.h, self.n = int(round(0.1 * sr)), int(round(0.4 * sr))
        if self.n != 4 * self.h:
            raise Unsupported("sample rate whose 400 ms block is not four 100 ms steps")
        self.left = np.zeros((nch, 0))
        self.steps = []
        self.factor = loudness._factor(sr)
        self.buf, self.buf_at, self.done, self.peak, self.total = np.zeros((nch, 0)), 0, 0, 0.0, 0

    def feed(self, x):
        y, self.z1 = signal.lfilter(*self.f1, x, axis=-1, zi=self.z1)
        y, self.z2 = signal.lfilter(*self.f2, y, axis=-1, zi=self.z2)
        y = np.concatenate([self.left, y * y], axis=1)
        k = y.shape[1] // self.h
        if k:
            self.steps.append(y[:, :k * self.h].reshape(self.nch, k, self.h).sum(axis=2))
        self.left = y[:, k * self.h:]
        self.buf = np.concatenate([self.buf, x], axis=1)
        self.total += x.shape[1]
        self._peaks(final=False)

    def _peaks(self, final):
        B = loudness.PEAK_BLOCK
        while True:
            a = self.done
            b = a + B
            if final:
                b = min(b, self.total)
                if a >= b:
                    return
            elif self.total < b + 32:
                return
            lo, hi = max(0, a - 32), min(self.total if final else b + 32, b + 32)
            seg = self.buf[:, lo - self.buf_at:hi - self.buf_at]
            if self.factor == 1:
                self.peak = max(self.peak, float(np.max(np.abs(seg[:, a - lo:b - lo]))))
            else:
                f = self.factor
                for ch in seg:
                    u = np.abs(signal.resample_poly(ch, f, 1))
                    base = u[:(len(u) // f) * f].reshape(-1, f).max(axis=1)
                    self.peak = max(self.peak, float(base[a - lo:a - lo + b - a].max()))
            self.done = b
            keep = max(0, b - 32)
            self.buf, self.buf_at = self.buf[:, keep - self.buf_at:], keep

    def result(self):
        """-> (integrated LUFS or None, true peak dBTP)."""
        self._peaks(final=True)
        tp = 20 * math.log10(self.peak) if self.peak > 0 else -math.inf
        if not self.steps:
            return None, tp
        s = np.concatenate(self.steps, axis=1)
        if s.shape[1] < 4:
            return None, tp
        ms = (s[:, :-3] + s[:, 1:-2] + s[:, 2:-1] + s[:, 3:]) / self.n
        z = loudness._weights(self.nch) @ ms
        with np.errstate(divide="ignore"):
            lk = -0.691 + 10 * np.log10(z)
        lk = lk[lk > loudness.ABS_GATE]
        if not len(lk):
            return None, tp
        rel = -0.691 + 10 * np.log10(np.mean(10 ** ((lk + 0.691) / 10))) + loudness.REL_GATE
        lk = lk[lk > rel]
        return float(-0.691 + 10 * np.log10(np.mean(10 ** ((lk + 0.691) / 10)))), tp


def _chunks(f, n, size, hop):
    """Yields (t0, frames) with frames (channels, k, size): the reference's analysis frames, unwindowed."""
    total = n_frames(n, size, hop)
    step = _chunk_frames(size)
    for t0 in range(0, total, step):
        t1 = min(total, t0 + step)
        a, b = t0 * hop - size, (t1 - 1) * hop  # this chunk's span in samples of the file (frames start `size` early)
        lo, hi = max(a, 0), min(b, n)
        if hi > lo:
            f.seek(lo)
            got = f.read(hi - lo, dtype="float64", always_2d=True).T
        else:
            got = np.zeros((f.channels, 0))
        block = np.zeros((f.channels, b - a))
        block[:, lo - a:lo - a + got.shape[1]] = got
        yield t0, np.lib.stride_tricks.sliding_window_view(block, size, axis=1)[:, ::hop]


def _apply(X, op, state):
    """One node on a chunk X (frames, bins) of one channel: a fixed gain per bin (gain, and dehum's one cut per line)."""
    Y = X * state
    return X + op["mix"] * (Y - X) if op["op"] == "dehum" and op["mix"] != 1 else Y


def _temp(dst):
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=os.path.dirname(os.path.abspath(str(dst))))
    os.close(fd)
    return tmp


def run(prog, src, dst, subtype="FLOAT"):
    """Checked program, source file -> destination file. Returns the info per node (f0, gain, loudness numbers)."""
    stages, loud = plan(prog)
    info, cur, temps, meter = {}, str(src), [], None
    if not stages:  # only a loudness node, or nothing: the "stage" is a copy
        stages = [(prog.size, [])]
    for si, (size, nodes) in enumerate(stages):
        final = si == len(stages) - 1
        with sf.SoundFile(cur) as f:
            sr, nch, n = f.samplerate, f.channels, f.frames
            hop, w, freqs = size // prog.overlap, window(size), bin_freqs(size, sr)
            states = [[None] * nch for _ in nodes]
            for i, op in enumerate(nodes):
                p = op["params"]
                if op["op"] == "gain":
                    M = np.ones(len(freqs)) if op["at"] is None else sel_mask(op["at"], freqs, np.zeros(1))[:, 0]
                    for c in range(nch):
                        states[i][c] = 1.0 + op["mix"] * M * (10 ** (p["db"] / 20) - 1.0)
                    continue
                # dehum (always first in its stage): its f0 from the mean spectrum of its input (one pass), then one
                # fixed gain per bin (a notch per hum line), applied like a gain
                if p["search_hz"] > 0:
                    P, frames = np.zeros((nch, len(freqs))), 0
                    for _, fr in _chunks(f, n, size, hop):
                        for c in range(nch):
                            X = np.fft.rfft(fr[c] * w, axis=1)
                            P[c] += (X.real ** 2 + X.imag ** 2).sum(axis=0)
                        frames += fr.shape[1]
                    f0s = [refine_f0(P[c] / max(frames, 1), freqs, p["f0_hz"], p["harmonics"], p["search_hz"])
                           for c in range(nch)]
                else:
                    f0s = [p["f0_hz"]] * nch
                for c in range(nch):
                    states[i][c] = dehum_gains(p, dehum_lines(p, f0s[c], freqs), len(freqs))
                info[f"stage{si}"] = {"op": "dehum", "f0_hz": [round(v, 3) for v in f0s]}
            stage = Stage(size, prog.overlap, nch, n, nodes, states)
            if final and loud:
                meter = Meter(sr, nch)
            to = str(dst) if final and not loud else _temp(dst)
            with sf.SoundFile(to, "w", samplerate=sr, channels=nch, subtype=subtype if to == str(dst) else "DOUBLE") as out:
                def emit(y):
                    if y.shape[1]:
                        out.write(y.T)
                        if meter:
                            meter.feed(y)
                f.seek(0)
                for block in f.blocks(blocksize=1 << 16, dtype="float64", always_2d=True):
                    emit(stage.feed(block.T) if nodes else block.T)
                if nodes:
                    emit(stage.feed(np.zeros((nch, 0)), last=True))
        if cur in temps:
            os.remove(cur)
        if to != str(dst):
            temps.append(to)
        cur = tmp = to
    if loud:
        p = loud["params"]
        lu, tp = meter.result()
        if lu is None:
            g, note = 0.0, "no loudness to measure (silent, or under 0.4 s): left as it was"
        else:
            want = p["target_lufs"] - lu
            g, note = min(want, p["max_true_peak_dbtp"] - tp - TP_MARGIN_DB), None
        info["loudness"] = {"lufs_before": None if lu is None else round(lu, 2), "gain_db": round(g, 2),
                            "short_by_lu": None if lu is None else round(max(0.0, want - g), 2),
                            "lufs": None if lu is None else round(lu + g, 2), "true_peak_dbtp": round(tp + g, 2), "note": note}
        k = 10 ** (g / 20)
        with sf.SoundFile(tmp) as a, sf.SoundFile(str(dst), "w", samplerate=sr, channels=nch, subtype=subtype) as b:
            for block in a.blocks(blocksize=1 << 16, dtype="float64", always_2d=True):
                b.write(block * k)
        os.remove(tmp)
    return info
