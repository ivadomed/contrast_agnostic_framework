#!/usr/bin/env python3
"""
Synthesize the soundtrack for the PALETTE-Aug video (no samples, no licences): a 120 BPM
groove (pad, bass, kick, snare, hats, arpeggio) whose sections follow the scenes, plus sound
effects placed on the exact frames of on-screen events. Every timing comes from
src/timeline.json, the same file the Remotion composition reads, so picture and sound
cannot drift apart.

Usage (inside a job): python sound_design.py --timeline src/timeline.json --out public/audio/soundtrack.wav
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- helpers
def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def env(n, a=0.005, d=0.1, s=0.0, r=0.05, hold=0.0):
    """ADSR envelope of n samples (times in seconds)."""
    t = np.arange(n) / SR
    e = np.zeros(n)
    a_ = max(a, 1e-4)
    e = np.where(t < a_, t / a_, e)
    dt = t - a_
    e = np.where((t >= a_) & (dt < d), 1 - (1 - s) * dt / max(d, 1e-4), e)
    e = np.where((t >= a_ + d) & (t < a_ + d + hold), s, e)
    rt = t - (a_ + d + hold)
    e = np.where(t >= a_ + d + hold, s * np.clip(1 - rt / max(r, 1e-4), 0, 1), e)
    return e


def saw(freq, n, detune=0.0):
    t = np.arange(n) / SR
    ph = (t * freq * (1 + detune)) % 1.0
    return 2 * ph - 1


def note_hz(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    pitch, octave = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((names[pitch] + 12 * (octave - 4)) / 12)


class Track:
    def __init__(self, seconds):
        self.n = int(seconds * SR)
        self.l = np.zeros(self.n)
        self.r = np.zeros(self.n)

    def add(self, sig, t0, gain=1.0, pan=0.0):
        i0 = int(round(t0 * SR))
        if i0 >= self.n or i0 + len(sig) <= 0:
            return
        s0 = max(0, -i0)
        i0 = max(0, i0)
        seg = sig[s0:s0 + self.n - i0] * gain
        gl, gr = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        self.l[i0:i0 + len(seg)] += seg * gl * 1.414
        self.r[i0:i0 + len(seg)] += seg * gr * 1.414

    def stereo(self):
        return np.stack([self.l, self.r], axis=1)


# ---------------------------------------------------------------- instruments
def kick():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 45 + 95 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    click = hp(RNG.standard_normal(n), 2000) * np.exp(-t / 0.004) * 0.3
    return (np.sin(ph) * np.exp(-t / 0.16) + click) * 0.9


def snare():
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    noise = bp(RNG.standard_normal(n), 1200, 6000) * np.exp(-t / 0.07)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04)
    return noise * 0.7 + tone * 0.5


def hat(open_=False):
    n = int((0.18 if open_ else 0.05) * SR)
    t = np.arange(n) / SR
    return hp(RNG.standard_normal(n), 7500) * np.exp(-t / (0.06 if open_ else 0.012))


def pad_chord(notes, dur):
    n = int(dur * SR)
    sig = np.zeros(n)
    for nm in notes:
        f = note_hz(nm)
        for dt in (-0.004, 0.0, 0.005):
            sig += saw(f, n, dt)
    sig = lp(sig / (3 * len(notes)), 1400)
    return sig * env(n, a=0.35, d=0.2, s=0.8, r=0.6, hold=max(0.0, dur - 1.15))


def bass_note(nm, dur):
    n = int(dur * SR)
    f = note_hz(nm)
    sig = saw(f, n) * 0.6 + np.sin(2 * np.pi * f * np.arange(n) / SR) * 0.6
    return lp(sig, 420) * env(n, a=0.004, d=dur * 0.7, s=0.25, r=0.05)


def pluck(f, dur=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    tri = 2 * np.abs(2 * ((t * f) % 1) - 1) - 1
    return (np.sin(2 * np.pi * f * t) * 0.6 + tri * 0.4) * np.exp(-t / 0.09)


# ---------------------------------------------------------------- sound effects
def whoosh(dur, lo=300, hi=5000):
    n = int(dur * SR)
    noise = RNG.standard_normal(n)
    out = np.zeros(n)
    # time-varying band: process in short blocks with a rising centre frequency
    blk = 512
    for b in range(0, n, blk):
        p = b / max(1, n - 1)
        c = lo * (hi / lo) ** p
        seg = noise[max(0, b - 2048):b + blk]
        y = bp(seg, c * 0.6, min(c * 1.6, SR / 2 - 100))
        out[b:b + blk] = y[-len(out[b:b + blk]):]
    shape = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    return out * shape


def chime(f=1318.5):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 1.5 * t) + 0.25 * np.sin(2 * np.pi * f * 2 * t)) * np.exp(-t / 0.25) * 0.5


def thud():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 55 + 60 * np.exp(-t / 0.05)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
    grit = lp(RNG.standard_normal(n), 600) * np.exp(-t / 0.06) * 0.4
    return body + grit


def tick(f=2200):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.012)


def pop(f0=900, f1=480):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.02)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05)


def impact():
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    f = 38 + 80 * np.exp(-t / 0.06)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.6)
    air = lp(RNG.standard_normal(n), 3000) * np.exp(-t / 0.25) * 0.35
    return boom + air


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 220 * (4 ** (t / dur))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3
    return (tone + whoosh(dur, 400, 7000) * 0.8) * (t / dur) ** 2


def reverb(x, seconds=1.2, mix=0.18):
    n = int(seconds * SR)
    ir = RNG.standard_normal(n) * np.exp(-np.arange(n) / SR / (seconds / 5))
    ir = lp(ir, 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = fftconvolve(x, ir[:, None] if x.ndim == 2 else ir, axes=0)[:len(x)]
    return x * (1 - mix) + wet * mix


# ---------------------------------------------------------------- timing helpers mirrored from Main.tsx
def iter_count(g, P0, P1, RAMP):
    gg = min(g, RAMP)
    n = -(RAMP / (P0 - P1)) * math.log((P0 - (P0 - P1) * gg / RAMP) / P0)
    return n + max(0, g - RAMP) / P1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    TL = json.loads(Path(a.timeline).read_text())
    fps, bpm = TL["fps"], TL["bpm"]
    beat = 60.0 / bpm
    starts, acc = {}, 0
    for k, d in TL["scenes"]:
        starts[k] = (acc, d)
        acc += d
    total_s = acc / fps
    F = lambda frame: frame / fps                     # frame -> seconds
    sc = lambda k: (starts[k][0] / fps, (starts[k][0] + starts[k][1]) / fps)

    music, sfx = Track(total_s + 2.5), Track(total_s + 2.5)

    # ---------------- music: sections follow the scenes
    prog = [("A", ["A3", "C4", "E4"], "A1"), ("F", ["F3", "A3", "C4"], "F1"),
            ("C", ["C3", "E3", "G3"], "C2"), ("G", ["G3", "B3", "D4"], "G1")]
    bar = 4 * beat
    n_bars = int(math.ceil(total_s / bar))
    title_t = F(TL["mosaicTitle"])
    groove = {"gallery", "draws", "bench", "ablation"}

    def scene_at(t):
        for k, (s0, d) in starts.items():
            if s0 / fps <= t < (s0 + d) / fps:
                return k
        return "end"

    for b in range(n_bars):
        t0 = b * bar
        _, chord, root = prog[b % 4]
        k = scene_at(t0 + 0.01)
        if k == "end" and t0 > sc("end")[0] + 0.5:
            continue
        music.add(pad_chord(chord, bar + 0.6), t0, gain=0.22 if k != "mosaic" else 0.16)
        for q in range(4):                                   # quarter-note grid
            tq = t0 + q * beat
            kq = scene_at(tq + 0.01)
            if kq == "end":
                continue
            drums = tq >= title_t - 0.01 and kq != "texture"
            if drums:
                music.add(kick(), tq, gain=0.75 if kq in groove else 0.5)
            if kq in groove and q in (1, 3):
                music.add(snare(), tq, gain=0.32, pan=0.1)
            if kq in groove | {"problem", "transform", "texture"}:
                music.add(hat(), tq + beat / 2, gain=0.12 if kq != "texture" else 0.07, pan=0.35)
                if kq in groove | {"transform"}:
                    music.add(hat(), tq + beat / 4, gain=0.06, pan=-0.3)
                    music.add(hat(), tq + 3 * beat / 4, gain=0.06, pan=-0.3)
            if kq in groove | {"transform", "texture"}:
                for e8 in range(2):                          # eighth-note bass
                    music.add(bass_note(root if e8 == 0 else root[:-1] + str(int(root[-1]) + 1), beat / 2), tq + e8 * beat / 2, gain=0.3)
            if kq in {"gallery", "draws"}:                   # sixteenth arpeggio
                notes = [note_hz(chord[i % 3][:-1] + str(int(chord[i % 3][-1]) + 1)) for i in range(4)]
                for s16 in range(4):
                    music.add(pluck(notes[(q * 4 + s16) % 4]), tq + s16 * beat / 4, gain=0.07, pan=(-0.4 if s16 % 2 else 0.4))
    # final chord under the end card
    e0, e1 = sc("end")
    music.add(pad_chord(["A3", "C4", "E4", "A4"], e1 - e0 + 2.0), e0, gain=0.3)
    music.add(bass_note("A1", 2.5), e0, gain=0.35)

    # ---------------- sound effects on on-screen events
    m0 = sc("mosaic")[0]
    music.add(riser(title_t - 0.4), 0.4, gain=0.25)
    sfx.add(impact(), m0 + title_t, gain=0.7)

    p0 = sc("problem")[0]
    P = TL["problem"]
    sfx.add(whoosh(F(P["pass1"][1] - P["pass1"][0])), p0 + F(P["pass1"][0]), gain=0.35)
    sfx.add(chime(), p0 + F(P["pred"]), gain=0.45)
    if "move" in P:
        sfx.add(whoosh(F(P["move"][1] - P["move"][0]), 250, 2500), p0 + F(P["move"][0]), gain=0.3)
    for j in range(3):
        base = P["start2"] + j * P["seg"]
        sfx.add(whoosh(F(P["pass2"][1] - P["pass2"][0])), p0 + F(base + P["pass2"][0]), gain=0.3)
        sfx.add(thud(), p0 + F(base + P["pass2"][1]), gain=0.6)

    t0 = sc("transform")[0]
    k = TL["transformSpeed"]
    for step in TL["transformSteps"][1:]:
        sfx.add(tick(1760), t0 + F(step / k), gain=0.35)
        sfx.add(pop(700, 420), t0 + F(step / k), gain=0.2)
    sw = TL["transformSweep"]
    sfx.add(whoosh(F((sw[1] - sw[0]) / k), 500, 8000), t0 + F(sw[0] / k), gain=0.3)

    g0 = sc("gallery")[0]
    G = TL["gallery"]
    pent = [note_hz(n) for n in ["A5", "C6", "D6", "E6", "G6", "A6"]]
    gdur = starts["gallery"][1]
    for i in range(7):
        ts = G["t0"] + i * G["stagger"]
        sfx.add(whoosh(F(G["scan"]), 800, 6000), g0 + F(ts), gain=0.12, pan=-0.6 + 0.2 * i)
        fr = ts + G["scan"] + G["period"] - G["xf"]
        j = 0
        while fr < gdur - 8:
            sfx.add(pluck(pent[(i + j) % len(pent)], 0.2), g0 + F(fr), gain=0.05, pan=-0.6 + 0.2 * i)
            fr += G["period"]
            j += 1

    d0 = sc("draws")[0]
    D = TL["draws"]
    ddur = starts["draws"][1]
    scale = [note_hz(n) for n in ["A4", "C5", "D5", "E5", "G5", "A5", "C6", "D6", "E6", "G6", "A6"]]
    prev_k = -1
    for g in range(1, ddur - D["start"]):
        n = iter_count(g, D["P0"], D["P1"], D["RAMP"])
        kk = int(math.floor(n))
        if kk != prev_k:                                     # a new iteration starts on this frame
            fr = D["start"] + g
            sfx.add(tick(scale[min(kk, len(scale) - 1)] * 2), d0 + F(fr), gain=0.3)
            period = D["P0"] - (D["P0"] - D["P1"]) * min(g, D["RAMP"]) / D["RAMP"]
            sfx.add(pluck(scale[min(kk, len(scale) - 1)], 0.15), d0 + F(fr + D["pred"][0] * period), gain=0.08)
            prev_k = kk

    B = TL["bars"]
    for scene, nbars in (("bench", TL["bench"]), ("ablation", TL["ablation"])):
        s0 = sc(scene)[0]
        for i in range(nbars):
            sfx.add(pop(800 + 60 * i, 450 + 30 * i), s0 + F(B["delay"] + i * B["step"] + 6), gain=0.35, pan=-0.5 + i / max(1, nbars - 1))
            sfx.add(chime(1760 + 40 * i), s0 + F(B["star"] + i * B["step"]), gain=0.12, pan=-0.5 + i / max(1, nbars - 1))

    x0 = sc("texture")[0]
    sfx.add(whoosh(1.0, 200, 3000), x0, gain=0.25)
    sfx.add(impact(), sc("end")[0], gain=0.55)

    # ---------------- mix
    mix = reverb(music.stereo(), 1.4, 0.16) + reverb(sfx.stereo(), 0.9, 0.12)
    mix = mix[: int((total_s + 1.0) * SR)]
    n = len(mix)
    fade_in = np.clip(np.arange(n) / (0.3 * SR), 0, 1)
    fade_out = np.clip((n - np.arange(n)) / (1.8 * SR), 0, 1)
    mix *= (fade_in * fade_out)[:, None]
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    mix *= 0.89 / max(1e-6, np.max(np.abs(mix)))           # -1 dBFS peak
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(a.out, SR, (mix * 32767).astype(np.int16))
    print(f"[audio] {a.out} {n / SR:.1f}s")


if __name__ == "__main__":
    main()
