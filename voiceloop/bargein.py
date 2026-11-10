"""Barge-in: stop TTS when the user speaks over it.

The mic hears user speech + the assistant's own TTS echo + noise. Echo
strategies: 'none' (raw mic to the VAD), 'ref_sub' (subtract the known TTS
reference with an estimated echo gain/delay: a crude stand-in for real AEC) and
'ref_sub_dtd' (ref_sub + a double-talk gate: the residual's RMS must exceed
`dtd_ratio` x the estimated echo RMS + a noise floor).
Detection fires after `min_speech_ms` of consecutive speech frames.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import SR


@dataclass
class Detection:
    fired: bool
    detect_sample: int | None = None    # sample index at end of the frame that fired
    compute_ms_at_fire: float | None = None
    frame_compute_ms: list[float] = field(default_factory=list)


class TTSPlayer:
    """Simulated playback: emits TTS audio frame by frame until stopped."""

    def __init__(self, audio: np.ndarray):
        self.audio = audio
        self.pos = 0
        self.stopped_at: int | None = None

    @property
    def playing(self) -> bool:
        return self.stopped_at is None and self.pos < len(self.audio)

    def next_frame(self, n: int) -> np.ndarray:
        if not self.playing:
            return np.zeros(n, dtype=np.float32)
        f = self.audio[self.pos:self.pos + n]
        self.pos += n
        return np.pad(f, (0, n - len(f)))

    def stop(self, at_sample: int):
        if self.stopped_at is None:
            self.stopped_at = at_sample


class BargeInDetector:
    def __init__(self, vad, min_speech_ms: float = 60.0, echo: str = "none", echo_gain_est: float = 0.0,
                 echo_delay_est: int = 0, dtd_ratio: float = 0.5, noise_floor: float = 0.01):
        if echo not in ("none", "ref_sub", "ref_sub_dtd"):
            raise ValueError("echo must be 'none', 'ref_sub' or 'ref_sub_dtd'")
        self.vad = vad
        self.frame = vad.frame_samples
        frame_ms = 1000 * self.frame / SR
        self.need = max(1, int(np.ceil(min_speech_ms / frame_ms)))
        self.echo = echo
        self.g = echo_gain_est
        self.d = echo_delay_est
        self.dtd_ratio = dtd_ratio
        self.noise_floor = noise_floor

    def run(self, mic: np.ndarray, tts_ref: np.ndarray | None = None, start_sample: int = 0) -> Detection:
        """Stream `mic` from `start_sample` (the moment TTS starts) and return the first barge-in."""
        self.vad.reset()
        run = 0
        times = []
        ref = np.zeros(len(mic), dtype=np.float32) if tts_ref is None else tts_ref
        for s in range(start_sample, len(mic) - self.frame + 1, self.frame):
            f = mic[s:s + self.frame]
            t0 = time.perf_counter()
            r = None
            if self.echo != "none":
                lo = s - self.d
                r = ref[max(0, lo):max(0, lo + self.frame)]
                r = np.pad(r, (self.frame - len(r), 0)) if lo < 0 else np.pad(r, (0, self.frame - len(r)))
                f = f - self.g * r
            sp = self.vad.is_speech(f.astype(np.float32))
            if sp and self.echo == "ref_sub_dtd":
                # Double-talk gate: residual must clearly exceed what leftover echo could explain.
                echo_rms = self.g * float(np.sqrt(np.mean(r ** 2)))
                sp = float(np.sqrt(np.mean(f ** 2))) > self.dtd_ratio * echo_rms + self.noise_floor
            run = run + 1 if sp else 0
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            if run >= self.need:
                return Detection(True, s + self.frame, dt, times)
        return Detection(False, None, None, times)
