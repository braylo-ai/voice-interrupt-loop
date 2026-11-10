"""Frame-driven voice loop state machine.

LISTENING --(speech then `endpoint_ms` silence)--> THINKING (STT -> reply -> TTS)
--> SPEAKING --(barge-in: `min_speech_ms` of user speech)--> LISTENING (TTS stopped,
the interrupting speech is kept as the start of the next user turn).
Time is the mic stream's sample clock; compute time of STT/TTS is measured and logged.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import SR
from .bargein import TTSPlayer


class VoiceLoop:
    def __init__(self, vad, stt: Callable[[np.ndarray], tuple[str, float]], reply: Callable[[str], str],
                 tts: Callable[[str], tuple[np.ndarray, float]], endpoint_ms: float = 500, min_speech_ms: float = 60,
                 echo_gain_est: float = 0.0, pre_roll_ms: float = 200):
        self.vad, self.stt, self.reply, self.tts = vad, stt, reply, tts
        self.F = vad.frame_samples
        fms = 1000 * self.F / SR
        self.endpoint_frames = max(1, int(np.ceil(endpoint_ms / fms)))
        self.barge_frames = max(1, int(np.ceil(min_speech_ms / fms)))
        self.pre_roll = int(pre_roll_ms * SR / 1000)
        self.g = echo_gain_est
        self.state = "LISTENING"
        self.events: list[dict] = []
        self.player: TTSPlayer | None = None
        self._buf: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self._heard_speech = False
        self.played = []  # TTS frames actually emitted (for echo simulation by the caller)

    def _log(self, t_sample: int, kind: str, **kw):
        self.events.append({"t_ms": round(1000 * t_sample / SR, 1), "event": kind, **kw})

    def speaker_frame(self) -> np.ndarray:
        """Audio the loop is playing during the next frame (caller feeds it back as echo)."""
        if self.state == "SPEAKING" and self.player is not None:
            return self.player.next_frame(self.F)
        return np.zeros(self.F, dtype=np.float32)

    def step(self, mic: np.ndarray, t_sample: int, speaker: np.ndarray | None = None) -> None:
        f = mic if speaker is None else mic - self.g * speaker
        speech = self.vad.is_speech(f.astype(np.float32))
        if speech and speaker is not None and self.state == "SPEAKING":
            # double-talk gate (same rule as BargeInDetector 'ref_sub_dtd')
            echo_rms = self.g * float(np.sqrt(np.mean(speaker ** 2)))
            speech = float(np.sqrt(np.mean(f ** 2))) > 0.5 * echo_rms + 0.01
        if self.state == "SPEAKING":
            self._speech_run = self._speech_run + 1 if speech else 0
            self._buf.append(mic)
            self._buf = self._buf[-self.barge_frames - 2:]
            if self._speech_run >= self.barge_frames:
                self.player.stop(t_sample + self.F)
                self._log(t_sample + self.F, "barge_in", tts_played_ms=round(1000 * self.player.pos / SR, 1))
                self.state = "LISTENING"
                self._heard_speech, self._silence_run = True, 0
                return
            if not self.player.playing:
                self._log(t_sample + self.F, "tts_done")
                self.state, self._buf = "LISTENING", []
            return
        # LISTENING
        self._buf.append(mic)
        if speech:
            self._heard_speech, self._silence_run = True, 0
        elif self._heard_speech:
            self._silence_run += 1
            if self._silence_run >= self.endpoint_frames:
                self._log(t_sample + self.F, "endpoint")
                self._think(t_sample + self.F)
        else:
            self._buf = self._buf[-max(1, self.pre_roll // self.F):]

    def _think(self, t_sample: int):
        audio = np.concatenate(self._buf)
        self._buf, self._heard_speech, self._silence_run = [], False, 0
        text, stt_ms = self.stt(audio)
        t0 = time.perf_counter()
        answer = self.reply(text)
        reply_ms = (time.perf_counter() - t0) * 1000
        wav, tts_s = self.tts(answer)
        self._log(t_sample, "turn", transcript=text, stt_ms=round(stt_ms, 1), reply=answer,
                  reply_ms=round(reply_ms, 2), tts_ms=round(tts_s * 1000, 1))
        self.player = TTSPlayer(wav)
        self.state = "SPEAKING"
        self._speech_run = 0
        self.vad.reset()
