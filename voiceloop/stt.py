"""faster-whisper STT: endpointed final transcript plus optional growing-buffer partials."""
from __future__ import annotations

import re
import time

import numpy as np


class WhisperSTT:
    def __init__(self, size: str = "small", compute_type: str = "int8", device: str = "cpu"):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper-{size}-{compute_type}-{device}"
        self.model = WhisperModel(size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        segs, _ = self.model.transcribe(audio.astype(np.float32), language="en", beam_size=1,
                                        vad_filter=False, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, (time.perf_counter() - t0) * 1000


def normalize(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(1, len(r))
