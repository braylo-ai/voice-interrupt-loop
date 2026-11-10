"""Streaming VADs with one interface: frame_samples, is_speech(frame) -> bool, reset()."""
from __future__ import annotations

import numpy as np

from . import SR


class SileroVAD:
    """Silero VAD (ONNX). Fixed 512-sample (32 ms) frames at 16 kHz."""

    name = "silero"
    frame_samples = 512

    def __init__(self, threshold: float = 0.5):
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self.model = load_silero_vad(onnx=True)
        self.threshold = threshold
        self.last_prob = 0.0

    def reset(self):
        self.model.reset_states()

    def is_speech(self, frame: np.ndarray) -> bool:
        self.last_prob = float(self.model(self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)), SR))
        return self.last_prob >= self.threshold


class WebRTCVAD:
    """Google WebRTC VAD (GMM). 30 ms frames, aggressiveness 0-3."""

    name = "webrtc"
    frame_samples = 480

    def __init__(self, aggressiveness: int = 3):
        import webrtcvad

        self.vad = webrtcvad.Vad(aggressiveness)

    def reset(self):
        pass

    def is_speech(self, frame: np.ndarray) -> bool:
        pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes()
        return self.vad.is_speech(pcm, SR)


class EnergyVAD:
    """Trivial RMS-threshold VAD. Baseline and test double (no model needed)."""

    name = "energy"

    def __init__(self, threshold: float = 0.02, frame_samples: int = 320):
        self.threshold = threshold
        self.frame_samples = frame_samples

    def reset(self):
        pass

    def is_speech(self, frame: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(frame ** 2))) >= self.threshold
