"""Audio helpers: macOS `say` TTS to 16 kHz mono, mixing, onset detection."""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import SR


def say_to_array(text: str, voice: str = "Samantha", rate: int = 185) -> tuple[np.ndarray, float]:
    """Synthesize with macOS `say`. Returns (float32 mono @16 kHz, synth wall seconds)."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.wav"
        t0 = time.perf_counter()
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(out), "--data-format=LEI16@16000", text],
                       check=True, capture_output=True)
        wall = time.perf_counter() - t0
        x, sr = sf.read(out, dtype="float32")
    assert sr == SR, sr
    return (x if x.ndim == 1 else x.mean(1)), wall


def write_wav(path: str | Path, x: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, SR, subtype="PCM_16")


def read_wav(path: str | Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr}")
    return x if x.ndim == 1 else x.mean(1)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def first_onset(x: np.ndarray, frame: int = 160, rel_db: float = -30.0) -> int:
    """Sample index of the first 10 ms frame whose RMS is within `rel_db` of the loudest frame."""
    n = len(x) // frame
    if n == 0:
        return 0
    e = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    thr = e.max() * 10 ** (rel_db / 20)
    idx = np.flatnonzero(e >= thr)
    return int(idx[0] * frame) if len(idx) else 0


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20)


def mix_at(base: np.ndarray, add: np.ndarray, start: int, gain: float = 1.0) -> np.ndarray:
    out = base.copy()
    end = min(len(out), start + len(add))
    if end > start:
        out[start:end] += gain * add[: end - start]
    return out
