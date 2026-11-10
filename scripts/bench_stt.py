"""STT latency + WER for the barge-in utterances (faster-whisper small, int8, CPU).

Input per positive scenario: mic from onset-200 ms to user end+300 ms after reference
subtraction (0.8x echo gain, so residual echo remains), i.e. what the loop would send
to STT after an endpoint. latency = transcribe() wall time on this machine.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voiceloop import SR  # noqa: E402
from voiceloop.audio import db_to_gain, read_wav  # noqa: E402
from voiceloop.stt import WhisperSTT, wer  # noqa: E402


def main(size: str = "small"):
    man = json.loads((ROOT / "data" / "manifest.json").read_text())
    audio = ROOT / "data" / "audio"
    stt = WhisperSTT(size)
    stt.transcribe(np.zeros(SR, dtype=np.float32))  # warm-up (excluded)
    rows = []
    for s in man["scenarios"]:
        if s["kind"] != "positive":
            continue
        mic = read_wav(audio / s["file"])
        ref = read_wav(audio / s["ref"])
        r = np.zeros(len(mic), dtype=np.float32)
        d = s["echo_delay"]
        r[d:d + len(ref)] = ref[: len(mic) - d]
        clean = mic - 0.8 * db_to_gain(s["echo_db"]) * r
        a, b = max(0, s["onset_sample"] - SR // 5), min(len(mic), s["user_end_sample"] + int(0.3 * SR))
        seg = clean[a:b]
        text, ms = stt.transcribe(seg)
        rows.append({"file": s["file"], "ref": s["user_text"], "hyp": text, "wer": float(wer(s["user_text"], text)),
                     "latency_ms": ms, "audio_s": len(seg) / SR, "echo_db": s["echo_db"]})
        print(f"{ms:7.1f} ms  WER={rows[-1]['wer']:.2f}  {s['user_text']!r} -> {text!r}")
    lat = [r["latency_ms"] for r in rows]
    out = {
        "label": man["label"], "model": stt.name, "date": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0]}",
        "n": len(rows), "mean_wer": statistics.fmean(r["wer"] for r in rows),
        "exact_match_rate": float(sum(r["wer"] == 0 for r in rows) / len(rows)),
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
        "rtf_mean": statistics.fmean(r["latency_ms"] / 1000 / r["audio_s"] for r in rows),
        "rows": rows,
    }
    (ROOT / "results" / f"stt_{size}.json").write_text(json.dumps(out, indent=1))
    print({k: out[k] for k in ("n", "mean_wer", "exact_match_rate", "latency_ms", "rtf_mean")})


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
