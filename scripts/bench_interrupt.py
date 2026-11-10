"""Measure barge-in (interruption) latency on the synthetic test set.

latency_ms = (end of the frame where barge-in fired - ground-truth user onset) / SR
           + compute time of that frame. Playback-device buffer latency is NOT included.
A fire before the onset on a positive clip, or any fire on a negative clip, is a false barge-in.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voiceloop import SR  # noqa: E402
from voiceloop.audio import db_to_gain, read_wav  # noqa: E402
from voiceloop.bargein import BargeInDetector  # noqa: E402
from voiceloop.vad import EnergyVAD, SileroVAD, WebRTCVAD  # noqa: E402


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else None


def evaluate(make_vad, echo, scen, audio_dir, min_speech_ms=60, gain_err=0.8):
    vad = make_vad()
    lat, fp_pos, miss, fp_neg, n_neg, frame_ms = [], 0, 0, 0, 0, []
    for s in scen:
        mic = read_wav(audio_dir / s["file"])
        ref = read_wav(audio_dir / s["ref"])
        ref_full = np.zeros(len(mic), dtype=np.float32)
        ref_full[: len(ref)] = ref[: len(mic)]
        det = BargeInDetector(vad, min_speech_ms=min_speech_ms, echo=echo,
                              echo_gain_est=gain_err * db_to_gain(s["echo_db"]), echo_delay_est=s["echo_delay"])
        d = det.run(mic, ref_full)
        frame_ms += d.frame_compute_ms
        if s["kind"] == "negative":
            n_neg += 1
            fp_neg += d.fired
            continue
        if not d.fired or d.detect_sample > s["user_end_sample"]:
            miss += 1
        elif d.detect_sample < s["onset_sample"]:
            fp_pos += 1
        else:
            lat.append(1000 * (d.detect_sample - s["onset_sample"]) / SR + d.compute_ms_at_fire)
    n_pos = sum(s["kind"] == "positive" for s in scen)
    fms = 1000 * vad.frame_samples / SR
    return {
        "vad": vad.name, "echo": echo, "frame_ms": fms, "min_speech_ms": min_speech_ms,
        "n_positive": n_pos, "detected": len(lat), "missed": miss, "false_barge_in_before_onset": fp_pos,
        "n_negative": n_neg, "false_barge_in_echo_only": fp_neg,
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "min": min(lat) if lat else None,
                       "max": max(lat) if lat else None},
        "frame_compute_ms": {"p50": pct(frame_ms, 50), "p95": pct(frame_ms, 95)},
        "rtf": float(np.mean(frame_ms) / fms) if frame_ms else None,
    }


def main():
    audio_dir = ROOT / "data" / "audio"
    man = json.loads((ROOT / "data" / "manifest.json").read_text())
    scen = man["scenarios"]
    configs = [("silero", lambda: SileroVAD(0.5)), ("webrtc", lambda: WebRTCVAD(3)), ("energy", lambda: EnergyVAD(0.02))]
    rows = []
    for name, mk in configs:
        for echo in ("none", "ref_sub", "ref_sub_dtd"):
            t0 = time.time()
            r = evaluate(mk, echo, scen, audio_dir)
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print(json.dumps({k: r[k] for k in ("vad", "echo", "detected", "missed", "false_barge_in_before_onset",
                                                 "false_barge_in_echo_only", "latency_ms")}))
    out = {"label": man["label"], "date": time.strftime("%Y-%m-%d"),
           "machine": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0]}",
           "notes": "ref_sub uses the true echo delay and 0.8x the true echo gain (imperfect reference "
                    "subtraction, not real AEC); ref_sub_dtd adds a double-talk energy gate (dtd_ratio 0.5). Latency excludes output-device buffering.",
           "results": rows}
    (ROOT / "results" / "interrupt_latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
