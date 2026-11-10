"""Full loop on a scripted SYNTHETIC conversation: ask -> endpoint -> STT -> reply -> TTS -> barge-in.

The 'reply' step is a canned stub (no LLM) so the run measures the audio pipeline only.
Simulation: mic = user speech + echo of what the loop is playing (-12 dB, same frame,
no delay) + noise. The sample clock does not advance during STT/TTS compute; those
compute times are logged per turn.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voiceloop import SR  # noqa: E402
from voiceloop.audio import db_to_gain, first_onset, say_to_array  # noqa: E402
from voiceloop.loop import VoiceLoop  # noqa: E402
from voiceloop.stt import WhisperSTT  # noqa: E402
from voiceloop.vad import SileroVAD  # noqa: E402

REPLY = ("Here is the plan for tomorrow. First we review the design, then we write the tests, and after lunch "
         "we ship the change to staging and watch the error rate for an hour.")


def main():
    vad = SileroVAD(0.5)
    stt = WhisperSTT("small")
    stt.transcribe(np.zeros(SR, dtype=np.float32))
    g = db_to_gain(-12)
    loop = VoiceLoop(vad, stt.transcribe, reply=lambda t: REPLY if "plan" in t.lower() else "Okay, stopping.",
                     tts=lambda text: say_to_array(text, "Samantha"), echo_gain_est=g)
    q, _ = say_to_array("What is the plan for tomorrow?", "Daniel")
    stop, _ = say_to_array("Wait, stop.", "Fred")
    rng = np.random.default_rng(0)
    F = vad.frame_samples
    user = np.zeros(int(30 * SR), dtype=np.float32)
    user[int(0.5 * SR): int(0.5 * SR) + len(q)] += q
    barge_after_tts_s = 2.0
    t, tts_start, injected, onset = 0, None, False, None
    while t + F <= len(user):
        spk = loop.speaker_frame()
        if loop.state == "SPEAKING" and tts_start is None:
            tts_start = t
        if tts_start is not None and not injected and t >= tts_start + int(barge_after_tts_s * SR):
            user[t: t + len(stop)] += stop
            onset, injected = t + first_onset(stop), True
        mic = user[t:t + F] + g * spk + rng.normal(0, db_to_gain(-50), F).astype(np.float32)
        loop.step(mic, t, speaker=spk)
        t += F
        if injected and len([e for e in loop.events if e["event"] == "turn"]) >= 2:
            break
    barge = next(e for e in loop.events if e["event"] == "barge_in")
    out = {"label": "SYNTHETIC conversation (macOS say voices); reply is a canned stub",
           "date": time.strftime("%Y-%m-%d"), "stt": stt.name, "vad": "silero",
           "events": loop.events, "barge_onset_ms": round(1000 * onset / SR, 1),
           "interrupt_latency_ms_sim_clock": round(barge["t_ms"] - 1000 * onset / SR, 1)}
    (ROOT / "results" / "demo_loop.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
