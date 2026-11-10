"""Build the barge-in test set from macOS `say` voices (SYNTHETIC speech, not human recordings).

Positive scenario: assistant TTS plays from t=0; the mic hears its echo (40 ms delay,
-12 or -20 dB) + white noise (-50 dBFS) + a user phrase starting at a known offset.
Ground-truth onset = offset + first energetic 10 ms frame of the clean user clip.
Negative scenario: echo + noise only (any barge-in is a false positive).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voiceloop import SR  # noqa: E402
from voiceloop.audio import db_to_gain, first_onset, mix_at, say_to_array, write_wav  # noqa: E402

ASSISTANT = [
    "Sure. The quarterly report shows revenue up eight percent, driven mainly by the new subscription tier, "
    "while costs stayed flat across operations, marketing and support.",
    "To reset your router, unplug it for thirty seconds, plug it back in, wait for the status light to turn "
    "solid green, and then reconnect your laptop to the network.",
    "Here is the plan for tomorrow. First we review the design, then we write the tests, and after lunch we "
    "ship the change to staging and watch the error rate for an hour.",
]
USER = ["Wait, stop.", "Hold on a second.", "No, that's wrong.", "Actually, can you repeat that?",
        "Sorry, I have a question.", "Okay, stop talking."]
VOICES = ["Daniel", "Fred", "Karen", "Moira", "Rishi", "Tessa"]
ECHO_DB = [-12, -20]
ECHO_DELAY = int(0.040 * SR)
NOISE_DB = -50


def main(out_dir: Path = ROOT / "data" / "audio"):
    rng = random.Random(7)
    nrng = np.random.default_rng(7)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = [say_to_array(t, "Samantha")[0] for t in ASSISTANT]
    users = {}
    for i, phrase in enumerate(USER):
        for j, v in enumerate(VOICES):
            if (i + j) % 2 == 0:  # 18 phrase/voice combos
                users[(phrase, v)] = say_to_array(phrase, v)[0]
    manifest = []
    k = 0
    for (phrase, voice), u in users.items():
        for edb in ECHO_DB:
            a = rng.randrange(len(tts))
            t = tts[a]
            offset = int(rng.uniform(0.8, 3.0) * SR)
            n = max(len(t) + ECHO_DELAY, offset + len(u)) + SR // 2
            mic = nrng.normal(0, db_to_gain(NOISE_DB), n).astype(np.float32)
            mic = mix_at(mic, t, ECHO_DELAY, db_to_gain(edb))
            mic = mix_at(mic, u, offset, 1.0)
            onset = offset + first_onset(u)
            name = f"pos_{k:03d}.wav"
            write_wav(out_dir / name, np.clip(mic, -1, 1))
            write_wav(out_dir / f"ref_{k:03d}.wav", t)
            manifest.append({"file": name, "ref": f"ref_{k:03d}.wav", "kind": "positive", "assistant": a,
                             "user_text": phrase, "voice": voice, "echo_db": edb, "echo_delay": ECHO_DELAY,
                             "onset_sample": onset, "user_end_sample": offset + len(u)})
            k += 1
    for a, t in enumerate(tts):
        for edb in ECHO_DB:
            n = len(t) + ECHO_DELAY + SR // 2
            mic = nrng.normal(0, db_to_gain(NOISE_DB), n).astype(np.float32)
            mic = mix_at(mic, t, ECHO_DELAY, db_to_gain(edb))
            name = f"neg_{a}_{abs(edb)}.wav"
            write_wav(out_dir / name, np.clip(mic, -1, 1))
            write_wav(out_dir / f"ref_neg_{a}_{abs(edb)}.wav", t)
            manifest.append({"file": name, "ref": f"ref_neg_{a}_{abs(edb)}.wav", "kind": "negative",
                             "assistant": a, "echo_db": edb, "echo_delay": ECHO_DELAY})
    (ROOT / "data" / "manifest.json").write_text(json.dumps({
        "label": "SYNTHETIC: macOS `say` voices mixed with simulated echo and noise",
        "sr": SR, "noise_dbfs": NOISE_DB, "scenarios": manifest}, indent=1))
    print(f"wrote {len(manifest)} scenarios to {out_dir}")


if __name__ == "__main__":
    main()
