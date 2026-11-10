import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voiceloop import SR  # noqa: E402
from voiceloop.audio import db_to_gain, first_onset, mix_at, rms  # noqa: E402
from voiceloop.bargein import BargeInDetector, TTSPlayer  # noqa: E402
from voiceloop.loop import VoiceLoop  # noqa: E402
from voiceloop.stt import normalize, wer  # noqa: E402
from voiceloop.vad import EnergyVAD  # noqa: E402

rng = np.random.default_rng(0)


def tone(sec, amp=0.3, f=220):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_first_onset_and_mix():
    x = np.concatenate([np.zeros(SR // 2, np.float32), tone(0.5)])
    assert abs(first_onset(x) - SR // 2) <= 160
    y = mix_at(np.zeros(100, np.float32), np.ones(50, np.float32), 80)
    assert y[79] == 0 and y[80:].sum() == 20


def test_db_gain_and_rms():
    assert db_to_gain(-20) == pytest.approx(0.1)
    assert rms(np.ones(10)) == pytest.approx(1.0, abs=1e-6)


def test_wer():
    assert wer("Wait, stop.", "wait stop") == 0
    assert wer("hold on a second", "hold on") == 0.5
    assert wer("a b", "a x b") == 0.5
    assert normalize("That's IT!") == ["that's", "it"]


def test_tts_player_stops():
    p = TTSPlayer(np.ones(1000, np.float32))
    assert p.next_frame(400).sum() == 400
    p.stop(400)
    assert not p.playing and p.next_frame(400).sum() == 0 and p.stopped_at == 400


def test_bargein_latency_quantised_by_frames():
    vad = EnergyVAD(0.05, frame_samples=320)  # 20 ms frames
    mic = np.zeros(SR * 2, np.float32)
    onset = SR  # 1.0 s
    mic[onset:] = tone(1.0)
    d = BargeInDetector(vad, min_speech_ms=60).run(mic)
    assert d.fired
    lat_ms = 1000 * (d.detect_sample - onset) / SR
    assert 60 <= lat_ms <= 80  # 3 frames of 20 ms


def test_bargein_no_fire_on_silence():
    d = BargeInDetector(EnergyVAD(0.05), min_speech_ms=60).run(rng.normal(0, 0.003, SR).astype(np.float32))
    assert not d.fired and len(d.frame_compute_ms) > 0


def test_echo_without_cancellation_false_fires():
    ref = tone(1.0, 0.5)
    mic = 0.25 * ref  # echo only, loud
    assert BargeInDetector(EnergyVAD(0.05), echo="none").run(mic, ref).fired


def test_ref_sub_dtd_rejects_residual_echo_but_catches_user():
    ref = tone(2.0, 0.5, 300)
    g = 0.25
    echo_only = g * ref
    det = lambda: BargeInDetector(EnergyVAD(0.02), echo="ref_sub_dtd", echo_gain_est=0.8 * g)  # noqa: E731
    assert not det().run(echo_only, ref).fired
    user = np.zeros_like(ref)
    user[SR:] = tone(1.0, 0.3, 150)
    d = det().run(echo_only + user, ref)
    assert d.fired and d.detect_sample >= SR


def test_bad_echo_mode():
    with pytest.raises(ValueError):
        BargeInDetector(EnergyVAD(), echo="magic")


def run_loop(barge: bool):
    vad = EnergyVAD(0.05, frame_samples=320)
    calls = []
    loop = VoiceLoop(vad, stt=lambda a: (calls.append(len(a)) or "hello", 1.0), reply=lambda t: "reply",
                     tts=lambda t: (tone(1.0, 0.4, 500), 0.01), endpoint_ms=200, min_speech_ms=60)
    user = np.zeros(SR * 4, np.float32)
    user[SR // 4: SR // 4 + SR // 2] = tone(0.5)
    t, F, tts_start = 0, 320, None
    while t + F <= len(user):
        spk = loop.speaker_frame()
        if loop.state == "SPEAKING" and tts_start is None:
            tts_start = t
            if barge:
                user[t + SR // 4: t + SR // 4 + SR // 4] = tone(0.25)
        loop.step(user[t:t + F], t)  # no echo in this unit test
        t += F
    return loop, calls, tts_start


def test_loop_turn_then_tts_done():
    loop, calls, _ = run_loop(barge=False)
    kinds = [e["event"] for e in loop.events]
    assert kinds[:3] == ["endpoint", "turn", "tts_done"]
    assert calls and loop.events[1]["transcript"] == "hello"


def test_loop_barge_in_stops_tts():
    loop, _, tts_start = run_loop(barge=True)
    b = next(e for e in loop.events if e["event"] == "barge_in")
    kinds = [e["event"] for e in loop.events]
    # barge-in comes before any tts_done, and the interrupting speech becomes the next turn
    assert kinds.index("barge_in") < kinds.index("turn", 2)
    assert b["tts_played_ms"] < 1000  # TTS (1 s) was cut short
    lat = b["t_ms"] - 1000 * (tts_start + SR // 4) / SR
    assert 0 < lat <= 80


@pytest.mark.slow
def test_silero_detects_tone_free_speech_like_noise_gracefully():
    pytest.importorskip("silero_vad")
    from voiceloop.vad import SileroVAD

    v = SileroVAD()
    assert v.frame_samples == 512
    assert v.is_speech(np.zeros(512, np.float32)) is False
