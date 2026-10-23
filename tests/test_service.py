import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from voice_demo.audio import RATE, load_audio
from voice_demo.config import Settings
from voice_demo.core import Turn
from voice_demo.evaluation import evaluate
from voice_demo.service import VoiceService


class FakeBackend:
    """Deterministic test double; these tests do not measure model accuracy."""
    fingerprint = "test-model@revision-1"

    def embed(self, wave):
        return np.array([1.0, 0.0]) if np.mean(wave) > 0 else np.array([0.0, 1.0])

    def speech_regions(self, wave):
        return [(0, len(wave) / RATE)] if np.max(np.abs(wave)) > .01 else []

    def diarize(self, wave, num_speakers=None):
        return ([Turn(0, 8, "A"), Turn(6, 12, "B")], [Turn(0, 8, "A"), Turn(8, 12, "B")])

    def transcribe(self, wave):
        return [{"start": 1, "end": 2, "text": "本人", "timing": "word"},
                {"start": 6.5, "end": 7.5, "text": "重叠", "timing": "word"},
                {"start": 9, "end": 10, "text": "别人", "timing": "word"}]


@pytest.fixture
def setup(tmp_path):
    first = tmp_path / "self.wav"
    other = tmp_path / "other.wav"
    silence = tmp_path / "silence.wav"
    sf.write(first, np.full(12 * RATE, .2), RATE)
    sf.write(other, np.full(12 * RATE, -.2), RATE)
    sf.write(silence, np.zeros(12 * RATE), RATE)
    service = VoiceService(Settings(data_dir=tmp_path / "data"), FakeBackend())
    profile = service.enroll("测试用户", [first])
    return service, profile, first, other, silence


def test_enroll_verify_and_silence(setup):
    service, profile, first, other, silence = setup
    assert service.verify(profile["profile_id"], first)["identity"] == "self"
    assert service.verify(profile["profile_id"], other)["identity"] == "other"
    assert service.verify(profile["profile_id"], silence)["identity"] == "unknown"
    with pytest.raises(ValueError, match="可用"):
        service.enroll("静音", [silence])


def test_model_revision_mismatch_rejects_profile(setup):
    service, profile, first, _, _ = setup
    service.backend.fingerprint = "test-model@revision-2"
    with pytest.raises(ValueError, match="版本"):
        service.verify(profile["profile_id"], first)


def test_end_to_end_exports_and_overlap_unknown(setup, tmp_path):
    service, profile, _, _, _ = setup
    meeting = tmp_path / "meeting.wav"
    sf.write(meeting, np.r_[np.full(8 * RATE, .2), np.full(4 * RATE, -.2)], RATE)
    result, report, archive = service.analyze(profile["profile_id"], meeting)
    assert result["speakers"]["A"]["identity"] == "self"
    assert result["speakers"]["B"]["identity"] == "other"
    assert result["speakers"]["A"]["clean_speech_seconds"] == 6
    assert result["speakers"]["B"]["clean_speech_seconds"] == 4
    assert [w["identity"] for w in result["words"]] == ["self", "unknown", "other"]
    assert all(s["identity"] == "unknown" for s in result["segments"])
    assert json.loads(report.read_text())["asr_enabled"] is True
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"result.json", "clips/0000.wav", "clips/0001.wav"}
    clip, rate = sf.read(report.parent / "clips/0000.wav")
    assert rate == RATE and len(clip) == 8 * RATE


def test_no_asr_does_not_call_transcriber(setup):
    service, profile, first, _, _ = setup
    service.backend.transcribe = lambda _: pytest.fail("ASR must not run")
    result, _, _ = service.analyze(profile["profile_id"], first, with_asr=False)
    assert result["words"] == []


def test_no_speech_produces_empty_report(setup):
    service, profile, _, _, silence = setup
    service.backend.diarize = lambda *_: pytest.fail("no speech must not diarize")
    result, _, _ = service.analyze(profile["profile_id"], silence)
    assert result["segments"] == [] and result["words"] == []


def test_stereo_resampling_and_duration_limit(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.full((48000, 2), .2), 48000)
    assert len(load_audio(path)) == RATE
    with pytest.raises(ValueError, match="超过"):
        load_audio(path, max_seconds=.5)


def test_store_path_traversal_rejected(setup):
    service, *_ = setup
    with pytest.raises(ValueError):
        service.profiles.get("../../.env")


def test_batch_evaluation_keeps_unknown_in_denominator(setup, tmp_path):
    service, profile, first, other, silence = setup
    manifest = tmp_path / "trials.jsonl"
    trials = [{"audio": p.name, "expected": label, "profile_id": profile["profile_id"]}
              for p, label in ((first, "self"), (other, "other"), (silence, "self"))]
    manifest.write_text("\n".join(json.dumps(t) for t in trials))
    report = evaluate(service, manifest)
    assert report["summary"]["self_miss_rate_including_unknown"] == .5
    assert report["summary"]["decision_coverage"] == pytest.approx(2 / 3)


def test_delete_only_selected_profile(setup):
    service, profile, first, *_ = setup
    second = service.enroll("另一模板", [first])
    service.profiles.delete(profile["profile_id"])
    assert service.profiles.get(second["profile_id"])["name"] == "另一模板"
    with pytest.raises(ValueError, match="不存在"):
        service.profiles.get(profile["profile_id"])
