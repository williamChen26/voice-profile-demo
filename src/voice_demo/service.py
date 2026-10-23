from dataclasses import asdict
from pathlib import Path
import time
import uuid
import zipfile

import soundfile as sf

from .audio import RATE, crop, load_audio, quality
from .backends import LocalBackend
from .core import centroid, cosine, decide, intersect, timeline_regions, windows, word_speaker
from .store import ProfileStore, timestamp, write_json


class VoiceService:
    def __init__(self, settings, backend=None):
        self.settings = settings
        self.backend = backend or LocalBackend(settings)
        self.profiles = ProfileStore(settings.data_dir)

    def _evidence(self, wave, regions):
        evidence = []
        for start, end in windows(regions):
            snippet = crop(wave, start, end)
            if quality(snippet)["clipped_fraction"] > 0.02:
                continue
            evidence.append({"start": start, "end": end, "duration": len(snippet) / RATE,
                             "vector": self.backend.embed(snippet)})
        return evidence

    def _load_profile(self, identifier):
        profile = self.profiles.get(identifier)
        if profile["model_fingerprint"] != self.backend.fingerprint:
            raise ValueError("声纹模型或预处理版本与录入时不同，请重新录入；不能跨模型比较向量")
        return profile

    def _match(self, evidence, profile):
        seconds = sum(e["duration"] for e in evidence)
        score = None
        scores = []
        if evidence:
            scores = [cosine(e["vector"], profile["centroid"]) for e in evidence]
            score = cosine(centroid([e["vector"] for e in evidence], [e["duration"] for e in evidence]), profile["centroid"])
        identity, reason = decide(score, seconds, self.settings.accept, self.settings.reject,
                                  self.settings.min_speech, scores)
        return {
            "identity": identity, "similarity": score, "reason": reason,
            "clean_speech_seconds": round(seconds, 3),
            "windows": [{"start": e["start"], "end": e["end"], "similarity": s} for e, s in zip(evidence, scores)],
        }

    def _metadata(self):
        return {
            "created_at": timestamp(),
            "thresholds": {"accept": self.settings.accept, "reject": self.settings.reject,
                           "min_speech_seconds": self.settings.min_speech, "calibrated": False},
            "backend": self.backend.metadata() if hasattr(self.backend, "metadata") else {"embedding": self.backend.fingerprint},
        }

    def enroll(self, name, paths):
        if not paths:
            raise ValueError("请提供至少一段本人单独说话的录音")
        if len(paths) > 10:
            raise ValueError("每次最多录入 10 个文件")
        evidence, samples = [], []
        for index, path in enumerate(paths):
            wave = load_audio(path, self.settings.max_audio)
            regions = self.backend.speech_regions(wave)
            current = self._evidence(wave, regions)
            for item in current:
                item["sample_index"] = index
            evidence.extend(current)
            samples.append({"sample_index": index, **quality(wave),
                            "usable_seconds": sum(e["duration"] for e in current)})
        seconds = sum(e["duration"] for e in evidence)
        if seconds < self.settings.min_enroll:
            raise ValueError(f"可用的连续语音仅 {seconds:.1f} 秒，录入至少需要 {self.settings.min_enroll:g} 秒。请录制更长的单人语音，避免过多停顿或削波。")
        center = centroid([e["vector"] for e in evidence], [e["duration"] for e in evidence])
        return self.profiles.create(name, {
            "schema_version": 1, "model_fingerprint": self.backend.fingerprint,
            "centroid": center.tolist(), "usable_seconds": seconds, "samples": samples,
            "vectors": [{**e, "vector": e["vector"].tolist()} for e in evidence],
            "template_similarity": [cosine(e["vector"], center) for e in evidence],
            "notes": ["录入音频必须由用户确认仅包含本人。VAD 不验证身份，也不保证排除多人。", "模板不自动更新；原始录入音频不复制到声纹库。"],
        })

    def verify(self, identifier, path):
        profile = self._load_profile(identifier)
        wave = load_audio(path, self.settings.max_audio)
        evidence = self._evidence(wave, self.backend.speech_regions(wave))
        return {"profile_id": identifier, "profile_name": profile["name"],
                **self._match(evidence, profile), "audio_quality": quality(wave), **self._metadata(),
                "notes": ["单段验证假定输入为单人语音；多人录音请使用 analyze。", "相似度不是概率；默认阈值尚未校准。"]}

    def analyze(self, identifier, path, with_asr=True, num_speakers=None):
        if num_speakers is not None and (isinstance(num_speakers, bool) or not isinstance(num_speakers, int) or num_speakers < 1):
            raise ValueError("实际说话人数必须为正整数，或留空自动估计")
        started = time.perf_counter()
        profile = self._load_profile(identifier)
        wave = load_audio(path, self.settings.max_audio)
        speech = self.backend.speech_regions(wave)
        if speech:
            turns, exclusive = self.backend.diarize(wave, num_speakers)
        else:
            turns, exclusive = [], []
        clean, overlap = timeline_regions(turns)
        matches = {}
        for speaker in sorted({t.speaker for t in turns}):
            evidence = self._evidence(wave, intersect(clean.get(speaker, []), speech))
            matches[speaker] = self._match(evidence, profile)
        words = self.backend.transcribe(wave) if with_asr and speech else []
        for word in words:
            speaker, is_overlap = word_speaker(word["start"], word["end"], exclusive, overlap)
            if word.get("timing") != "word":
                speaker = None
            word["speaker_id"] = speaker
            word["overlap"] = is_overlap
            word["identity"] = matches.get(speaker, {}).get("identity", "unknown")
        job_dir = self.settings.data_dir / "jobs" / uuid.uuid4().hex
        job_dir.mkdir(parents=True, mode=0o700)
        clips_dir = job_dir / "clips"
        clips_dir.mkdir()
        segments = []
        for index, turn in enumerate(turns):
            clip = f"{index:04d}.wav"
            sf.write(str(clips_dir / clip), crop(wave, turn.start, turn.end), RATE, subtype="PCM_16")
            has_overlap = bool(intersect([(turn.start, turn.end)], overlap))
            segments.append({**asdict(turn), "identity": "unknown" if has_overlap else matches[turn.speaker]["identity"],
                             "overlap": has_overlap, "clip": f"clips/{clip}",
                             "identity_source": "speaker_cluster"})
        elapsed = time.perf_counter() - started
        result = {
            "profile_id": identifier, "profile_name": profile["name"], "speakers": matches,
            "segments": segments, "exclusive_diarization": [asdict(t) for t in exclusive],
            "overlap_regions": overlap, "words": words, "asr_enabled": with_asr,
            "audio_quality": quality(wave), "processing_seconds": elapsed,
            "real_time_factor": elapsed / (len(wave) / RATE), **self._metadata(),
            "notes": ["切出的音频仍保留原混音，重叠部分未做人声分离。", "身份按说话人簇聚合推断，分段错误可能传播。", "相似度不是概率；默认阈值尚未校准。"],
        }
        report = job_dir / "result.json"
        write_json(report, result)
        archive = job_dir / "result.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            output.write(report, "result.json")
            for clip in sorted(clips_dir.glob("*.wav")):
                output.write(clip, f"clips/{clip.name}")
        return result, report, archive
