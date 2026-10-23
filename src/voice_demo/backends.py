"""Replace this adapter to add ECAPA later; never compare vectors across adapters."""
from importlib.metadata import version
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import RATE
from .core import Turn

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


class AudioBackend(Protocol):
    @property
    def fingerprint(self) -> str: ...
    def embed(self, wave: np.ndarray) -> np.ndarray: ...
    def speech_regions(self, wave: np.ndarray) -> list[tuple[float, float]]: ...
    def diarize(self, wave: np.ndarray, num_speakers: int | None = None): ...
    def transcribe(self, wave: np.ndarray) -> list[dict]: ...


class LocalBackend:
    def __init__(self, settings):
        self.settings = settings
        self._embedding = self._vad = self._pipeline = self._asr = None
        self._fingerprint = None
        self._diarization_revision = None

    @property
    def device(self):
        import torch
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("指定了 CUDA，但当前 PyTorch 没有可用的 CUDA 设备")
        return ("cuda" if torch.cuda.is_available() else "cpu") if self.settings.device == "auto" else self.settings.device

    def _load_embedding(self):
        if self._embedding is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from pyannote.audio import Model, Inference
        try:
            checkpoint = hf_hub_download(EMBEDDING_MODEL, "pytorch_model.bin", token=os.getenv("HF_TOKEN") or None)
            revision = Path(checkpoint).parent.name
            model = Model.from_pretrained(checkpoint)
            model.eval()
            self._embedding = Inference(model, window="whole", device=torch.device(self.device))
            self._fingerprint = f"{EMBEDDING_MODEL}@{revision};pyannote.audio={version('pyannote.audio')};preprocess=mono16k-v1"
        except Exception as exc:
            raise RuntimeError(f"WeSpeaker 加载失败（{type(exc).__name__}）。请检查网络、Hugging Face 访问和依赖；可运行 voice-demo doctor。") from exc

    @property
    def fingerprint(self):
        self._load_embedding()
        return self._fingerprint

    def embed(self, wave):
        import torch
        self._load_embedding()
        with torch.inference_mode():
            return self._embedding({"waveform": torch.from_numpy(wave.copy()).unsqueeze(0), "sample_rate": RATE}).reshape(-1)

    def speech_regions(self, wave):
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps
        if self._vad is None:
            self._vad = load_silero_vad(onnx=True)
        regions = get_speech_timestamps(
            torch.from_numpy(wave.copy()), self._vad, sampling_rate=RATE,
            min_speech_duration_ms=250, min_silence_duration_ms=150, speech_pad_ms=30,
        )
        return [(r["start"] / RATE, r["end"] / RATE) for r in regions]

    def diarize(self, wave, num_speakers=None):
        import torch
        from huggingface_hub import hf_hub_download
        from pyannote.audio import Pipeline
        if self._pipeline is None:
            try:
                config = hf_hub_download(DIARIZATION_MODEL, "config.yaml", token=os.getenv("HF_TOKEN") or None)
                self._diarization_revision = Path(config).parent.name
                self._pipeline = Pipeline.from_pretrained(
                    DIARIZATION_MODEL, revision=self._diarization_revision, token=os.getenv("HF_TOKEN") or None,
                )
                self._pipeline.to(torch.device(self.device))
            except Exception as exc:
                raise RuntimeError(
                    f"Community-1 加载失败（{type(exc).__name__}）。请先在模型页面接受条件，并在本地 .env 填写有读取权限的 HF_TOKEN。"
                ) from exc
        kwargs = {} if num_speakers is None else {"num_speakers": num_speakers}
        output = self._pipeline({"waveform": torch.from_numpy(wave.copy()).unsqueeze(0), "sample_rate": RATE}, **kwargs)
        def convert(annotation):
            return [Turn(float(segment.start), float(segment.end), str(speaker))
                    for segment, _, speaker in annotation.itertracks(yield_label=True)]
        return convert(output.speaker_diarization), convert(output.exclusive_speaker_diarization)

    def transcribe(self, wave):
        from faster_whisper import WhisperModel
        if self._asr is None:
            self._asr = WhisperModel(
                self.settings.asr_model, device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
                download_root=str(self.settings.data_dir / "cache" / "whisper"),
            )
        segments, _ = self._asr.transcribe(wave, language=self.settings.language, word_timestamps=True, vad_filter=True)
        words = []
        for segment in segments:
            if segment.words:
                words.extend({"start": float(w.start), "end": float(w.end), "text": w.word,
                              "asr_probability": float(w.probability), "timing": "word"} for w in segment.words)
            else:
                words.append({"start": float(segment.start), "end": float(segment.end),
                              "text": segment.text, "asr_probability": None, "timing": "segment"})
        return words

    def metadata(self):
        return {
            "embedding": self._fingerprint,
            "diarization": DIARIZATION_MODEL,
            "diarization_revision": self._diarization_revision,
            "asr": self.settings.asr_model,
            "language": self.settings.language,
            "device": self.device,
            "packages": {p: version(p) for p in ("pyannote.audio", "torch", "faster-whisper", "silero-vad")},
        }
