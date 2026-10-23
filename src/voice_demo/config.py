from dataclasses import dataclass
from pathlib import Path
import math
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    device: str = "auto"
    asr_model: str = "large-v3"
    language: str | None = "zh"
    accept: float = 0.70
    reject: float = 0.45
    min_speech: float = 3.0
    min_enroll: float = 10.0
    max_audio: float = 1800.0

    def __post_init__(self):
        if not (-1 <= self.reject < self.accept <= 1):
            raise ValueError("阈值必须满足 -1 <= reject < accept <= 1")
        for field in ("min_speech", "min_enroll", "max_audio"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} 必须是有限正数")
        if self.min_enroll < self.min_speech:
            raise ValueError("录入语音时长不能小于验证语音时长")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("DEVICE 仅支持 auto、cpu、cuda")

    @classmethod
    def from_env(cls):
        load_dotenv()
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
        root = Path(os.getenv("DATA_DIR", "data")).resolve()
        os.environ.setdefault("HF_HOME", str(root / "cache" / "huggingface"))
        os.environ.setdefault("TORCH_HOME", str(root / "cache" / "torch"))
        os.environ.setdefault("MPLCONFIGDIR", str(root / "cache" / "matplotlib"))
        os.environ.setdefault("GRADIO_TEMP_DIR", str(root / "tmp"))
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            device=os.getenv("DEVICE", "auto"),
            asr_model=os.getenv("ASR_MODEL", "large-v3"),
            language=os.getenv("ASR_LANGUAGE", "zh") or None,
            accept=float(os.getenv("ACCEPT_THRESHOLD", "0.70")),
            reject=float(os.getenv("REJECT_THRESHOLD", "0.45")),
            min_speech=float(os.getenv("MIN_SPEECH_SECONDS", "3")),
            min_enroll=float(os.getenv("MIN_ENROLL_SECONDS", "10")),
            max_audio=float(os.getenv("MAX_AUDIO_SECONDS", "1800")),
        )
