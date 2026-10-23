from pathlib import Path
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16000


def load_audio(path, max_seconds=1800):
    """Decode with bounded memory. WAV/FLAC via soundfile; other media via PyAV."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("找不到音频文件")
    if path.stat().st_size > 200 * 1024 * 1024:
        raise ValueError("Demo 单文件最大 200 MB")
    try:
        audio = sf.SoundFile(str(path))
    except (sf.LibsndfileError, RuntimeError):
        audio = None
    if audio is not None:
        with audio:
            if len(audio) / audio.samplerate > max_seconds:
                raise ValueError(f"录音超过 {max_seconds:g} 秒限制")
            rate = audio.samplerate
            wave = audio.read(dtype="float32", always_2d=True).mean(axis=1)
        if rate != RATE:
            divisor = math.gcd(rate, RATE)
            wave = resample_poly(wave, RATE // divisor, rate // divisor)
    else:
        import av
        frames = []
        count = 0
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("文件中没有音轨")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
            for frame in container.decode(audio=0):
                # Packed output remains aligned to the original audio's relative time axis.
                for output in resampler.resample(frame):
                    chunk = output.to_ndarray().reshape(-1)
                    count += chunk.size
                    if count > max_seconds * RATE:
                        raise ValueError(f"录音超过 {max_seconds:g} 秒限制")
                    frames.append(chunk)
            for output in resampler.resample(None):
                frames.append(output.to_ndarray().reshape(-1))
        wave = np.concatenate(frames) if frames else np.array([], dtype=np.float32)
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError("音频为空或包含无效采样值")
    if len(wave) > max_seconds * RATE + RATE // 100:
        raise ValueError(f"录音超过 {max_seconds:g} 秒限制")
    return np.ascontiguousarray(wave, dtype=np.float32)


def crop(wave, start, end):
    return wave[max(0, round(start * RATE)):min(len(wave), round(end * RATE))]


def quality(wave):
    return {
        "duration_seconds": round(len(wave) / RATE, 3),
        "clipped_fraction": float(np.mean(np.abs(wave) >= 0.999)),
        "rms": float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))),
    }
