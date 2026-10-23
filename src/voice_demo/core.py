"""Model-independent scoring and timeline operations."""
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def normalize(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(value)
    if not value.size or not np.isfinite(value).all() or not math.isfinite(norm) or norm < 1e-10:
        raise ValueError("声纹向量为空、非有限值或零向量")
    return value / norm


def centroid(vectors, weights):
    rows = np.stack([normalize(v) for v in vectors])
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(rows) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("声纹权重必须为有限正数且与向量数量一致")
    return normalize(np.average(rows, axis=0, weights=weights))


def cosine(left, right):
    a, b = normalize(left), normalize(right)
    if a.shape != b.shape:
        raise ValueError("声纹维度不同，请使用相同模型重新录入")
    return float(np.clip(a @ b, -1, 1))


def decide(score, seconds, accept, reject, minimum, window_scores=()):
    if not (-1 <= reject < accept <= 1):
        raise ValueError("无效的判定阈值")
    if seconds < minimum or score is None or not math.isfinite(score):
        return "unknown", "insufficient_clean_speech"
    if window_scores and min(window_scores) <= reject and max(window_scores) >= accept:
        return "unknown", "conflicting_windows"
    if score >= accept:
        return "self", "above_accept_threshold"
    if score <= reject:
        return "other", "below_reject_threshold"
    return "unknown", "between_thresholds"


def intersect(regions, mask):
    return [(max(a, c), min(b, d)) for a, b in regions for c, d in mask
            if min(b, d) > max(a, c)]


def windows(regions, maximum=5.0, minimum=1.5):
    """Non-overlapping windows; balance a region to avoid tiny tail fragments."""
    result = []
    for start, end in regions:
        if end - start < minimum:
            continue
        count = math.ceil((end - start) / maximum)
        step = (end - start) / count
        result.extend((start + i * step, start + (i + 1) * step) for i in range(count))
    return result


def timeline_regions(turns):
    """Sweep original diarization, preserving overlap instead of trusting exclusive labels."""
    events = defaultdict(list)
    for t in turns:
        if t.end > t.start:
            events[t.start].append((t.speaker, 1))
            events[t.end].append((t.speaker, -1))
    active = defaultdict(int)
    clean = defaultdict(list)
    overlaps = []
    times = sorted(events)
    for index, start in enumerate(times[:-1]):
        for speaker, delta in events[start]:
            active[speaker] += delta
        end = times[index + 1]
        speakers = [s for s, count in active.items() if count > 0]
        destination = clean[speakers[0]] if len(speakers) == 1 else overlaps if len(speakers) > 1 else None
        if destination is not None:
            if destination and abs(destination[-1][1] - start) < 1e-8:
                destination[-1] = (destination[-1][0], end)
            else:
                destination.append((start, end))
    return dict(clean), overlaps


def word_speaker(start, end, turns, overlap_regions):
    if end <= start:
        return None, False
    overlap = any(min(end, b) > max(start, a) for a, b in overlap_regions)
    if overlap:
        return None, True
    durations = defaultdict(float)
    for t in turns:
        durations[t.speaker] += max(0.0, min(end, t.end) - max(start, t.start))
    candidates = sorted(durations.items(), key=lambda pair: pair[1], reverse=True)
    if not candidates or candidates[0][1] < (end - start) * 0.5:
        return None, False
    if len(candidates) > 1 and abs(candidates[0][1] - candidates[1][1]) < 1e-8:
        return None, False
    return candidates[0][0], False
