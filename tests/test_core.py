import numpy as np
import pytest

from voice_demo.config import Settings
from voice_demo.core import Turn, centroid, cosine, decide, timeline_regions, windows, word_speaker
from voice_demo.evaluation import summarize


def test_cosine_is_scale_invariant_and_centroid_normalized():
    assert cosine([1, 2], [10, 20]) == pytest.approx(1)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1)
    assert np.linalg.norm(centroid([[1, 0], [0, 2]], [3, 1])) == pytest.approx(1)


@pytest.mark.parametrize("bad", [[], [0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_embedding_never_matches(bad):
    with pytest.raises(ValueError):
        cosine(bad, [1, 0])


@pytest.mark.parametrize("score,seconds,expected", [(.7, 3, "self"), (.45, 3, "other"), (.6, 3, "unknown"), (.99, 2.99, "unknown"), (None, 0, "unknown")])
def test_threshold_boundaries_and_short_audio(score, seconds, expected):
    assert decide(score, seconds, .7, .45, 3)[0] == expected


def test_conflicting_windows_are_not_overruled_by_average():
    assert decide(.8, 10, .7, .45, 3, [.99, .2]) == ("unknown", "conflicting_windows")


def test_overlap_is_subtracted_from_both_speakers():
    clean, overlap = timeline_regions([Turn(0, 6, "A"), Turn(2, 4, "B"), Turn(8, 10, "B")])
    assert clean == {"A": [(0, 2), (4, 6)], "B": [(8, 10)]}
    assert overlap == [(2, 4)]
    assert word_speaker(2.5, 3, [Turn(0, 6, "A")], overlap) == (None, True)


def test_adjacent_same_speaker_merge_but_silence_does_not():
    clean, _ = timeline_regions([Turn(0, 2, "A"), Turn(2, 4, "A"), Turn(5, 6, "A")])
    assert clean["A"] == [(0, 4), (5, 6)]


def test_no_assignment_across_silence_or_tied_boundary():
    turns = [Turn(0, 1, "A"), Turn(1, 2, "B")]
    assert word_speaker(.5, 1.5, turns, []) == (None, False)
    assert word_speaker(5, 6, turns, []) == (None, False)


def test_windows_preserve_duration_without_padding_into_other_speaker():
    result = windows([(3, 13.3), (20, 20.8)])
    assert len(result) == 3
    assert sum(end - start for start, end in result) == pytest.approx(10.3)
    assert result[-1][1] == pytest.approx(13.3)


def test_unknowns_count_as_self_misses():
    rows = [{"expected": "self", "identity": "unknown"}, {"expected": "other", "identity": "unknown"}]
    result = summarize(rows)
    assert result["false_accept_rate"] == 0
    assert result["self_miss_rate_including_unknown"] == 1
    assert result["decision_coverage"] == 0


def test_invalid_threshold_config():
    with pytest.raises(ValueError):
        Settings(accept=.4, reject=.7)
