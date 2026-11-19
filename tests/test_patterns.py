import numpy as np
import pytest

from tml.patterns import make_cue, make_patterns, make_signatures


def test_patterns_have_exact_activity_and_are_reproducible() -> None:
    first = make_patterns(30, 6, 8, np.random.default_rng(42))
    second = make_patterns(30, 6, 8, np.random.default_rng(42))

    assert first.shape == (8, 30)
    np.testing.assert_array_equal(first.sum(axis=1), np.full(8, 6))
    np.testing.assert_array_equal(first, second)


def test_signatures_are_bipolar_and_reproducible() -> None:
    first = make_signatures(7, 12, np.random.default_rng(5))
    second = make_signatures(7, 12, np.random.default_rng(5))

    assert first.shape == (7, 12)
    assert set(np.unique(first)) == {-1, 1}
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("args", "message"),
    [((0, 1, 1), "N and M"), ((3, 0, 1), "K must"), ((3, 4, 1), "K must")],
)
def test_invalid_pattern_dimensions_raise(
    args: tuple[int, int, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_patterns(*args, np.random.default_rng(0))


def test_invalid_signature_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="M and D"):
        make_signatures(2, 0, np.random.default_rng(0))


def test_make_cue_replaces_expected_target_cells_with_distractors() -> None:
    pattern = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)

    cue = make_cue(pattern, 0.5, 0.5, np.random.default_rng(10))

    assert cue.sum() == 2
    assert np.dot(cue, pattern) == 1
    np.testing.assert_array_equal(
        cue,
        make_cue(pattern, 0.5, 0.5, np.random.default_rng(10)),
    )


def test_make_cue_rejects_invalid_corruption() -> None:
    pattern = np.array([1, 1, 0, 0], dtype=np.int8)

    with pytest.raises(ValueError, match="corruption"):
        make_cue(pattern, 0.5, 1.1, np.random.default_rng(0))
