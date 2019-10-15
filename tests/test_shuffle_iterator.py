from typing import Iterable, Sized
from unittest.mock import Mock

import pytest

from text2array import ShuffleIterator


def test_init(rng, samples):
    iter_ = ShuffleIterator(samples, rng=rng)
    assert isinstance(iter_, Sized)
    assert len(iter_) == len(samples)
    assert isinstance(iter_, Iterable)
    assert_shuffled(samples, list(iter_))


def test_init_kwargs(rng):
    ss = [{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}]
    iter_ = ShuffleIterator(ss, key=lambda s: s['i'], scale=2, rng=rng)
    assert_shuffled(ss, list(iter_))


def test_rng_called_correctly(rng, samples):
    mock_rng = Mock(wraps=rng)

    _ = list(ShuffleIterator(samples, rng=mock_rng))
    assert mock_rng.shuffle.call_count == 1

    ss = [{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}]
    _ = list(ShuffleIterator(ss, key=lambda s: s['i'], rng=mock_rng))
    assert mock_rng.uniform.call_count == len(ss)


def test_init_zero_scale(rng):
    ss = [{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}]
    key = lambda s: s['i']
    iter_ = ShuffleIterator(ss, key=key, scale=0, rng=rng)
    assert sorted(ss, key=key) == list(iter_)


def test_init_negative_scale(rng, samples):
    with pytest.raises(ValueError) as exc:
        ShuffleIterator(samples, scale=-1)
    assert 'scale cannot be less than 0' in str(exc.value)


def assert_shuffled(before, after):
    assert before != after and len(before) == len(after) and all(x in after for x in before)
