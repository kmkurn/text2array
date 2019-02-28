from typing import Iterable, Sized

import pytest

from text2array import ShuffleIterator


def test_init(setup_rng, samples):
    iter_ = ShuffleIterator(samples)
    assert isinstance(iter_, Sized)
    assert len(iter_) == len(samples)
    assert isinstance(iter_, Iterable)
    assert_shuffled(samples, list(iter_))


def test_init_kwargs(setup_rng):
    ss = [{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}]
    iter_ = ShuffleIterator(ss, key=lambda s: s['i'], scale=2)
    assert_shuffled(ss, list(iter_))


def test_init_zero_scale(setup_rng):
    ss = [{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}]
    key = lambda s: s['i']
    iter_ = ShuffleIterator(ss, key=key, scale=0)
    assert sorted(ss, key=key) == list(iter_)


def test_init_negative_scale(setup_rng, samples):
    with pytest.raises(ValueError) as exc:
        ShuffleIterator(samples, scale=-1)
    assert 'scale cannot be less than 0' in str(exc.value)


def assert_shuffled(before, after):
    assert before != after and len(before) == len(after) and all(x in after for x in before)
