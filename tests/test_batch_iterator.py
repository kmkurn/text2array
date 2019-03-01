from typing import Iterable, Sized

import pytest

from text2array import Batch, BatchIterator


def test_init(samples):
    iter_ = BatchIterator(samples)
    assert isinstance(iter_, Sized)
    assert len(iter_) == len(samples)
    assert isinstance(iter_, Iterable)
    assert all(isinstance(b, Batch) for b in iter_)
    assert all(len(b) == 1 for b in iter_)
    assert all(b[0] == s for b, s in zip(iter_, samples))


def test_init_stream(stream):
    iter_ = BatchIterator(stream)
    with pytest.raises(TypeError):
        len(iter_)


def test_init_kwargs():
    ss = [{'i': i} for i in range(5)]
    bsz = 2
    iter_ = BatchIterator(ss, batch_size=bsz)
    assert iter_.batch_size == bsz
    assert len(iter_) == len(ss) // bsz + (1 if len(ss) % bsz != 0 else 0)
    assert all(len(b) <= bsz for b in iter_)

    bs = list(iter_)
    assert list(bs[0]) == [ss[0], ss[1]]
    assert list(bs[1]) == [ss[2], ss[3]]
    assert list(bs[2]) == [ss[4]]


def test_init_nonpositive_batch_size(samples):
    with pytest.raises(ValueError) as exc:
        BatchIterator(samples, batch_size=0)
    assert 'batch size must be greater than 0' in str(exc.value)
