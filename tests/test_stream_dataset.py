from collections.abc import Iterable
from itertools import takewhile

import pytest

from text2tensor import StreamDataset


def test_init(counter):
    dat = StreamDataset(counter)
    assert isinstance(dat, Iterable)


def test_init_stream_non_iterable():
    with pytest.raises(TypeError) as exc:
        StreamDataset(5)
    assert '"stream" is not iterable' in str(exc.value)


def test_iter(stream_dataset):
    it = takewhile(lambda x: x < 5, stream_dataset)
    assert list(it) == list(range(5))


def test_batch(finite_stream_dataset):
    bsize = 2
    minibatches = finite_stream_dataset.batch(bsize)
    assert isinstance(minibatches, Iterable)

    it = iter(minibatches)
    assert next(it) == [0, 1]
    assert next(it) == [2, 3]
    assert next(it) == [4, 5]
    while True:
        try:
            assert len(next(it)) <= bsize
        except StopIteration:
            break


def test_batch_exactly(finite_stream_dataset):
    bsize = 2
    minibatches = finite_stream_dataset.batch_exactly(bsize)
    assert isinstance(minibatches, Iterable)

    it = iter(minibatches)
    assert next(it) == [0, 1]
    assert next(it) == [2, 3]
    assert next(it) == [4, 5]
    while True:
        try:
            assert len(next(it)) == bsize
        except StopIteration:
            break


def test_batch_nonpositive_batch_size(stream_dataset):
    with pytest.raises(ValueError) as exc:
        stream_dataset.batch(0)
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        stream_dataset.batch_exactly(0)
    assert 'batch size must be greater than 0' in str(exc.value)
