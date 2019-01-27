from collections.abc import Iterable, Iterator

import pytest

from text2array import Batch, StreamDataset


def test_init(stream):
    dat = StreamDataset(stream)
    assert isinstance(dat, Iterable)
    assert list(dat) == list(stream)


def test_init_stream_non_iterable():
    with pytest.raises(TypeError) as exc:
        StreamDataset(5)
    assert '"stream" is not iterable' in str(exc.value)


def test_can_be_iterated_twice(stream_dataset):
    dat_lst1 = list(stream_dataset)
    dat_lst2 = list(stream_dataset)
    assert len(dat_lst1) == len(dat_lst2)
    assert len(dat_lst2) > 0


def test_batch(stream_dataset):
    bs = stream_dataset.batch(2)
    assert isinstance(bs, Iterator)
    bs_lst = list(bs)
    assert len(bs_lst) == 3
    assert all(isinstance(b, Batch) for b in bs_lst)
    dat = list(stream_dataset)
    assert list(bs_lst[0]) == [dat[0], dat[1]]
    assert list(bs_lst[1]) == [dat[2], dat[3]]
    assert list(bs_lst[2]) == [dat[4]]


def test_batch_size_evenly_divides(stream_dataset):
    bs = stream_dataset.batch(1)
    dat = list(stream_dataset)
    bs_lst = list(bs)
    assert len(bs_lst) == len(dat)
    for i in range(len(bs_lst)):
        assert list(bs_lst[i]) == [dat[i]]


def test_batch_exactly(stream_dataset):
    bs = stream_dataset.batch_exactly(2)
    assert isinstance(bs, Iterator)
    bs_lst = list(bs)
    assert len(bs_lst) == 2
    assert all(isinstance(b, Batch) for b in bs_lst)
    dat = list(stream_dataset)
    assert list(bs_lst[0]) == [dat[0], dat[1]]
    assert list(bs_lst[1]) == [dat[2], dat[3]]


def test_batch_nonpositive_batch_size(stream_dataset):
    with pytest.raises(ValueError) as exc:
        next(stream_dataset.batch(0))
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        next(stream_dataset.batch_exactly(0))
    assert 'batch size must be greater than 0' in str(exc.value)
