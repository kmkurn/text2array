from collections.abc import Iterable
from itertools import takewhile

import pytest

from text2array import StreamDataset, StreamBatches


def test_init(counter):
    dat = StreamDataset(counter)
    assert isinstance(dat, Iterable)
    it = takewhile(lambda x: x < 5, dat)
    assert list(it) == list(range(5))


def test_init_returns_iterable(finite_counter):
    dat = StreamDataset(finite_counter)
    dat_lst1 = list(dat)
    dat_lst2 = list(dat)
    assert len(dat_lst1) == len(dat_lst2)
    assert len(dat_lst2) > 0


def test_init_stream_non_iterable():
    with pytest.raises(TypeError) as exc:
        StreamDataset(5)
    assert '"stream" is not iterable' in str(exc.value)


def test_batch(finite_stream_dataset):
    bs = finite_stream_dataset.batch(2)
    assert isinstance(bs, StreamBatches)
    assert bs.batch_size == 2
    assert not bs.drop_last


def test_batch_exactly(finite_stream_dataset):
    bs = finite_stream_dataset.batch_exactly(2)
    assert isinstance(bs, StreamBatches)
    assert bs.batch_size == 2
    assert bs.drop_last


def test_batch_nonpositive_batch_size(stream_dataset):
    with pytest.raises(ValueError) as exc:
        stream_dataset.batch(0)
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        stream_dataset.batch_exactly(0)
    assert 'batch size must be greater than 0' in str(exc.value)
