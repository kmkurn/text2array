from collections.abc import Iterable

import pytest

from text2array import StreamDataset, StreamBatches


def test_init(counter):
    dat = StreamDataset(counter)
    assert isinstance(dat, Iterable)
    assert list(dat) == list(range(counter.limit))


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
    assert isinstance(bs, StreamBatches)
    assert bs.batch_size == 2
    assert not bs.drop_last


def test_batch_exactly(stream_dataset):
    bs = stream_dataset.batch_exactly(2)
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
