from collections.abc import Iterable

import numpy as np
import pytest

from text2array import StreamBatches


class TestInit:
    def test_ok(self, stream_dataset):
        bs = StreamBatches(stream_dataset, 2)
        assert bs.batch_size == 2
        assert not bs.drop_last
        assert isinstance(bs, Iterable)
        dat = list(stream_dataset)
        assert list(bs) == [[dat[0], dat[1]], [dat[2], dat[3]], [dat[4]]]

    def test_kwargs(self, stream_dataset):
        bs = StreamBatches(stream_dataset, 2, drop_last=True)
        assert bs.drop_last
        dat = list(stream_dataset)
        assert list(bs) == [[dat[0], dat[1]], [dat[2], dat[3]]]

    def test_divisible_length(self, stream_dataset):
        bs = StreamBatches(stream_dataset, 1)
        dat = list(stream_dataset)
        assert list(bs) == [[dat[0]], [dat[1]], [dat[2]], [dat[3]], [dat[4]]]

    def test_nonpositive_batch_size(self, stream_dataset):
        with pytest.raises(ValueError) as exc:
            StreamBatches(stream_dataset, 0)
        assert 'batch size must be greater than 0' in str(exc.value)


@pytest.fixture
def stream_batches(stream_dataset):
    return StreamBatches(stream_dataset, 2)


def test_can_be_iterated_twice(stream_batches):
    bs_lst1 = list(stream_batches)
    bs_lst2 = list(stream_batches)
    assert len(bs_lst1) == len(bs_lst2)
    assert len(bs_lst2) > 0


@pytest.mark.skip
def test_to_arrays(stream_batches):
    ts = stream_batches.to_arrays()
    assert isinstance(ts, Iterable)

    bs = list(stream_batches)
    for t, b in zip(ts, bs):
        assert isinstance(t.x, np.ndarray)
        assert t.x.dtype == np.int32
        assert t.x.tolist() == list(b.x)

        assert isinstance(t.y, np.ndarray)
        assert t.y.dtype == np.int32
        assert t.y.tolist() == list(b.y)
