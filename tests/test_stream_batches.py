from collections.abc import Iterable

import numpy as np
import pytest

from text2array import StreamBatches


class TestInit:
    def test_ok(self, finite_stream_dataset):
        bs = StreamBatches(finite_stream_dataset, 2)
        assert bs.batch_size == 2
        assert not bs.drop_last
        assert isinstance(bs, Iterable)
        assert list(bs) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]

    def test_kwargs(self, finite_stream_dataset):
        bs = StreamBatches(finite_stream_dataset, 2, drop_last=True)
        assert bs.drop_last
        assert list(bs) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def test_divisible_length(self, finite_stream_dataset):
        bs = StreamBatches(finite_stream_dataset, 1)
        assert list(bs) == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

    def test_nonpositive_batch_size(self, finite_stream_dataset):
        with pytest.raises(ValueError) as exc:
            StreamBatches(finite_stream_dataset, 0)
        assert 'batch size must be greater than 0' in str(exc.value)


@pytest.fixture
def stream_batches(stream_dataset):
    return StreamBatches(stream_dataset, 2)


@pytest.fixture
def finite_stream_batches(finite_stream_dataset):
    return StreamBatches(finite_stream_dataset, 2)


def test_can_be_iterated_twice(finite_stream_batches):
    bs_lst1 = list(finite_stream_batches)
    bs_lst2 = list(finite_stream_batches)
    assert len(bs_lst1) == len(bs_lst2)
    assert len(bs_lst2) > 0


def test_to_arrays(finite_stream_batches):
    ts = finite_stream_batches.to_arrays()
    assert isinstance(ts, Iterable)

    assert all(isinstance(t, np.ndarray) for t in ts)
    assert all(t.dtype == np.int32 for t in ts)
    assert [t.tolist() for t in ts] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]
