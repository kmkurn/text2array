from collections.abc import Iterable
from itertools import takewhile

import pytest
import torch

from text2tensor import StreamBatches


class TestInit:
    def test_init(self, finite_stream_dataset):
        bs = StreamBatches(finite_stream_dataset, 2)
        assert bs.batch_size == 2
        assert not bs.drop_last
        assert isinstance(bs, Iterable)

        it = iter(bs)
        assert next(it) == [0, 1]
        assert next(it) == [2, 3]
        assert next(it) == [4, 5]
        assert next(it) == [6, 7]
        assert next(it) == [8, 9]
        assert next(it) == [10]
        with pytest.raises(StopIteration):
            next(it)

    def test_kwargs(self, finite_stream_dataset):
        bs = StreamBatches(finite_stream_dataset, 2, drop_last=True)
        assert bs.drop_last

        it = iter(bs)
        assert next(it) == [0, 1]
        assert next(it) == [2, 3]
        assert next(it) == [4, 5]
        assert next(it) == [6, 7]
        assert next(it) == [8, 9]
        with pytest.raises(StopIteration):
            next(it)

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


def test_to_tensors(stream_batches):
    ts = stream_batches.to_tensors()
    assert isinstance(ts, Iterable)

    bs = takewhile(lambda b: sum(b) < 30, stream_batches)
    for t, b in zip(ts, bs):
        assert torch.is_tensor(t)
        assert t.dtype == torch.long
        assert t.dim() == 1
        assert t.size(0) == len(b)


def test_to_tensors_returns_iterable(finite_stream_batches):
    ts = finite_stream_batches.to_tensors()
    print(list(ts))
    ts_lst1 = list(ts)
    ts_lst2 = list(ts)
    assert len(ts_lst1) == len(ts_lst2)
    assert len(ts_lst2) > 0
