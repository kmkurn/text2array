from collections.abc import Sequence

import pytest
import torch

from text2tensor import Batches


class TestInit:
    def test_ok(self, dataset):
        bs = Batches(dataset, 2)
        assert bs.batch_size == 2
        assert isinstance(bs, Sequence)
        assert len(bs) == 3
        assert bs[0] == [0, 1]
        assert bs[1] == [2, 3]
        assert bs[2] == [4]

    def test_kwargs(self, dataset):
        bs = Batches(dataset, 2, drop_last=True)
        assert len(bs) == 2
        assert bs[0] == [0, 1]
        assert bs[1] == [2, 3]

    def test_nonpositive_batch_size(self, dataset):
        with pytest.raises(ValueError) as exc:
            Batches(dataset, 0)
        assert 'batch size must be greater than 0' in str(exc.value)


@pytest.fixture
def batches(dataset):
    return Batches(dataset, 2)


def test_getitem_negative_index(batches):
    assert batches[-1] == [4]


def test_getitem_index_error(batches):
    with pytest.raises(IndexError) as exc:
        batches[len(batches)]
    assert 'index out of range' in str(exc.value)


def test_to_tensors(batches):
    ts = batches.to_tensors()
    assert isinstance(ts, Sequence)
    assert len(ts) == len(batches)
    for i in range(len(ts)):
        t, b = ts[i], batches[i]
        assert torch.is_tensor(t)
        assert t.dtype == torch.long
        assert t.dim() == 1
        assert t.size(0) == len(b)
