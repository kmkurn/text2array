from collections.abc import Sequence

import numpy as np
import pytest

from text2array import Batches


class TestInit:
    def test_ok(self, dataset):
        bs = Batches(dataset, 2)
        assert bs.batch_size == 2
        assert not bs.drop_last
        assert isinstance(bs, Sequence)
        assert len(bs) == 3
        assert bs[0] == [dataset[0], dataset[1]]
        assert bs[1] == [dataset[2], dataset[3]]
        assert bs[2] == [dataset[4]]

    def test_kwargs(self, dataset):
        bs = Batches(dataset, 2, drop_last=True)
        assert bs.drop_last
        assert len(bs) == 2
        assert bs[0] == [dataset[0], dataset[1]]
        assert bs[1] == [dataset[2], dataset[3]]

    def test_nonpositive_batch_size(self, dataset):
        with pytest.raises(ValueError) as exc:
            Batches(dataset, 0)
        assert 'batch size must be greater than 0' in str(exc.value)


@pytest.fixture
def batches(dataset):
    return Batches(dataset, 2)


def test_getitem_negative_index(batches):
    n = len(batches)
    for i in range(n):
        assert batches[-i - 1] == batches[n - i - 1]


def test_getitem_index_error(batches):
    # index too large
    with pytest.raises(IndexError) as exc:
        batches[len(batches)]
    assert 'index out of range' in str(exc.value)

    # index too small
    with pytest.raises(IndexError) as exc:
        batches[-len(batches) - 1]
    assert 'index out of range' in str(exc.value)


@pytest.mark.skip
def test_to_arrays(batches):
    ts = batches.to_arrays()
    assert isinstance(ts, Sequence)
    assert len(ts) == len(batches)
    for i in range(len(ts)):
        t, b = ts[i], batches[i]

        assert isinstance(t.x, np.ndarray)
        assert t.x.dtype == np.int32
        assert t.x.tolist() == list(b.x)

        assert isinstance(t.y, np.ndarray)
        assert t.y.dtype == np.int32
        assert t.y.tolist() == list(b.y)
