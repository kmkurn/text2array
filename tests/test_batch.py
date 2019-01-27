from collections.abc import Sequence

import numpy as np
import pytest

from text2array import Batch, BatchArray


def test_init(samples):
    b = Batch(samples)
    assert isinstance(b, Sequence)
    assert len(b) == len(samples)
    for i in range(len(b)):
        assert b[i] == samples[i]


@pytest.fixture
def batch(samples):
    return Batch(samples)


def test_getattr(batch):
    assert isinstance(batch.x, Sequence)
    assert len(batch.x) == len(batch)
    for i in range(len(batch)):
        assert batch.x[i] == batch[i].x

    assert isinstance(batch.y, Sequence)
    assert len(batch.y) == len(batch)
    for i in range(len(batch)):
        assert batch.y[i] == pytest.approx(batch[i].y)


def test_getattr_invalid_name(batch):
    with pytest.raises(AttributeError) as exc:
        batch.z
    assert "some samples have no field 'z'" in str(exc.value)


def test_to_array(batch):
    arr = batch.to_array()
    assert isinstance(arr, BatchArray)

    assert isinstance(arr.x, np.ndarray)
    assert arr.x.tolist() == list(batch.x)
    assert isinstance(arr.y, np.ndarray)
    assert arr.y.shape[0] == len(batch)
    for i in range(len(batch)):
        assert arr.y[i] == pytest.approx(batch[i].y)


def test_to_array_no_common_field_names(samples):
    from text2array import SampleABC

    class FooSample(SampleABC):
        @property
        def fields(self):
            return {'z': 10}

    samples.append(FooSample())
    batch = Batch(samples)

    with pytest.raises(RuntimeError) as exc:
        batch.to_array()
    assert 'some samples have no common field names with the others' in str(exc.value)
