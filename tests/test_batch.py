from collections.abc import Sequence

import pytest

from text2array import Batch


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
        assert batch.y[i] == batch[i].y


def test_getattr_invalid_name(batch):
    with pytest.raises(AttributeError) as exc:
        batch.z
    assert "some samples have no field 'z'" in str(exc.value)
