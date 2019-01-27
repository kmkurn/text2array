from collections.abc import Mapping, Sequence

import numpy as np
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


def test_get(batch):
    assert isinstance(batch.get('i'), Sequence)
    assert len(batch.get('i')) == len(batch)
    for i in range(len(batch)):
        assert batch.get('i')[i] == batch[i].fields['i']

    assert isinstance(batch.get('f'), Sequence)
    assert len(batch.get('f')) == len(batch)
    for i in range(len(batch)):
        assert batch.get('f')[i] == pytest.approx(batch[i].fields['f'])


def test_get_invalid_name(batch):
    with pytest.raises(AttributeError) as exc:
        batch.get('foo')
    assert "some samples have no field 'foo'" in str(exc.value)


def test_to_array(batch):
    arr = batch.to_array()
    assert isinstance(arr, Mapping)

    assert isinstance(arr['i'], np.ndarray)
    assert arr['i'].tolist() == list(batch.get('i'))
    assert isinstance(arr['f'], np.ndarray)
    assert arr['f'].shape[0] == len(batch)
    for i in range(len(batch)):
        assert arr['f'][i] == pytest.approx(batch[i].fields['f'])


def test_to_array_no_common_field_names(samples):
    from text2array import SampleABC

    class FooSample(SampleABC):
        @property
        def fields(self):
            return {'foo': 10}

    samples_ = list(samples)
    samples_.append(FooSample())
    batch = Batch(samples_)

    with pytest.raises(RuntimeError) as exc:
        batch.to_array()
    assert 'some samples have no common field names with the others' in str(exc.value)
