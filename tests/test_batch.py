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
    assert isinstance(batch.int_, Sequence)
    assert len(batch.int_) == len(batch)
    for i in range(len(batch)):
        assert batch.int_[i] == batch[i].int_

    assert isinstance(batch.float_, Sequence)
    assert len(batch.float_) == len(batch)
    for i in range(len(batch)):
        assert batch.float_[i] == pytest.approx(batch[i].float_)


def test_getattr_invalid_name(batch):
    with pytest.raises(AttributeError) as exc:
        batch.foo
    assert "some samples have no field 'foo'" in str(exc.value)


def test_to_array(batch):
    arr = batch.to_array()
    assert isinstance(arr, BatchArray)

    assert isinstance(arr.int_, np.ndarray)
    assert arr.int_.tolist() == list(batch.int_)
    assert isinstance(arr.float_, np.ndarray)
    assert arr.float_.shape[0] == len(batch)
    for i in range(len(batch)):
        assert arr.float_[i] == pytest.approx(batch[i].float_)
    assert isinstance(arr.str1, np.ndarray)
    assert arr.str1.tolist() == list(batch.str1)


def test_to_array_with_vocab(batch):
    vocab = {
        'str1': {v: i
                 for i, v in enumerate(batch.str1)},
        'str2': {v: i
                 for i, v in enumerate(batch.str2)},
    }
    arr = batch.to_array(vocab=vocab)
    assert arr.str1.dtype.name.startswith('int')
    assert arr.str1.tolist() == [vocab['str1'][s.str1] for s in batch]
    assert arr.str2.dtype.name.startswith('int')
    assert arr.str2.tolist() == [vocab['str2'][s.str2] for s in batch]


def test_to_array_no_common_field_names(samples):
    from text2array import SampleABC

    class FooSample(SampleABC):
        @property
        def fields(self):
            return {'foo': 10}

    samples.append(FooSample())
    batch = Batch(samples)

    with pytest.raises(RuntimeError) as exc:
        batch.to_array()
    assert 'some samples have no common field names with the others' in str(exc.value)
