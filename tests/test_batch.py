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
        assert batch.get('i')[i] == batch[i]['i']

    assert isinstance(batch.get('f'), Sequence)
    assert len(batch.get('f')) == len(batch)
    for i in range(len(batch)):
        assert batch.get('f')[i] == pytest.approx(batch[i]['f'])

    assert isinstance(batch.get('is'), Sequence)
    assert len(batch.get('is')) == len(batch)
    for i in range(len(batch)):
        assert list(batch.get('is')[i]) == list(batch[i]['is'])

    assert isinstance(batch.get('fs'), Sequence)
    assert len(batch.get('fs')) == len(batch)
    for i in range(len(batch)):
        for f1, f2 in zip(batch.get('fs')[i], batch[i]['fs']):
            assert f1 == pytest.approx(f2)


def test_get_invalid_name(batch):
    with pytest.raises(AttributeError) as exc:
        batch.get('foo')
    assert "some samples have no field 'foo'" in str(exc.value)


def test_to_array(batch):
    arr = batch.to_array()
    assert isinstance(arr, Mapping)

    assert isinstance(arr['i'], np.ndarray)
    assert arr['i'].shape == (len(batch), )
    assert arr['i'].tolist() == list(batch.get('i'))

    assert isinstance(arr['f'], np.ndarray)
    assert arr['f'].shape == (len(batch), )
    for i in range(len(batch)):
        assert arr['f'][i] == pytest.approx(batch[i]['f'])

    assert isinstance(arr['is'], np.ndarray)
    maxlen = max(len(x) for x in batch.get('is'))
    assert arr['is'].shape == (len(batch), maxlen)
    for r, s in zip(arr['is'], batch):
        assert r[:len(s['is'])].tolist() == list(s['is'])
        assert all(c == 0 for c in r[len(s['is']):])

    assert isinstance(arr['fs'], np.ndarray)
    maxlen = max(len(x) for x in batch.get('fs'))
    assert arr['fs'].shape == (len(batch), maxlen)
    for r, s in zip(arr['fs'], batch):
        for c, f in zip(r, s['fs']):
            assert c == pytest.approx(f)
        assert all(c == pytest.approx(0, abs=1e-7) for c in r[len(s['fs']):])


def test_to_array_custom_padding(batch):
    arr = batch.to_array(pad_with=1)
    for r, s in zip(arr['is'], batch):
        assert all(c == 1 for c in r[len(s['is']):])
    for r, s in zip(arr['fs'], batch):
        assert all(c == pytest.approx(1) for c in r[len(s['fs']):])


def test_to_array_no_common_field_names(samples):
    samples_ = list(samples)
    samples_.append({'foo': 10})
    batch = Batch(samples_)

    with pytest.raises(RuntimeError) as exc:
        batch.to_array()
    assert 'some samples have no common field names with the others' in str(exc.value)
