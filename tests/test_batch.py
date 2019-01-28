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


def test_get_invalid_name(batch):
    with pytest.raises(AttributeError) as exc:
        batch.get('foo')
    assert "some samples have no field 'foo'" in str(exc.value)


class TestToArray:
    def test_ok(self, batch):
        arr = batch.to_array()
        assert isinstance(arr, Mapping)

        assert isinstance(arr['i'], np.ndarray)
        assert arr['i'].shape == (len(batch), )
        assert arr['i'].tolist() == list(batch.get('i'))

        assert isinstance(arr['f'], np.ndarray)
        assert arr['f'].shape == (len(batch), )
        for i in range(len(batch)):
            assert arr['f'][i] == pytest.approx(batch[i]['f'])

    def test_seq(self):
        ss = [{'is': [1, 2]}, {'is': [1]}, {'is': [1, 2, 3]}, {'is': [1, 2]}]
        b = Batch(ss)
        arr = b.to_array()

        assert isinstance(arr['is'], np.ndarray)
        assert arr['is'].tolist() == [[1, 2, 0], [1, 0, 0], [1, 2, 3], [1, 2, 0]]

    def test_seq_of_seq(self):
        ss = [
            {
                'iss': [
                    [1],
                ]
            },
            {
                'iss': [
                    [1],
                    [1, 2],
                ]
            },
            {
                'iss': [
                    [1],
                    [1, 2, 3],
                    [1, 2],
                ]
            },
            {
                'iss': [
                    [1],
                    [1, 2],
                    [1, 2, 3],
                    [1],
                ]
            },
            {
                'iss': [
                    [1],
                    [1, 2],
                    [1, 2, 3],
                ]
            },
        ]
        b = Batch(ss)
        arr = b.to_array()

        assert isinstance(arr['iss'], np.ndarray)
        assert arr['iss'].shape == (5, 4, 3)
        assert arr['iss'][0].tolist() == [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        assert arr['iss'][1].tolist() == [
            [1, 0, 0],
            [1, 2, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        assert arr['iss'][2].tolist() == [
            [1, 0, 0],
            [1, 2, 3],
            [1, 2, 0],
            [0, 0, 0],
        ]
        assert arr['iss'][3].tolist() == [
            [1, 0, 0],
            [1, 2, 0],
            [1, 2, 3],
            [1, 0, 0],
        ]
        assert arr['iss'][4].tolist() == [
            [1, 0, 0],
            [1, 2, 0],
            [1, 2, 3],
            [0, 0, 0],
        ]

    def test_seq_of_seq_of_seq(self):
        ss = [
            {
                'isss': [
                    [[1], [1, 2]],
                    [[1], [1, 2], [1, 2]],
                ]
            },
            {
                'isss': [
                    [[1, 2], [1]],
                ]
            },
            {
                'isss': [
                    [[1, 2], [1, 2]],
                    [[1], [1, 2], [1]],
                    [[1, 2], [1]],
                ]
            },
            {
                'isss': [
                    [[1]],
                    [[1], [1, 2]],
                    [[1], [1, 2]],
                ]
            },
            {
                'isss': [
                    [[1]],
                    [[1], [1, 2]],
                    [[1], [1, 2], [1, 2]],
                    [[1], [1, 2], [1, 2]],
                ]
            },
        ]
        b = Batch(ss)
        arr = b.to_array()

        assert isinstance(arr['isss'], np.ndarray)
        assert arr['isss'].shape == (5, 4, 3, 2)
        assert arr['isss'][0].tolist() == [
            [[1, 0], [1, 2], [0, 0]],
            [[1, 0], [1, 2], [1, 2]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ]
        assert arr['isss'][1].tolist() == [
            [[1, 2], [1, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ]
        assert arr['isss'][2].tolist() == [
            [[1, 2], [1, 2], [0, 0]],
            [[1, 0], [1, 2], [1, 0]],
            [[1, 2], [1, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ]
        assert arr['isss'][3].tolist() == [
            [[1, 0], [0, 0], [0, 0]],
            [[1, 0], [1, 2], [0, 0]],
            [[1, 0], [1, 2], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ]
        assert arr['isss'][4].tolist() == [
            [[1, 0], [0, 0], [0, 0]],
            [[1, 0], [1, 2], [0, 0]],
            [[1, 0], [1, 2], [1, 2]],
            [[1, 0], [1, 2], [1, 2]],
        ]

    def test_custom_padding(self):
        ss = [{'is': [1]}, {'is': [1, 2]}]
        b = Batch(ss)
        arr = b.to_array(pad_with=9)
        assert arr['is'].tolist() == [[1, 9], [1, 2]]

        ss = [{'iss': [[1, 2], [1]]}, {'iss': [[1]]}]
        b = Batch(ss)
        arr = b.to_array(pad_with=9)
        assert arr['iss'].tolist() == [[[1, 2], [1, 9]], [[1, 9], [9, 9]]]

    def test_no_common_field_names(self, samples):
        samples_ = list(samples)
        samples_.append({'foo': 10})
        batch = Batch(samples_)

        with pytest.raises(RuntimeError) as exc:
            batch.to_array()
        assert 'some samples have no common field names with the others' in str(exc.value)
