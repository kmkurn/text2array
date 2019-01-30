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


class TestToArray:
    def test_ok(self, batch):
        arr = batch.to_array()
        assert isinstance(arr, Mapping)

        assert isinstance(arr['i'], np.ndarray)
        assert arr['i'].shape == (len(batch), )
        assert arr['i'].tolist() == [s['i'] for s in batch]

        assert isinstance(arr['f'], np.ndarray)
        assert arr['f'].shape == (len(batch), )
        for i in range(len(batch)):
            assert arr['f'][i] == pytest.approx(batch[i]['f'])

    def test_empty(self):
        b = Batch([])
        assert not b.to_array()

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
