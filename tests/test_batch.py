from typing import Mapping, MutableSequence

import numpy as np
import pytest

from text2array import Batch


def test_init(samples):
    b = Batch(samples)
    assert isinstance(b, MutableSequence)
    assert len(b) == len(samples)
    for i in range(len(b)):
        assert b[i] == samples[i]


class TestToArray:
    def test_ok(self):
        ss = [{
            'i': 4,
            'f': 0.67
        }, {
            'i': 2,
            'f': 0.89
        }, {
            'i': 3,
            'f': 0.23
        }, {
            'i': 5,
            'f': 0.11
        }, {
            'i': 3,
            'f': 0.22
        }]
        b = Batch(ss)
        arr = b.to_array()
        assert isinstance(arr, Mapping)
        assert len(arr) == 2
        assert set(arr) == set(['i', 'f'])

        assert isinstance(arr['i'], np.ndarray)
        assert arr['i'].shape == (len(b), )
        assert arr['i'].tolist() == [s['i'] for s in b]

        assert isinstance(arr['f'], np.ndarray)
        assert arr['f'].shape == (len(b), )
        assert arr['f'].tolist() == [pytest.approx(s['f']) for s in b]

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

    def test_pad_with_dict(self):
        ss = [{'is': [1], 'iss': [[1, 2], [1]]}, {'is': [1, 2], 'iss': [[1]]}]
        b = Batch(ss)
        arr = b.to_array(pad_with=dict(iss=9))
        assert arr['is'].tolist() == [[1, 0], [1, 2]]
        assert arr['iss'].tolist() == [[[1, 2], [1, 9]], [[1, 9], [9, 9]]]

    def test_missing_field(self):
        b = Batch([{'a': 10}, {'b': 20}])
        with pytest.raises(KeyError) as exc:
            b.to_array()
        assert "some samples have no field 'a'" in str(exc.value)

    def test_inconsistent_depth(self):
        b = Batch([{'ws': [1, 2]}, {'ws': [[1, 2], [3, 4]]}])
        with pytest.raises(ValueError) as exc:
            b.to_array()
        assert "field 'ws' has inconsistent nesting depth" in str(exc.value)

    def test_str(self):
        b = Batch([{'w': 'a'}, {'w': 'b'}])
        arr = b.to_array()
        assert isinstance(arr['w'], np.ndarray)
        assert arr['w'].tolist() == ['a', 'b']
