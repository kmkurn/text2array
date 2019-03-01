from typing import Sequence

import pytest

from text2array import Dataset, Vocab


def test_init(samples):
    dat = Dataset(samples)
    assert isinstance(dat, Sequence)
    assert len(dat) == len(samples)
    for i in range(len(dat)):
        assert dat[i] == samples[i]


def test_init_samples_non_sequence():
    with pytest.raises(TypeError) as exc:
        Dataset(10)
    assert '"samples" is not a sequence' in str(exc.value)


class TestApplyVocab:
    def test_ok(self):
        dat = Dataset([{
            'w': 'a',
            'ws': ['a', 'b'],
            'cs': [['a', 'b'], ['b', 'a']],
            'i': 10,
            'j': 20
        }, {
            'w': 'b',
            'ws': ['a', 'a'],
            'cs': [['b', 'b'], ['b', 'a']],
            'i': 10,
            'j': 20
        }])
        vocab = {
            'w': {
                'a': 0,
                'b': 1
            },
            'ws': {
                'a': 2,
                'b': 3
            },
            'cs': {
                'a': 4,
                'b': 5
            },
            'j': {
                20: 2
            }
        }
        dat.apply_vocab(vocab)
        assert list(dat) == [{
            'w': 0,
            'ws': [2, 3],
            'cs': [[4, 5], [5, 4]],
            'i': 10,
            'j': 2
        }, {
            'w': 1,
            'ws': [2, 2],
            'cs': [[5, 5], [5, 4]],
            'i': 10,
            'j': 2
        }]

    def test_value_not_in_vocab(self):
        dat = Dataset([{'w': 'a'}])
        vocab = {'w': {'b': 0}}
        with pytest.raises(KeyError) as exc:
            dat.apply_vocab(vocab)
        assert "value 'a' not found in vocab" in str(exc.value)

        dat = Dataset([{'w': 10}])
        vocab = {'w': {11: 0}}
        with pytest.raises(KeyError) as exc:
            dat.apply_vocab(vocab)
        assert "value 10 not found in vocab" in str(exc.value)

    def test_with_vocab_object(self):
        dat = Dataset([{
            'ws': ['a', 'b'],
            'cs': [['a', 'c'], ['c', 'b', 'c']]
        }, {
            'ws': ['b'],
            'cs': [['b']]
        }])
        v = Vocab.from_samples(dat)
        dat.apply_vocab(v)
        assert list(dat) == [{
            'ws': [v['ws']['a'], v['ws']['b']],
            'cs': [[v['cs']['a'], v['cs']['c']], [v['cs']['c'], v['cs']['b'], v['cs']['c']]]
        }, {
            'ws': [v['ws']['b']],
            'cs': [[v['cs']['b']]]
        }]

    def test_immutable_seq(self, samples):
        ss = samples
        lstdat = Dataset(ss)
        tpldat = Dataset(tuple(ss))
        v = Vocab.from_samples(ss)
        lstdat.apply_vocab(v)
        tpldat.apply_vocab(v)
        assert list(lstdat) == list(tpldat)
