from typing import Iterator, Sequence

import pytest

from text2array import Batch, Dataset, Vocab


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


class TestShuffle:
    def test_mutable_seq(self, setup_rng, dataset):
        before = list(dataset)
        retval = dataset.shuffle()
        after = list(dataset)
        assert retval is dataset
        assert_shuffled(before, after)

    def test_immutable_seq(self, setup_rng, samples):
        dat = Dataset(tuple(samples))
        before = list(dat)
        retval = dat.shuffle()
        after = list(dat)
        assert retval is dat
        assert_shuffled(before, after)


class TestShuffleBy:
    dataset = Dataset([{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}])

    @staticmethod
    def key(sample):
        return sample['i']

    def test_ok(self, setup_rng):
        dat = self.dataset
        before = list(dat)
        retval = dat.shuffle_by(self.key)
        after = list(dat)
        assert retval is dat
        assert_shuffled(before, after)

    def test_zero_scale(self, setup_rng):
        dat = self.dataset
        before = list(dat)
        dat.shuffle_by(self.key, scale=0.)
        after = list(dat)
        assert sorted(before, key=self.key) == after

    def test_negative_scale(self, setup_rng):
        with pytest.raises(ValueError) as exc:
            self.dataset.shuffle_by(self.key, scale=-1)
        assert 'scale cannot be less than 0' in str(exc.value)


def assert_shuffled(before, after):
    assert before != after and len(before) == len(after) and all(x in after for x in before)


def test_batch():
    dat = Dataset([{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}])
    bs = dat.batch(2)
    assert isinstance(bs, Iterator)
    bs_lst = list(bs)
    assert len(bs_lst) == 3
    assert all(isinstance(b, Batch) for b in bs_lst)
    assert list(bs_lst[0]) == [dat[0], dat[1]]
    assert list(bs_lst[1]) == [dat[2], dat[3]]
    assert list(bs_lst[2]) == [dat[4]]


def test_batch_size_evenly_divides(dataset):
    bs = dataset.batch(1)
    bs_lst = list(bs)
    assert len(bs_lst) == len(dataset)
    for i in range(len(bs_lst)):
        assert list(bs_lst[i]) == [dataset[i]]


def test_batch_exactly():
    dat = Dataset([{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}])
    bs = dat.batch_exactly(2)
    assert isinstance(bs, Iterator)
    bs_lst = list(bs)
    assert len(bs_lst) == 2
    assert all(isinstance(b, Batch) for b in bs_lst)
    assert list(bs_lst[0]) == [dat[0], dat[1]]
    assert list(bs_lst[1]) == [dat[2], dat[3]]


def test_batch_nonpositive_batch_size(dataset):
    with pytest.raises(ValueError) as exc:
        next(dataset.batch(0))
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        next(dataset.batch_exactly(0))
    assert 'batch size must be greater than 0' in str(exc.value)


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
