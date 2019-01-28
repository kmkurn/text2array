from collections.abc import Iterator, Sequence

import pytest

from text2array import Batch, Dataset


def test_init(samples):
    dat = Dataset(samples)
    assert isinstance(dat, Sequence)
    assert len(dat) == 5
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
    # TODO make this a class variable
    @staticmethod
    def make_dataset():
        return Dataset([{
            'is': [1, 2, 3]
        }, {
            'is': [1]
        }, {
            'is': [1, 2]
        }, {
            'is': [1, 2, 3, 4, 5]
        }, {
            'is': [1, 2, 3, 4]
        }])

    @staticmethod
    def key(sample):
        return len(sample['is'])

    def test_ok(self, setup_rng):
        dat = self.make_dataset()
        before = list(dat)
        retval = dat.shuffle_by(self.key)
        after = list(dat)
        assert retval is dat
        assert_shuffled(before, after)

    def test_zero_scale(self, setup_rng):
        dat = self.make_dataset()
        before = list(dat)
        dat.shuffle_by(self.key, scale=0.)
        after = list(dat)
        assert sorted(before, key=self.key) == after

    def test_negative_scale(self, setup_rng):
        dat = self.make_dataset()
        with pytest.raises(ValueError) as exc:
            dat.shuffle_by(self.key, scale=-1)
        assert 'scale cannot be less than 0' in str(exc.value)


def assert_shuffled(before, after):
    assert before != after and len(before) == len(after) and all(x in after for x in before)


def test_batch(dataset):
    bs = dataset.batch(2)
    assert isinstance(bs, Iterator)
    bs_lst = list(bs)
    assert len(bs_lst) == 3
    assert all(isinstance(b, Batch) for b in bs_lst)
    assert list(bs_lst[0]) == [dataset[0], dataset[1]]
    assert list(bs_lst[1]) == [dataset[2], dataset[3]]
    assert list(bs_lst[2]) == [dataset[4]]


def test_batch_size_evenly_divides(dataset):
    bs = dataset.batch(1)
    bs_lst = list(bs)
    assert len(bs_lst) == len(dataset)
    for i in range(len(bs_lst)):
        assert list(bs_lst[i]) == [dataset[i]]


def test_batch_exactly(dataset):
    bs = dataset.batch_exactly(2)
    assert isinstance(bs, Iterator)
    bs_lst = list(bs)
    assert len(bs_lst) == 2
    assert all(isinstance(b, Batch) for b in bs_lst)
    assert list(bs_lst[0]) == [dataset[0], dataset[1]]
    assert list(bs_lst[1]) == [dataset[2], dataset[3]]


def test_batch_nonpositive_batch_size(dataset):
    with pytest.raises(ValueError) as exc:
        next(dataset.batch(0))
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        next(dataset.batch_exactly(0))
    assert 'batch size must be greater than 0' in str(exc.value)
