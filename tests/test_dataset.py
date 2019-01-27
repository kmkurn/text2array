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
    @pytest.fixture
    def tuple_dataset(self, samples):
        return Dataset(tuple(samples))

    def assert_shuffle(self, dataset):
        before = list(dataset)
        retval = dataset.shuffle()
        after = list(dataset)

        assert retval is dataset
        assert before != after and sorted(before) == sorted(after)

    def test_mutable_seq(self, setup_rng, dataset):
        self.assert_shuffle(dataset)

    def test_immutable_seq(self, setup_rng, tuple_dataset):
        self.assert_shuffle(tuple_dataset)


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
