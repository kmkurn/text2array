from collections.abc import Sequence

import pytest

from text2array import Batches, Dataset


def test_init():
    dat = Dataset(range(5))
    assert isinstance(dat, Sequence)
    assert len(dat) == 5
    for i in range(len(dat)):
        assert dat[i] == i


def test_init_samples_non_sequence():
    with pytest.raises(TypeError) as exc:
        Dataset(10)
    assert '"samples" is not a sequence' in str(exc.value)


class TestShuffle:
    @pytest.fixture
    def tuple_dataset(self):
        return Dataset(tuple(range(5)))

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
    assert isinstance(bs, Batches)
    assert bs.batch_size == 2
    assert not bs.drop_last


def test_batch_exactly(dataset):
    bs = dataset.batch_exactly(2)
    assert isinstance(bs, Batches)
    assert bs.batch_size == 2
    assert bs.drop_last


def test_batch_nonpositive_batch_size(dataset):
    with pytest.raises(ValueError) as exc:
        dataset.batch(0)
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        dataset.batch_exactly(0)
    assert 'batch size must be greater than 0' in str(exc.value)
