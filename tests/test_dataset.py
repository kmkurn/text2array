from collections.abc import Sequence

import pytest

from text2tensor import Dataset


def test_init():
    dat = Dataset(range(5))
    assert isinstance(dat, Sequence)


def test_init_samples_non_sequence():
    with pytest.raises(TypeError) as exc:
        Dataset(10)
    assert '"samples" is not a sequence' in str(exc.value)


# TODO put this inside test_init
def test_getitem(dataset):
    for i in range(5):
        assert dataset[i] == i


# TODO put this inside test_init
def test_len(dataset):
    assert len(dataset) == 5


class TestShuffle:
    @pytest.fixture
    def tuple_dataset(self):
        return Dataset(tuple(range(5)))

    def assert_shuffle(self, dataset):
        before = list(dataset)
        retval = dataset.shuffle()
        after = list(dataset)

        assert retval is dataset
        assert len(before) == len(after)
        assert all(v in after for v in before)
        assert before != after

    def test_mutable_seq(self, setup_rng, dataset):
        self.assert_shuffle(dataset)

    def test_immutable_seq(self, setup_rng, tuple_dataset):
        self.assert_shuffle(tuple_dataset)


def test_batch(dataset):
    minibatches = dataset.batch(2)
    assert isinstance(minibatches, Sequence)
    assert len(minibatches) == 3
    assert minibatches[0] == [0, 1]
    assert minibatches[1] == [2, 3]
    assert minibatches[2] == [4]


def test_batch_exactly(dataset):
    minibatches = dataset.batch_exactly(2)
    assert isinstance(minibatches, Sequence)
    assert len(minibatches) == 2
    assert minibatches[0] == [0, 1]
    assert minibatches[1] == [2, 3]


def test_batch_nonpositive_batch_size(dataset):
    with pytest.raises(ValueError) as exc:
        dataset.batch(0)
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        dataset.batch_exactly(0)
    assert 'batch size must be greater than 0' in str(exc.value)
