from collections.abc import Sequence

from text2tensor import Dataset

import pytest


def test_init():
    dat = Dataset(range(5))
    assert isinstance(dat, Sequence)


def test_init_samples_non_sequence():
    with pytest.raises(TypeError) as exc:
        Dataset(10)
    assert '"samples" is not a sequence' in str(exc.value)


@pytest.fixture
def dataset():
    return Dataset(range(5))


def test_getitem(dataset):
    for i in range(5):
        assert dataset[i] == i


def test_len(dataset):
    assert len(dataset) == 5


def test_shuffle(setup_rng, dataset):
    before = list(dataset)
    retval = dataset.shuffle()
    assert retval is dataset
    assert len(dataset) == len(before)
    assert all(s in dataset for s in before)
    assert list(dataset) != before


def test_batch(dataset):
    minibatches = dataset.batch(2)
    assert isinstance(minibatches, Sequence)
    assert len(minibatches) == 3
    assert minibatches[0] == range(0, 2)
    assert minibatches[1] == range(2, 4)
    assert minibatches[2] == range(4, 5)


def test_batch_exactly(dataset):
    minibatches = dataset.batch_exactly(2)
    assert isinstance(minibatches, Sequence)
    assert len(minibatches) == 2
    assert minibatches[0] == range(0, 2)
    assert minibatches[1] == range(2, 4)


def test_batch_nonpositive_batch_size(dataset):
    with pytest.raises(ValueError) as exc:
        dataset.batch(0)
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        dataset.batch_exactly(0)
    assert 'batch size must be greater than 0' in str(exc.value)
