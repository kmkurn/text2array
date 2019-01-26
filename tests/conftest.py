import pytest

from text2array import Dataset, StreamDataset


@pytest.fixture
def setup_rng():
    import random
    random.seed(42)


@pytest.fixture
def samples():
    return [Sample(i, i * i) for i in range(5)]


@pytest.fixture
def dataset(samples):
    return Dataset(samples)


@pytest.fixture
def stream(samples):
    return Stream(samples)


@pytest.fixture
def stream_dataset(stream):
    return StreamDataset(stream)


class Sample:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, s):
        return self.x < s.x or (self.x == s.x and self.y < s.y)


class Stream:
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples
