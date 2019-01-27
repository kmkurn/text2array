import pytest

from text2array import Dataset, SampleABC, StreamDataset


@pytest.fixture
def setup_rng():
    import random
    random.seed(42)


@pytest.fixture
def samples():
    return [TestSample(i, i * i) for i in range(5)]


@pytest.fixture
def dataset(samples):
    return Dataset(samples)


@pytest.fixture
def stream(samples):
    return Stream(samples)


@pytest.fixture
def stream_dataset(stream):
    return StreamDataset(stream)


class TestSample(SampleABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, s):
        return self.x < s.x or (self.x == s.x and self.y < s.y)

    @property
    def fields(self):
        return {'x': self.x, 'y': self.y}


class Stream:
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples
