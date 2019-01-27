import pytest

from text2array import Dataset, SampleABC, StreamDataset


@pytest.fixture
def setup_rng():
    import random
    random.seed(42)


@pytest.fixture
def samples():
    return [TestSample(i, (i + 1) / 3, f'word-{i}', f'token-{i}') for i in range(5)]


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
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __lt__(self, s):
        if self.x != s.x:
            return self.x < s.x
        if self.y != s.y:
            return self.y < s.y
        if self.z != s.z:
            return self.z < s.z
        return self.w < s.w

    @property
    def fields(self):
        return {'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w}


class Stream:
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples
