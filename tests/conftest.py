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
        self.int_ = x
        self.float_ = y
        self.str1 = z
        self.str2 = w

    def __lt__(self, s):
        if self.int_ != s.int_:
            return self.int_ < s.int_
        if self.float_ != s.float_:
            return self.float_ < s.float_
        if self.str1 != s.str1:
            return self.str1 < s.str1
        return self.str2 < s.str2

    @property
    def fields(self):
        return {'int_': self.int_, 'float_': self.float_, 'str1': self.str1, 'str2': self.str2}


class Stream:
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples
