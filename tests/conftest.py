import pytest

from text2array import Dataset, StreamDataset


class Counter:
    def __init__(self, limit=None):
        self.count = 0
        self.limit = limit

    def __iter__(self):
        self.count = 0
        while True:
            yield self.count
            self.count += 1
            if self.limit is not None and self.count >= self.limit:
                break


@pytest.fixture
def setup_rng():
    import random
    random.seed(42)


@pytest.fixture
def dataset():
    return Dataset(list(range(5)))


@pytest.fixture
def counter():
    return Counter(limit=11)


@pytest.fixture
def stream_dataset(counter):
    return StreamDataset(counter)
