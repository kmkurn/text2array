import pytest

from text2tensor import Dataset, StreamDataset


class Counter:
    def __init__(self, limit=None):
        self._count = 0
        self._limit = limit

    def reset(self):
        self._count = 0

    def __iter__(self):
        while True:
            yield self._count
            self._count += 1
            if self._limit is not None and self._count >= self._limit:
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
    c = Counter()
    yield Counter()
    c.reset()


@pytest.fixture
def stream_dataset():
    c = Counter()
    yield StreamDataset(c)
    c.reset()


@pytest.fixture
def finite_stream_dataset():
    c = Counter(limit=11)
    yield StreamDataset(c)
    c.reset()
