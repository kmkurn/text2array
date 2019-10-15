import pytest


@pytest.fixture
def rng():
    from random import Random
    return Random(42)


@pytest.fixture
def samples():
    return [{'i': i, 'f': (i + 1) / 3} for i in range(5)]


@pytest.fixture
def stream(samples):
    return Stream(samples)


@pytest.fixture
def stream_cls():
    return Stream


class Stream:
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples
