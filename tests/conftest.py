import pytest

from text2tensor import Dataset


@pytest.fixture
def setup_rng():
    import random
    random.seed(42)


@pytest.fixture
def dataset():
    return Dataset(list(range(5)))
