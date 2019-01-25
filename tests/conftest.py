import pytest


@pytest.fixture
def setup_rng():
    import random
    random.seed(42)
