from typing import Iterable, Sized

import pytest

from text2array import Batch, BucketIterator


def test_init():
    samples = [{"ns": list(range(n + 1))} for n in range(100)]
    bucket_key = lambda s: (len(s["ns"]) - 1) // 10

    iter_ = BucketIterator(samples, bucket_key, batch_size=3)

    assert isinstance(iter_, Sized)
    assert len(iter_) == 40
    assert isinstance(iter_, Iterable)
    assert all(isinstance(b, Batch) for b in iter_)
    assert all(len(b) <= iter_.batch_size for b in iter_)
    assert all(len(set(bucket_key(s) for s in b)) == 1 for b in iter_)


def test_shuffle_bucket(rng):
    samples = [{"ns": list(range(n + 1))} for n in range(100)]
    bucket_key = lambda s: (len(s["ns"]) - 1) // 10

    iter_ = BucketIterator(samples, bucket_key, batch_size=3)

    bs1 = [list(b) for b in iter_]
    bs2 = [list(b) for b in iter_]
    assert bs1 == bs2

    iter_ = BucketIterator(samples, bucket_key, batch_size=3, shuffle_bucket=True, rng=rng)

    bs1 = [list(b) for b in iter_]
    bs2 = [list(b) for b in iter_]
    assert bs1 != bs2


def test_sort_bucket():
    samples = [{"n": n} for n in range(100)]
    bucket_key = lambda s: s["n"] // 10

    iter_ = BucketIterator(
        samples, bucket_key, batch_size=3, sort_bucket=True, sort_bucket_by=lambda s: -s["n"]
    )

    for b in iter_:
        for s, s_ in zip(b, b[1:]):
            assert s["n"] > s_["n"]


def test_shuffle_and_sort_bucket():
    with pytest.warns(UserWarning):
        BucketIterator([], len, shuffle_bucket=True, sort_bucket=True)
