from typing import Iterable, Sized

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
