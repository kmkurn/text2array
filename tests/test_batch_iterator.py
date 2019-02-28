from typing import Iterable, Sized

from text2array import Batch, BatchIterator, Dataset


def test_init(dataset):
    iter_ = BatchIterator(dataset)
    assert isinstance(iter_, Sized)
    assert len(iter_) == len(dataset)
    assert isinstance(iter_, Iterable)
    assert all(isinstance(b, Batch) for b in iter_)
    assert all(len(b) == 1 for b in iter_)
    assert all(b[0] == s for b, s in zip(iter_, dataset))


def test_init_kwargs(dataset):
    dat = Dataset([{'i': i} for i in range(5)])
    bsz = 2
    iter_ = BatchIterator(dat, batch_size=bsz)
    assert iter_.batch_size == bsz
    assert len(iter_) == len(dataset) // bsz + (1 if len(dataset) % bsz != 0 else 0)
    assert all(len(b) <= bsz for b in iter_)

    bs = list(iter_)
    assert list(bs[0]) == [dat[0], dat[1]]
    assert list(bs[1]) == [dat[2], dat[3]]
    assert list(bs[2]) == [dat[4]]
