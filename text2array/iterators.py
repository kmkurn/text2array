from typing import Iterable, Iterator, Sized

from text2array import Batch, Dataset


# TODO accept iterable of samples?
class BatchIterator(Iterable[Batch], Sized):
    def __init__(self, dataset: Dataset, batch_size: int = 1) -> None:
        self._dat = dataset
        self._bsz = batch_size

    @property
    def batch_size(self) -> int:
        return self._bsz

    def __len__(self) -> int:
        n, b = len(self._dat), self._bsz
        return n // b + (1 if n % b != 0 else 0)

    def __iter__(self) -> Iterator[Batch]:
        return self._dat.batch(self._bsz)
