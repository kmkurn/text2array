from typing import Callable, Iterable, Iterator, Optional, Sized

from text2array import Batch, Dataset, Sample


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


# TODO accept sequence of samples?
class ShuffleIterator(Iterable[Sample], Sized):
    def __init__(
            self,
            dataset: Dataset,
            key: Optional[Callable[[Sample], int]] = None,
            scale: float = 1.0,
    ) -> None:
        if scale < 0:
            raise ValueError('scale cannot be less than 0')

        self._dat = dataset
        self._key = key
        self._scale = scale

    def __len__(self) -> int:
        return len(self._dat)

    def __iter__(self) -> Iterator[Sample]:
        if self._key is None:
            self._dat.shuffle()
        else:
            self._dat.shuffle_by(self._key, scale=self._scale)
        return iter(self._dat)
