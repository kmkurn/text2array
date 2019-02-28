from typing import Callable, Iterable, Iterator, Optional, Sequence, Sized
import random
import statistics as stat

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


class ShuffleIterator(Iterable[Sample], Sized):
    def __init__(
            self,
            samples: Sequence[Sample],
            key: Optional[Callable[[Sample], int]] = None,
            scale: float = 1.0,
    ) -> None:
        if scale < 0:
            raise ValueError('scale cannot be less than 0')

        self._samples = samples
        self._key = key
        self._scale = scale

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        if self._key is None:
            self._shuffle()
        else:
            self._shuffle_by(self._key, self._scale)
        return iter(self._samples)

    def _shuffle(self) -> None:
        self._samples = list(self._samples)
        random.shuffle(self._samples)

    def _shuffle_by(self, key: Callable[[Sample], int], scale: float) -> None:
        # TODO generate noises once and use those
        std = stat.stdev(key(s) for s in self._samples)
        z = scale * std
        self._samples = sorted(self._samples, key=lambda s: key(s) + random.uniform(-z, z))
