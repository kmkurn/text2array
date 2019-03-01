from typing import Callable, Iterable, Iterator, Optional, Sequence, Sized
import random
import statistics as stat

from text2array import Batch, Sample


class BatchIterator(Iterable[Batch], Sized):
    def __init__(self, samples: Iterable[Sample], batch_size: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError('batch size must be greater than 0')

        self._samples = samples
        self._bsz = batch_size

    @property
    def batch_size(self) -> int:
        return self._bsz

    def __len__(self) -> int:
        n = len(self._samples)  # type: ignore
        b = self._bsz
        return n // b + (1 if n % b != 0 else 0)

    def __iter__(self) -> Iterator[Batch]:
        it, exhausted = iter(self._samples), False
        while not exhausted:
            batch: list = []
            while not exhausted and len(batch) < self._bsz:
                try:
                    batch.append(next(it))
                except StopIteration:
                    exhausted = True
            if batch:
                yield Batch(batch)


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
        std = stat.stdev(key(s) for s in self._samples)
        z = scale * std

        noises = [random.uniform(-z, z) for _ in range(len(self._samples))]
        indices = list(range(len(self._samples)))
        indices.sort(key=lambda i: key(self._samples[i]) + noises[i])
        shuf_samples = [self._samples[i] for i in indices]

        self._samples = shuf_samples
