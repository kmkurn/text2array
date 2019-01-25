from collections.abc import Iterable, Iterator, MutableSequence, Sequence
import random


class Dataset(Sequence):
    def __init__(self, samples: Sequence) -> None:
        if not isinstance(samples, Sequence):
            raise TypeError('"samples" is not a sequence')

        self._samples = samples

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def shuffle(self) -> 'Dataset':
        if isinstance(self._samples, list):
            random.shuffle(self._samples)
        elif isinstance(self._samples, MutableSequence):
            self._shuffle_inplace()
        else:
            self._shuffle_copy()
        return self

    def batch(self, batch_size: int) -> list:
        if batch_size <= 0:
            raise ValueError('batch size must be greater than 0')

        minibatches = []
        for begin in range(0, len(self._samples), batch_size):
            end = begin + batch_size
            minibatches.append(self._samples[begin:end])
        return minibatches

    def batch_exactly(self, batch_size: int) -> list:
        minibatches = self.batch(batch_size)
        if len(self._samples) % batch_size != 0:
            assert len(minibatches[-1]) < batch_size
            minibatches = minibatches[:-1]
        return minibatches

    def _shuffle_inplace(self) -> None:
        assert isinstance(self._samples, MutableSequence)
        n = len(self._samples)
        for i in range(n):
            j = random.randrange(n)
            temp = self._samples[i]
            self._samples[i] = self._samples[j]
            self._samples[j] = temp

    def _shuffle_copy(self) -> None:
        shuf_indices = list(range(len(self._samples)))
        random.shuffle(shuf_indices)
        shuf_samples = [self._samples[i] for i in shuf_indices]
        self._samples = shuf_samples


class StreamDataset(Iterable):
    def __init__(self, stream: Iterable) -> None:
        if not isinstance(stream, Iterable):
            raise TypeError('"stream" is not iterable')

        self._stream = stream

    def __iter__(self) -> Iterator:
        return iter(self._stream)

    def batch(self, batch_size: int) -> Iterable:
        return _Minibatches(self._stream, batch_size)

    def batch_exactly(self, batch_size: int) -> Iterable:
        return _Minibatches(self._stream, batch_size, drop=True)


class _Minibatches(Iterable):
    def __init__(self, stream: Iterable, bsize: int, drop: bool = False) -> None:
        if bsize <= 0:
            raise ValueError('batch size must be greater than 0')

        self._stream = stream
        self._bsize = bsize
        self._drop = drop

    def __iter__(self) -> Iterator:
        it, exhausted = iter(self._stream), False
        while not exhausted:
            minibatch: list = []
            while not exhausted and len(minibatch) < self._bsize:
                try:
                    minibatch.append(next(it))
                except StopIteration:
                    exhausted = True
            if not self._drop or len(minibatch) == self._bsize:
                yield minibatch
