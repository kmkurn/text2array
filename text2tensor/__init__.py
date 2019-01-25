from collections.abc import Iterable, Iterator, MutableSequence, Sequence
import abc
import random
import typing as ty


class DatasetABC(Iterable, metaclass=abc.ABCMeta):  # pragma: no cover
    @abc.abstractmethod
    def batch(self, batch_size: int) -> ty.Iterable[Sequence]:
        pass

    @abc.abstractmethod
    def batch_exactly(self, batch_size: int) -> ty.Iterable[Sequence]:
        pass


class Dataset(DatasetABC, Sequence):
    """A dataset that fits in memory (no streaming).

    Args:
        samples: Sequence of samples the dataset should contain.
    """

    def __init__(self, samples: Sequence) -> None:
        if not isinstance(samples, Sequence):
            raise TypeError('"samples" is not a sequence')

        self._samples = samples

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def shuffle(self) -> 'Dataset':
        """Shuffle the dataset.

        This method shuffles in-place if ``samples`` is a mutable sequence.
        Otherwise, a copy is made and then shuffled. This copy is a mutable
        sequence, so subsequent shuffling will be done in-place.

        Returns:
            The dataset object itself (useful for chaining).
        """
        if isinstance(self._samples, MutableSequence):
            self._shuffle_inplace()
        else:
            self._shuffle_copy()
        return self

    def batch(self, batch_size: int) -> ty.List[Sequence]:
        """Group the samples in the dataset into batches.

        Args:
            batch_size: Maximum number of samples in each batch.

        Returns:
            The list of batches.
        """
        if batch_size <= 0:
            raise ValueError('batch size must be greater than 0')

        batches = []
        for begin in range(0, len(self._samples), batch_size):
            end = begin + batch_size
            batches.append(self._samples[begin:end])
        return batches

    def batch_exactly(self, batch_size: int) -> ty.List[Sequence]:
        """Group the samples in the dataset into batches of exact size.

        If the length of ``samples`` is not divisible by ``batch_size``, the last
        batch (whose length is less than ``batch_size``) is dropped.

        Args:
            batch_size: Number of samples in each batch.

        Returns:
            The list of batches.
        """
        batches = self.batch(batch_size)
        if len(self._samples) % batch_size != 0:
            assert len(batches[-1]) < batch_size
            batches = batches[:-1]
        return batches

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


class StreamDataset(DatasetABC, Iterable):
    """A dataset that streams its samples.

    Args:
        stream: Stream of examples the dataset should stream from.
    """

    def __init__(self, stream: Iterable) -> None:
        if not isinstance(stream, Iterable):
            raise TypeError('"stream" is not iterable')

        self._stream = stream

    def __iter__(self) -> Iterator:
        return iter(self._stream)

    def batch(self, batch_size: int) -> ty.Iterable[Sequence]:
        """Group the samples in the dataset into batches.

        Args:
            batch_size: Maximum number of samples in each batch.

        Returns:
            The iterable of batches.
        """
        return _Batches(self._stream, batch_size)

    def batch_exactly(self, batch_size: int) -> ty.Iterable[Sequence]:
        """Group the samples in the dataset into batches of exact size.

        If the length of ``samples`` is not divisible by ``batch_size``, the last
        batch (whose length is less than ``batch_size``) is dropped.

        Args:
            batch_size: Number of samples in each batch.

        Returns:
            The iterable of batches.
        """
        return _Batches(self._stream, batch_size, drop=True)


class _Batches(ty.Iterable[Sequence]):
    def __init__(self, stream: Iterable, bsize: int, drop: bool = False) -> None:
        if bsize <= 0:
            raise ValueError('batch size must be greater than 0')

        self._stream = stream
        self._bsize = bsize
        self._drop = drop

    def __iter__(self) -> ty.Iterator[Sequence]:
        it, exhausted = iter(self._stream), False
        while not exhausted:
            batch: list = []
            while not exhausted and len(batch) < self._bsize:
                try:
                    batch.append(next(it))
                except StopIteration:
                    exhausted = True
            if not self._drop or len(batch) == self._bsize:
                yield batch
