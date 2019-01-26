from collections.abc import \
    Iterable as IterableABC, MutableSequence as MutableSequenceABC, Sequence as SequenceABC
from typing import Iterable, Iterator, List, Sequence
import abc
import random

import torch


class DatasetABC(Iterable[int], metaclass=abc.ABCMeta):  # pragma: no cover
    @abc.abstractmethod
    def batch(self, batch_size: int) -> Iterable[Sequence[int]]:
        pass

    @abc.abstractmethod
    def batch_exactly(self, batch_size: int) -> Iterable[Sequence[int]]:
        pass


Batch = Sequence[int]


class Dataset(DatasetABC, Sequence[int]):
    """A dataset that fits in memory (no streaming).

    Args:
        samples: Sequence of samples the dataset should contain.
    """

    def __init__(self, samples: Sequence[int]) -> None:
        if not isinstance(samples, SequenceABC):
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
        if isinstance(self._samples, MutableSequenceABC):
            self._shuffle_inplace()
        else:
            self._shuffle_copy()
        return self

    def batch(self, batch_size: int) -> List[Batch]:
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

    def batch_exactly(self, batch_size: int) -> List[Batch]:
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
        assert isinstance(self._samples, MutableSequenceABC)
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


class StreamDataset(DatasetABC, Iterable[int]):
    """A dataset that streams its samples.

    Args:
        stream: Stream of examples the dataset should stream from.
    """

    def __init__(self, stream: Iterable[int]) -> None:
        if not isinstance(stream, IterableABC):
            raise TypeError('"stream" is not iterable')

        self._stream = stream

    def __iter__(self) -> Iterator[int]:
        return iter(self._stream)

    def batch(self, batch_size: int) -> Iterable[Batch]:
        """Group the samples in the dataset into batches.

        Args:
            batch_size: Maximum number of samples in each batch.

        Returns:
            The iterable of batches.
        """
        return _Batches(self._stream, batch_size)

    def batch_exactly(self, batch_size: int) -> Iterable[Batch]:
        """Group the samples in the dataset into batches of exact size.

        If the length of ``samples`` is not divisible by ``batch_size``, the last
        batch (whose length is less than ``batch_size``) is dropped.

        Args:
            batch_size: Number of samples in each batch.

        Returns:
            The iterable of batches.
        """
        return _Batches(self._stream, batch_size, drop=True)


class _Batches(Iterable[Batch]):
    def __init__(self, stream: Iterable[int], bsize: int, drop: bool = False) -> None:
        if bsize <= 0:
            raise ValueError('batch size must be greater than 0')

        self._stream = stream
        self._bsize = bsize
        self._drop = drop

    def __iter__(self) -> Iterator[Batch]:
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


class BatchesABC(Iterable[Batch], metaclass=abc.ABCMeta):  # pragma: no cover
    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        pass

    @abc.abstractmethod
    def to_tensors(self) -> Iterable[torch.LongTensor]:
        pass


class Batches(BatchesABC, Sequence[Batch]):
    """A class to represent a sequence of minibatches.

    Args:
        dataset: Dataset to make batches from.
        batch_size: Maximum number of samples in each batch.
        drop_last (optional): Whether to drop the last batch when ``batch_size`` does not
            evenly divide the length of ``dataset``.
    """

    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False) -> None:
        if batch_size <= 0:
            raise ValueError('batch size must be greater than 0')

        self._dataset = dataset
        self._bsize = batch_size
        self._drop = drop_last

    @property
    def batch_size(self) -> int:
        return self._bsize

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('index out of range')
        if index < 0:
            index += len(self)

        begin = index * self._bsize
        end = begin + self._bsize
        return self._dataset[begin:end]

    def __len__(self) -> int:
        q, r = divmod(len(self._dataset), self._bsize)
        return q + (1 if q > 0 and not self._drop else 0)

    def to_tensors(self) -> List[torch.LongTensor]:
        """Convert each minibatch into a tensor.

        Returns:
            The list of tensors.
        """
        ts = []
        for b in self:
            t = torch.tensor(b, dtype=torch.long)
            ts.append(t)
        return ts


class StreamBatches(BatchesABC, Iterable[Batch]):
    """A class to represent an iterable of minibatches.

    Args:
        dataset: Dataset to make batches from.
        batch_size: Maximum number of samples in each batch.
        drop_last (optional): Whether to drop the last batch when ``batch_size`` does not
            evenly divide the length of ``dataset``.
    """

    def __init__(
            self, dataset: StreamDataset, batch_size: int, drop_last: bool = False) -> None:
        if batch_size <= 0:
            raise ValueError('batch size must be greater than 0')

        self._dataset = dataset
        self._bsize = batch_size
        self._drop = drop_last

    @property
    def batch_size(self) -> int:
        return self._bsize

    def __iter__(self) -> Iterator[Batch]:
        it, exhausted = iter(self._dataset), False
        while not exhausted:
            batch: list = []
            while not exhausted and len(batch) < self._bsize:
                try:
                    batch.append(next(it))
                except StopIteration:
                    exhausted = True
            if not self._drop or len(batch) == self._bsize:
                yield batch

    def to_tensors(self) -> Iterable[torch.LongTensor]:
        """Convert each minibatch into a tensor.

        Returns:
            The iterable of tensors.
        """
        return self._StreamTensors(self)

    class _StreamTensors(Iterable[torch.LongTensor]):
        def __init__(self, bs: 'StreamBatches') -> None:
            self._bs = bs

        def __iter__(self) -> Iterator[torch.LongTensor]:
            yield from (torch.tensor(b, dtype=torch.long) for b in self._bs)
