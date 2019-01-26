from typing import Iterable, Iterator, Sequence
import abc

import numpy as np

from .datasets import Dataset, StreamDataset

Batch = Sequence[int]


class BatchesABC(Iterable[Batch], metaclass=abc.ABCMeta):  # pragma: no cover
    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def drop_last(self) -> bool:
        pass

    @abc.abstractmethod
    def to_arrays(self) -> Iterable[np.ndarray]:
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

    @property
    def drop_last(self) -> bool:
        return self._drop

    def __getitem__(self, index):
        if index >= len(self) or index < -len(self):
            raise IndexError('index out of range')
        if index < 0:
            index += len(self)

        begin = index * self._bsize
        end = begin + self._bsize
        return self._dataset[begin:end]

    def __len__(self) -> int:
        q, r = divmod(len(self._dataset), self._bsize)
        return q + (1 if q > 0 and not self._drop else 0)

    def to_arrays(self) -> Sequence[np.ndarray]:
        """Convert each minibatch into an ndarray.

        Returns:
            The sequence of arrays.
        """
        ts = []
        for b in self:
            t = np.array(b, np.int32)
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

    @property
    def drop_last(self) -> bool:
        return self._drop

    def __iter__(self) -> Iterator[Batch]:
        it, exhausted = iter(self._dataset), False
        while not exhausted:
            batch: list = []
            while not exhausted and len(batch) < self._bsize:
                try:
                    batch.append(next(it))
                except StopIteration:
                    exhausted = True
            if len(batch) == self._bsize or (batch and not self._drop):
                yield batch

    def to_arrays(self) -> Iterable[np.ndarray]:
        """Convert each minibatch into an ndarray.

        Returns:
            The iterable of arrays.
        """
        return self._StreamArrays(self)

    class _StreamArrays(Iterable[np.ndarray]):
        def __init__(self, bs: 'StreamBatches') -> None:
            self._bs = bs

        def __iter__(self) -> Iterator[np.ndarray]:
            return (np.array(b, np.int32) for b in self._bs)
