from typing import Callable, Iterable, Iterator, Mapping, MutableSequence, Sequence
import abc
import random
import statistics as stat

from .batches import Batch
from .samples import FieldName, FieldValue, Sample


class DatasetABC(Iterable[Sample], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def batch(self, batch_size: int) -> Iterator[Batch]:
        pass

    def batch_exactly(self, batch_size: int) -> Iterator[Batch]:
        """Group the samples in the dataset into batches of exact size.

        If the number of samples is not divisible by ``batch_size``, the last
        batch (whose length is less than ``batch_size``) is dropped.

        Args:
            batch_size: Number of samples in each batch.

        Returns:
            ~typing.Iterator[Batch]: The iterator of batches.
        """
        return (b for b in self.batch(batch_size) if len(b) == batch_size)

    @abc.abstractmethod
    def apply_vocab(self, vocab: Mapping[FieldName, Mapping[FieldValue, FieldValue]]) -> None:
        pass

    @classmethod
    def _apply_vocab_to_sample(
            cls,
            vocab: Mapping[FieldName, Mapping[FieldValue, FieldValue]],
            sample: Sample,
    ) -> Sample:
        s = {}
        for name, val in sample.items():
            try:
                vb = vocab[name]
            except KeyError:
                s[name] = val
            else:
                s[name] = cls._apply_vb_to_val(vb, val)
        return s

    @classmethod
    def _apply_vb_to_val(
            cls,
            vb: Mapping[FieldValue, FieldValue],
            val: FieldValue,
    ) -> FieldValue:
        if isinstance(val, str) or not isinstance(val, Sequence):
            try:
                return vb[val]
            except KeyError:
                raise KeyError(f'value {val!r} not found in vocab')

        return [cls._apply_vb_to_val(vb, v) for v in val]


class Dataset(DatasetABC, Sequence[Sample]):
    """Dataset that fits all in memory (no streaming).

    Args:
        samples (~typing.Sequence[Sample]): Sequence of samples the dataset
            should contain. This sequence should support indexing by a
            positive/negative index of type `int` or a `slice` object.
    """

    def __init__(self, samples: Sequence[Sample]) -> None:
        if not isinstance(samples, Sequence):
            raise TypeError('"samples" is not a sequence')

        self._samples = samples

    def __getitem__(self, index) -> Sample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def shuffle(self) -> 'Dataset':
        """Shuffle the dataset.

        This method shuffles in-place if ``samples`` is a mutable sequence.
        Otherwise, a copy is made and then shuffled. This copy is a mutable
        sequence, so subsequent shuffling will be done in-place.

        Returns:
            Dataset: This dataset object (useful for chaining).
        """
        if not isinstance(self._samples, MutableSequence):
            self._samples = list(self._samples)
        self._shuffle_inplace()
        return self

    def shuffle_by(self, key: Callable[[Sample], int], scale: float = 1.) -> 'Dataset':
        """Shuffle the dataset by the given key.

        This method essentially performs noisy sorting. The samples in the dataset are
        sorted ascending by the value of the given key, plus some random noise
        :math:`\epsilon \sim` Uniform :math:`(-z, z)`, where :math:`z` equals ``scale``
        times the standard deviation of key values. This formulation means that ``scale``
        regulates how noisy the sorting is. The larger it is, the more noisy the sorting
        becomes, i.e. it resembles random shuffling more closely. In an extreme case
        where ``scale=0``, this method just sorts the samples by ``key``. This method is
        useful when working with text data, where we want to shuffle the dataset and also
        minimize padding by ensuring that sentences of similar lengths are not too far apart.

        Args:
            key (typing.Callable[[Sample], int]): Callable to get the key value of a
                given sample.
            scale: Value to regulate the noise of the sorting. Must not be negative.

        Returns:
            Dataset: This dataset object (useful for chaining).
        """
        if scale < 0:
            raise ValueError('scale cannot be less than 0')

        std = stat.stdev(key(s) for s in self._samples)
        z = scale * std
        self._samples = sorted(self._samples, key=lambda s: key(s) + random.uniform(-z, z))
        return self

    def batch(self, batch_size: int) -> Iterator[Batch]:
        """Group the samples in the dataset into batches.

        Args:
            batch_size: Maximum number of samples in each batch.

        Returns:
            ~typing.Iterator[Batch]: The iterator of batches.
        """
        if batch_size <= 0:
            raise ValueError('batch size must be greater than 0')

        for begin in range(0, len(self._samples), batch_size):
            end = begin + batch_size
            yield Batch(self._samples[begin:end])

    def apply_vocab(self, vocab: Mapping[FieldName, Mapping[FieldValue, FieldValue]]) -> None:
        """Apply a vocabulary to this dataset.

        Applying a vocabulary means mapping all the (nested) field values to the corresponding
        values according to the mapping specified by the vocabulary. Field names that have
        no entry in the vocabulary are ignored. This method applies the vocabulary in-place
        when the dataset holds a mutable sequence of samples. Otherwise, a mutable copy of
        samples is made and the vocabulary is applied on it.

        Args:
            vocab (~typing.Mapping[FieldName, ~typing.Mapping[FieldValue, FieldValue]]): The
                vocabulary to apply.
        """
        if not isinstance(self._samples, MutableSequence):
            self._samples = list(self._samples)
        self._apply_vocab_inplace(vocab)

    def _shuffle_inplace(self) -> None:
        assert isinstance(self._samples, MutableSequence)
        n = len(self._samples)
        for i in range(n):
            j = random.randrange(n)
            temp = self._samples[i]
            self._samples[i] = self._samples[j]
            self._samples[j] = temp

    def _apply_vocab_inplace(
            self,
            vocab: Mapping[FieldName, Mapping[FieldValue, FieldValue]],
    ) -> None:
        assert isinstance(self._samples, MutableSequence)
        for i in range(len(self._samples)):
            self._samples[i] = self._apply_vocab_to_sample(vocab, self._samples[i])


class StreamDataset(DatasetABC):
    """Dataset that streams its samples.

    Args:
        stream (~typing.Iterable[Sample]): Stream of samples the dataset
            should stream from.
    """

    def __init__(self, stream: Iterable[Sample]) -> None:
        if not isinstance(stream, Iterable):
            raise TypeError('"stream" is not iterable')

        self._stream = stream

    def __iter__(self) -> Iterator[Sample]:
        try:
            vocab = self._vocab
        except AttributeError:
            yield from self._stream
            return

        for s in self._stream:
            yield self._apply_vocab_to_sample(vocab, s)

    def batch(self, batch_size: int) -> Iterator[Batch]:
        """Group the samples in the dataset into batches.

        Args:
            batch_size: Maximum number of samples in each batch.

        Returns:
            ~typing.Iterator[Batch]: The iterator of batches.
        """
        if batch_size <= 0:
            raise ValueError('batch size must be greater than 0')

        it, exhausted = iter(self._stream), False
        while not exhausted:
            batch: list = []
            while not exhausted and len(batch) < batch_size:
                try:
                    batch.append(next(it))
                except StopIteration:
                    exhausted = True
            if batch:
                yield Batch(batch)

    def apply_vocab(self, vocab: Mapping[FieldName, Mapping[FieldValue, FieldValue]]) -> None:
        """Apply a vocabulary to this dataset.

        Applying a vocabulary means mapping all the (nested) field values to the corresponding
        values according to the mapping specified by the vocabulary. Field names that have
        no entry in the vocabulary are ignored. Note that since the dataset holds a stream of
        samples, the actual application is delayed until the dataset is iterated. Therefore,
        ``vocab`` must still exist when that happens.

        Args:
            vocab (~typing.Mapping[FieldName, ~typing.Mapping[FieldValue, FieldValue]]): The
                vocabulary to apply.
        """
        self._vocab = vocab
