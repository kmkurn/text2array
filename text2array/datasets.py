from typing import Iterable, Iterator, Mapping, MutableSequence, Sequence
import abc

from .samples import FieldName, FieldValue, Sample


class DatasetABC(Iterable[Sample], metaclass=abc.ABCMeta):
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
