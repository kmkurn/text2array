from collections import Counter
from collections.abc import Sequence as SequenceABC
from typing import Iterable, Iterator, Mapping, Sequence

from .samples import FieldName, FieldValue, Sample


class Vocab(Mapping[FieldName, 'VocabEntry']):
    """Vocabulary containing the mapping from string field values to their integer indices.

    A vocabulary does not hold the mapping directly, but rather it stores a mapping from
    field names to :class:`VocabEntry` objects. These objects are the one actually holding
    the str-to-int mapping for that particular field name. In other words, the actual
    vocabulary is stored in :class:`VocabEntry` and namespaced by this :class:`Vocab` object.

    Args:
        m: Mapping from :obj:`FieldName` to its :class:`VocabEntry`.
    """

    def __init__(self, m: Mapping[FieldName, 'VocabEntry']) -> None:
        self._m = m

    def __len__(self) -> int:
        return len(self._m)

    def __iter__(self) -> Iterator[FieldName]:
        return iter(self._m)

    def __getitem__(self, name: FieldName) -> 'VocabEntry':
        try:
            return self._m[name]
        except KeyError:
            raise RuntimeError(f"no vocabulary found for field name '{name}'")

    # TODO limit vocab size
    @classmethod
    def from_samples(cls, samples: Iterable[Sample], min_count: int = 2) -> 'Vocab':
        """Make an instance of this class from an iterable of samples.

        A vocabulary is only made for fields whose value is a string or a (nested)
        sequence of strings. It is important that ``samples`` be a true iterable, i.e.
        it can be iterated more than once. No exception is raised when this is violated.

        Args:
            samples: Iterable of samples.
            min_count: Remove from the vocabulary string field values occurring fewer
                than this number of times.

        Returns:
            Vocabulary instance.
        """
        try:
            first = cls._head(samples)
        except StopIteration:
            return cls({})

        m = {
            name: VocabEntry.from_iterable(
                cls._flatten(cls._get_values(samples, name)), min_count=min_count)
            for name, value in first.items()
            if cls._needs_vocab(value)
        }
        return cls(m)

    @staticmethod
    def _head(x: Iterable[Sample]) -> Sample:
        return next(iter(x))

    @staticmethod
    def _get_values(ss: Iterable[Sample], name: FieldName) -> Iterator[FieldValue]:
        return (s[name] for s in ss)

    @classmethod
    def _needs_vocab(cls, val: FieldValue) -> bool:
        if isinstance(val, str):
            return True
        if isinstance(val, SequenceABC):
            if not val:
                raise ValueError('field values must not be an empty sequence')
            return cls._needs_vocab(val[0])
        return False

    @classmethod
    def _flatten(cls, xs) -> Iterator[str]:
        if isinstance(xs, str):
            yield xs
            return

        # must be an iterable, due to how we use this function
        for x in xs:
            yield from cls._flatten(x)


# TODO think if this class needs separate test cases
class VocabEntry(Sequence[str]):
    """Vocabulary entry that holds the actual str-to-int/int-to-str mapping.

    Args:
        strings: Sequence of distinct strings that serves as the int-to-str mapping.
    """

    def __init__(self, strings: Sequence[str]) -> None:
        # TODO maybe force strings to have no duplicates?
        self._itos = strings
        self._stoi = _StringStore.from_itos(strings)

    def __len__(self) -> int:
        return len(self._itos)

    def __getitem__(self, index) -> str:
        return self._itos[index]

    @property
    def stoi(self) -> Mapping[str, int]:
        """The str-to-int mapping."""
        return self._stoi

    @classmethod
    def from_iterable(cls, iterable: Iterable[str], min_count: int = 2) -> 'VocabEntry':
        """Make an instance of this class from an iterable of strings.

        Args:
            iterable: Iterable of strings.
            min_count: Remove from the vocabulary strings occurring fewer than this number
                of times.

        Returns:
            Vocab entry instance.
        """
        # TODO customize these tokens
        itos = ['<pad>', '<unk>']
        c = Counter(iterable)
        for v, f in c.most_common():
            if f < min_count:
                break
            itos.append(v)
        return cls(itos)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._itos!r})'


class _StringStore(Mapping[str, int]):
    def __init__(self, m: Mapping[str, int]) -> None:
        self._m = m

    def __len__(self) -> int:
        return len(self._m)

    def __iter__(self) -> Iterator[str]:
        return iter(self._m)

    def __getitem__(self, s: str) -> int:
        try:
            return self._m[s]
        except KeyError:
            # TODO customize unk id
            return 1

    @classmethod
    def from_itos(cls, itos: Sequence[str]) -> '_StringStore':
        assert len(set(itos)) == len(itos), 'itos cannot have duplicate strings'
        stoi = {s: i for i, s in enumerate(itos)}
        return cls(stoi)
