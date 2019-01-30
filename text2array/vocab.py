from collections import Counter, OrderedDict
from collections.abc import Sequence as SequenceABC
from typing import Iterable, Iterator, Mapping, MutableMapping, Optional

from .samples import FieldName, FieldValue, Sample


class Vocab(Mapping[FieldName, Mapping[str, int]]):
    """Namespaced vocabulary storing the mapping from field names to their actual vocabulary.

    A vocabulary does not hold the str-to-int mapping directly, but rather it stores a mapping
    from field names to the corresponding str-to-int mappings. These mappings are the actual
    vocabulary for that particular field name. In other words, the actual vocabulary for each
    field name is namespaced by the field name and all of them are handled this :class:`Vocab`
    object.

    Args:
        m: Mapping from :obj:`FieldName` to its str-to-int mapping.
    """

    def __init__(self, m: Mapping[FieldName, Mapping[str, int]]) -> None:
        self._m = m

    def __len__(self) -> int:
        return len(self._m)

    def __iter__(self) -> Iterator[FieldName]:
        return iter(self._m)

    def __getitem__(self, name: FieldName) -> Mapping[str, int]:
        try:
            return self._m[name]
        except KeyError:
            raise KeyError(f"no vocabulary found for field name '{name}'")

    @classmethod
    def from_samples(
            cls,
            samples: Iterable[Sample],
            options: Optional[Mapping[FieldName, dict]] = None,
    ) -> 'Vocab':
        """Make an instance of this class from an iterable of samples.

        A vocabulary is only made for fields whose value is a string token or a (nested)
        sequence of string tokens. It is important that ``samples`` be a true iterable, i.e.
        it can be iterated more than once. No exception is raised when this is violated.

        Args:
            samples: Iterable of samples.
            options: Mapping from field names to dictionaries to control the creation of
                the str-to-int mapping. Allowed dictionary keys are:

                * ``min_count``(:obj:`int`): Exclude tokens occurring fewer than this number
                    of times from the vocabulary (default: 2).
                * ``pad``(:obj:`str`): String token to represent padding tokens. If ``None``,
                    no padding token is added to the vocabulary. Otherwise, it is the
                    first entry in the vocabulary (index is 0) (default: ``<pad>``).
                * ``unk``(:obj:`str`): String token to represent unknown tokens with. If
                    ``None``, no unknown token is added to the vocabulary. This means when
                    querying the vocabulary with such token, an error is raised. Otherwise,
                    it is the first entry in the vocabulary *after* ``pad``, if any (index is
                    either 0 or 1) (default: ``<unk>``).
                * ``max_size``(:obj:`int`): Maximum size of the vocabulary, excluding ``pad``
                    and ``unk``. If ``None``, no limit on the vocabulary size. Otherwise, at
                    most, only this number of most frequent tokens are included in the
                    vocabulary. Note that ``min_count`` also sets the maximum size implicitly.
                    So, the size is limited by whichever is smaller. (default: ``None``).

        Returns:
            Vocabulary instance.
        """
        # TODO don't waste the first
        try:
            first = cls._head(samples)
        except StopIteration:
            return cls({})

        if options is None:
            options = {}

        m = {
            name: _StringStore._from_iterable(
                cls._flatten(cls._get_values(samples, name)), **options.get(name, {}))
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


class _StringStore(Mapping[str, int]):
    def __init__(self, m: Mapping[str, int], unk_id: Optional[int] = None) -> None:
        assert unk_id is None or unk_id >= 0
        self._m = m
        self._unk_id = unk_id

    def __len__(self) -> int:
        return len(self._m)

    def __iter__(self) -> Iterator[str]:
        return iter(self._m)

    def __getitem__(self, s: str) -> int:
        try:
            return self._m[s]
        except KeyError:
            if self._unk_id is not None:
                return self._unk_id
            raise KeyError(f"'{s}' not found in vocabulary")

    def __contains__(self, s) -> bool:
        return s in self._m

    @classmethod
    def _from_iterable(
            cls,
            iterable: Iterable[str],
            min_count: int = 2,
            unk: Optional[str] = '<unk>',
            pad: Optional[str] = '<pad>',
            max_size: Optional[int] = None,
    ) -> '_StringStore':
        stoi: MutableMapping[str, int] = OrderedDict()
        if pad is not None:
            stoi[pad] = len(stoi)
        if unk is not None:
            stoi[unk] = len(stoi)

        n = len(stoi)
        c = Counter(iterable)
        for s, f in c.most_common():
            if f < min_count or (max_size is not None and len(stoi) - n >= max_size):
                break
            stoi[s] = len(stoi)

        unk_id = None if unk is None else stoi[unk]
        return cls(stoi, unk_id=unk_id)
