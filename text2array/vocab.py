from collections import Counter, OrderedDict
from collections.abc import Sequence as SequenceABC
from typing import Iterable, Iterator, Mapping, Optional

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

    # TODO limit vocab size for each field name
    @classmethod
    def from_samples(
            cls,
            samples: Iterable[Sample],
            options: Optional[Mapping[FieldName, dict]] = None,
    ) -> 'Vocab':
        """Make an instance of this class from an iterable of samples.

        A vocabulary is only made for fields whose value is a string or a (nested)
        sequence of strings. It is important that ``samples`` be a true iterable, i.e.
        it can be iterated more than once. No exception is raised when this is violated.

        Args:
            samples: Iterable of samples.
            options: Mapping from field names to dictionaries to control the creation of
                the str-to-int mapping. Allowed dictionary keys are:

                * ``min_count`` - Exclude strings occurring fewer than this number of times
                    from the vocabulary.

        Returns:
            Vocabulary instance.
        """
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
    def _from_iterable(cls, iterable: Iterable[str], min_count: int = 2) -> '_StringStore':
        # TODO customize these tokens
        stoi = OrderedDict([('<pad>', 0), ('<unk>', 1)])
        c = Counter(iterable)
        for s, f in c.most_common():
            if f < min_count:
                break
            stoi[s] = len(stoi)
        return cls(stoi)
