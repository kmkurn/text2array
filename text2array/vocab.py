from collections import Counter
from collections.abc import Sequence as SequenceABC
from typing import Iterable, Iterator, Mapping, Sequence

from .samples import FieldName, FieldValue, Sample


class Vocab(Mapping[FieldName, 'VocabEntry']):
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

    # TODO mention in docstring that samples must be able to be iterated twice
    @classmethod
    def from_samples(cls, samples: Iterable[Sample]) -> 'Vocab':
        # TODO handle when samples is empty
        m = {
            name: VocabEntry.from_iterable(cls._flatten(cls._get_values(samples, name)))
            for name, value in cls._head(samples).items()
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
            assert len(val) > 0
            return cls._needs_vocab(val[0])
        return False

    @classmethod
    def _flatten(cls, xs) -> Iterator[str]:
        if isinstance(xs, str):
            yield xs
            return

        try:
            iter(xs)
        except TypeError:  # pragma: no cover
            yield xs
        else:
            for x in xs:
                yield from cls._flatten(x)


# TODO think if this class needs separate test cases
class VocabEntry(Sequence[str]):
    def __init__(self, strings: Sequence[str]) -> None:
        self._itos = strings
        self._stoi = _StringStore.from_itos(strings)

    def __len__(self) -> int:
        return len(self._itos)

    def __getitem__(self, index) -> str:
        return self._itos[index]

    @property
    def stoi(self) -> Mapping[str, int]:
        return self._stoi

    @classmethod
    def from_iterable(cls, iterable: Iterable[str]) -> 'VocabEntry':
        itos = ['<pad>', '<unk>']
        c = Counter(iterable)
        for v, f in c.most_common():
            if f < 2:
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
            return 1

    @classmethod
    def from_itos(cls, itos: Sequence[str]) -> '_StringStore':
        assert len(set(itos)) == len(itos), 'itos cannot have duplicate strings'
        stoi = {s: i for i, s in enumerate(itos)}
        return cls(stoi)
