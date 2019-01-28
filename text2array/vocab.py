from collections import Counter
from typing import Iterable, Mapping, Sequence

from .datasets import Dataset
from .samples import FieldName, FieldValue


class Vocab:
    def __init__(self, mapping: Mapping[FieldName, 'VocabEntry']) -> None:
        self._map = mapping

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> 'Vocab':
        vals = cls._get_values(dataset, 'w')
        return cls({'w': VocabEntry.from_iterable(vals)})

    def of(self, name: str) -> 'VocabEntry':
        return self._map[name]

    @staticmethod
    def _get_values(dat: Dataset, name: FieldName) -> Sequence[FieldValue]:
        return [s[name] for s in dat]


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


class _StringStore(Mapping[str, int]):
    def __init__(self, mapping: Mapping[str, int]) -> None:
        self._map = mapping

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def __getitem__(self, s):
        try:
            return self._map[s]
        except KeyError:
            return 1

    @classmethod
    def from_itos(cls, itos: Sequence[str]) -> '_StringStore':
        assert len(set(itos)) == len(itos), 'itos cannot have duplicate strings'
        stoi = {s: i for i, s in enumerate(itos)}
        return cls(stoi)
