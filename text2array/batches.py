from collections.abc import Sequence as SequenceABC
from functools import reduce
from typing import Any, List, Mapping, Sequence, Set, Union

import numpy as np

from .samples import FieldName, FieldValue, Sample


class Batch(Sequence[Sample]):
    """A class to represent a single batch.

    Args:
        samples: Sequence of samples this batch should contain.
    """

    def __init__(self, samples: Sequence[Sample]) -> None:
        self._samples = samples

    def __getitem__(self, index) -> Sample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def to_array(self, pad_with: int = 0) -> Mapping[FieldName, np.ndarray]:
        """Convert the batch into :class:`np.ndarray`.

        Args:
            pad_with: Pad sequential field values with this number.

        Returns:
            A mapping from field names to :class:`np.ndarray`s whose first
            dimension corresponds to the batch size as returned by ``__len__``.
        """
        if not self._samples:
            return {}

        arr = {}
        for name in self._samples[0].keys():
            data = self._get_values(name)
            # Get max length for all depths, 1st elem is batch size
            maxlens = self._get_maxlens(data)
            # Get padding for all depths
            paddings = self._get_paddings(maxlens, pad_with)
            # Pad the data
            data = self._pad(data, maxlens, paddings, 0)

            arr[name] = np.array(data)

        return arr

    def _get_values(self, name: str) -> Sequence[FieldValue]:
        try:
            return [s[name] for s in self._samples]
        except KeyError:
            raise KeyError(f"some samples have no field '{name}'")

    @classmethod
    def _get_maxlens(cls, data: Sequence[Any]) -> List[int]:
        assert data

        # Base case
        if not isinstance(data[0], SequenceABC):
            return [len(data)]

        # Recursive case
        maxlenss = [cls._get_maxlens(x) for x in data]
        assert all(len(x) == len(maxlenss[0]) for x in maxlenss)

        maxlens = reduce(lambda ml1, ml2: [max(l1, l2) for l1, l2 in zip(ml1, ml2)], maxlenss)
        maxlens.insert(0, len(data))
        return maxlens

    @classmethod
    def _get_paddings(cls, maxlens: List[int], with_: int) -> List[Union[int, List[int]]]:
        res: list = [with_]
        for maxlen in reversed(maxlens[1:]):
            res.append([res[-1] for _ in range(maxlen)])
        res.reverse()
        return res

    @classmethod
    def _pad(
            cls,
            data: Sequence[Any],
            maxlens: List[int],
            paddings: List[Union[int, List[int]]],
            depth: int,
    ) -> Sequence[Any]:
        assert data
        assert len(maxlens) == len(paddings)
        assert depth < len(maxlens)

        # Base case
        if not isinstance(data[0], SequenceABC):
            data_ = list(data)
        # Recursive case
        else:
            data_ = [cls._pad(x, maxlens, paddings, depth + 1) for x in data]

        for _ in range(maxlens[depth] - len(data)):
            data_.append(paddings[depth])
        return data_
