from collections.abc import Sequence as SequenceABC
from typing import Mapping, Sequence, Set, Union, cast

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

    def get(self, name: str) -> Sequence[FieldValue]:
        try:
            return [s[name] for s in self._samples]
        except KeyError:
            raise AttributeError(f"some samples have no field '{name}'")

    def to_array(self, pad_with: int = 0) -> Mapping[FieldName, np.ndarray]:
        """Convert the batch into :class:`np.ndarray`.

        Args:
            pad_with: Pad sequential field values with this number.

        Returns:
            A mapping from field names to :class:`np.ndarray`s whose first
            dimension corresponds to the batch size as returned by ``__len__``.
        """
        common: Set[FieldName] = set()
        for s in self._samples:
            if common:
                common.intersection_update(s)
            else:
                common = set(s)

        if not common:
            raise RuntimeError('some samples have no common field names with the others')
        assert self._samples  # if `common` isn't empty, neither is `_samples`

        arrs = {}
        for name in common:
            vs = self.get(name)
            if isinstance(vs[0], SequenceABC):
                vs = cast(Union[Sequence[Sequence[float]], Sequence[Sequence[int]]], vs)
                maxlen = max(len(v) for v in vs)
                vs = self._pad(vs, maxlen, with_=pad_with)
            arrs[name] = np.array(vs)

        return arrs

    @staticmethod
    def _pad(
            vs: Union[Sequence[Sequence[float]], Sequence[Sequence[int]]],
            maxlen: int,
            with_: int = 0,
    ) -> Union[Sequence[Sequence[float]], Sequence[Sequence[int]]]:
        res = []
        for v in vs:
            v, n = list(v), len(v)
            for _ in range(maxlen - n):
                v.append(with_)
            res.append(v)
        return res
