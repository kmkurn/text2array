from typing import Mapping, Sequence, Set

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

    def to_array(self) -> Mapping[FieldName, np.ndarray]:
        """Convert the batch into :class:`np.ndarray`.

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

        return {name: np.array(self.get(name)) for name in common}
