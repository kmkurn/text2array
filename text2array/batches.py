from typing import Mapping, Optional, Sequence, Set

import numpy as np

from .samples import FieldName, FieldValue, SampleABC


class Batch(Sequence[SampleABC]):
    """A class to represent a single batch.

    Args:
        samples: Sequence of samples this batch should contain.
    """

    def __init__(self, samples: Sequence[SampleABC]) -> None:
        self._samples = samples

    def __getitem__(self, index) -> SampleABC:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def __getattr__(self, name: str) -> Sequence[FieldValue]:
        try:
            return [s.fields[name] for s in self._samples]
        except KeyError:
            raise AttributeError(f"some samples have no field '{name}'")

    def to_array(self, vocab: Optional['Vocab'] = None) -> 'BatchArray':
        # TODO fix docstring
        """Convert the batch into numpy array.

        Returns:
            A :class:`BatchArray` object that has attribute names matching those of
            the field names in ``samples``. The value of such attribute is an array
            whose first dimension corresponds to the batch size as returned by
            ``__len__``.
        """
        common: Set[FieldName] = set()
        for s in self._samples:
            if common:
                common.intersection_update(s.fields)
            else:
                common = set(s.fields)

        if not common:
            raise RuntimeError('some samples have no common field names with the others')
        assert self._samples  # if `common` isn't empty, neither is `samples`

        arr = BatchArray()
        for name in common:
            if type(getattr(self._samples[0], name)) == str and vocab is not None:
                stoi = vocab.get(name)
                if stoi is not None:
                    vs = [stoi[v] for v in getattr(self, name)]
                else:
                    vs = getattr(self, name)
            else:
                vs = getattr(self, name)
            setattr(arr, name, np.array(vs))
        return arr


Vocab = Mapping[FieldName, Mapping[str, int]]


class BatchArray:
    pass
