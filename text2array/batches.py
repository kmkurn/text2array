from typing import Sequence

from .samples import FieldValue, SampleABC


class Batch(Sequence[SampleABC]):
    """A class to represent a single batch.i

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
