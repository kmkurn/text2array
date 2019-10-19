# Copyright 2019 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import UserList
from functools import reduce
from typing import Dict, List, Mapping, MutableSequence, Optional, Sequence, Union, cast

import numpy as np

from .samples import FieldName, FieldValue, Sample


class Batch(UserList, MutableSequence[Sample]):
    """A class to represent a single batch.

    Args:
        samples (~typing.Sequence[Sample]): Sequence of samples this batch
            should contain.
    """

    def __init__(self, samples: Optional[Sequence[Sample]] = None) -> None:
        # constructor required; see https://docs.python.org/3.6/library/collections.html#collections.UserList
        if samples is None:
            samples = []
        super().__init__(samples)

    def to_array(
            self,
            pad_with: Union[int, Mapping[FieldName, int]] = 0,
    ) -> Dict[FieldName, np.ndarray]:
        """Convert the batch into `~numpy.ndarray`.

        Args:
            pad_with: Pad sequential field values with this number. Can
                also be a mapping from field names to padding number for
                that field. Fields whose name is not in the mapping will
                be padded with zeros.

        Returns:
            A mapping from field names to arrays whose first dimension
            corresponds to the batch size as returned by `len`.
        """
        if not self:
            return {}

        field_names = self[0].keys()

        if isinstance(pad_with, int):
            pad_dict = {name: pad_with for name in field_names}
        else:
            pad_dict = cast(dict, pad_with)

        arr = {}
        for name in field_names:
            values = self._get_values(name)

            # Get max length for all depths, 1st elem is batch size
            try:
                maxlens = self._get_maxlens(values)
            except self._InconsistentDepthError:
                raise ValueError(f"field '{name}' has inconsistent nesting depth")

            # Get padding for all depths
            paddings = self._get_paddings(maxlens, pad_dict.get(name, 0))
            # Pad the values
            data = self._pad(values, maxlens, paddings, 0)

            arr[name] = np.array(data)

        return arr

    def _get_values(self, name: str) -> Sequence[FieldValue]:
        try:
            return [s[name] for s in self]
        except KeyError:
            raise KeyError(f"some samples have no field '{name}'")

    @classmethod
    def _get_maxlens(cls, values: Sequence[FieldValue]) -> List[int]:
        assert values

        # Base case
        if isinstance(values[0], str) or not isinstance(values[0], Sequence):
            return [len(values)]

        # Recursive case
        maxlenss = [cls._get_maxlens(x) for x in values]
        if not all(len(x) == len(maxlenss[0]) for x in maxlenss):
            raise cls._InconsistentDepthError

        maxlens = reduce(lambda ml1, ml2: [max(l1, l2) for l1, l2 in zip(ml1, ml2)], maxlenss)
        maxlens.insert(0, len(values))
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
            values: Sequence[FieldValue],
            maxlens: List[int],
            paddings: List[Union[int, List[int]]],
            depth: int,
    ) -> Sequence[FieldValue]:
        assert values
        assert len(maxlens) == len(paddings)
        assert depth < len(maxlens)

        # Base case
        if isinstance(values[0], str) or not isinstance(values[0], Sequence):
            values_ = list(values)
        # Recursive case
        else:
            values_ = [cls._pad(x, maxlens, paddings, depth + 1) for x in values]

        for _ in range(maxlens[depth] - len(values)):
            values_.append(paddings[depth])
        return values_

    class _InconsistentDepthError(Exception):
        pass
