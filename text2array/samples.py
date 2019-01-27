from typing import Mapping, Sequence, Union

FieldName = str
FieldValue = Union[float, int, Sequence[float], Sequence[int]]
Sample = Mapping[FieldName, FieldValue]
