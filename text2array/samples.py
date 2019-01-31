from typing import Mapping, Sequence, Union

# TODO remove these "type ignore" once mypy supports recursive types
# see: https://github.com/python/mypy/issues/731
FieldName = str
FieldValue = Union[float, int, str, Sequence['FieldValue']]  # type: ignore
Sample = Mapping[FieldName, FieldValue]  # type: ignore
