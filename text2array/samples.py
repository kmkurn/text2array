from typing import Mapping, Union
import abc

FieldName = str
FieldValue = Union[float, int]


class SampleABC(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def fields(self) -> Mapping[FieldName, FieldValue]:
        pass
