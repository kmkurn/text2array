from typing import Mapping
import abc

FieldName = str
FieldValue = int


class SampleABC(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def fields(self) -> Mapping[FieldName, FieldValue]:
        pass
