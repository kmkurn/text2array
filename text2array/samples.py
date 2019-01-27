from typing import Mapping, Union
import abc

FieldName = str
FieldValue = Union[str, float, int]


# TODO properly ignore abstractmethod from coverage
class Sample(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def fields(self) -> Mapping[FieldName, FieldValue]:
        pass
