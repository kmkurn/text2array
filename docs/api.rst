API reference
=============

Type aliases
------------

* ``Sample = Mapping[FieldName, FieldValue]``
* ``FieldName = str``
* ``FieldValue = Union[float, int, str, Sequence[FieldValue]``

Classes
-------

.. autoclass:: text2array.datasets.DatasetABC

.. currentmodule:: text2array

.. autoclass:: Dataset

.. autoclass:: StreamDataset

.. autoclass:: BatchIterator

.. autoclass:: ShuffleIterator

.. autoclass:: Batch

.. autoclass:: Vocab
