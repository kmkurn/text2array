Tutorial
========

.. currentmodule:: text2array

Sample
------

``Sample`` is just a ``Mapping[FieldName, FieldValue]``, where ``FieldName = str`` and
``FieldValue = Union[float, int, str, Sequence['FieldValue']]``. It is easiest to use a
`dict` to represent a sample, but you can essentially use any object you like as long
as it implements ``Mapping[FieldName, FieldValue]`` (which can be ensured by subclassing
from this type).

Vocabulary
----------

After creating samples, we need to build a vocabulary. A vocabulary holds the
str-to-int (or int-to-str) mapping for each field. Building a vocabulary from scratch
is tedious. So, it's common to build the vocabulary from the given samples. The `Vocab`
class can be used for this purpose.

.. code-block:: python

    >>> from text2array import Vocab
    >>> samples = [
    ...   {'ws': ['john', 'talks'], 'i': 10, 'label': 'pos'},
    ...   {'ws': ['john', 'loves', 'mary'], 'i': 20, 'label': 'pos'},
    ...   {'ws': ['mary'], 'i': 30, 'label': 'neg'}
    ... ]
    >>> vocab = Vocab.from_samples(samples, options={'ws': dict(min_count=2)})
    >>> vocab.keys()
    dict_keys(['ws', 'label'])
    >>> dict(vocab['ws'])
    {'<pad>': 0, '<unk>': 1, 'john': 2, 'mary': 3}
    >>> dict(vocab['label'])
    {'<unk>': 0, 'pos': 1}
    >>> 'john' in vocab['ws'], 'talks' in vocab['ws']
    (True, False)
    >>> vocab['ws']['john'], vocab['ws']['talks']
    (2, 1)

There are several things to note:

#. Vocabularies are only created for fields which contain `str` values.
#. Non-sequence fields do not have a padding token in the vocabulary.
#. Out-of-vocabulary words are assigned a single ID for unknown words.

`Vocab.from_samples` accepts an ``Iterable[Sample]``, which means it does not care
if all the samples fit in the memory. You can pass an iterable that streams the
samples from disk if you like. See the documentation to see other arguments that
it accepts to customize vocabulary creation.

Applying vocabulary
^^^^^^^^^^^^^^^^^^^

Once a vocabulary is built, we need to apply it to our samples. Applying a vocabulary
means mapping all field values according to the mapping specified by the vocabulary.
Continuing from the previous example:

.. code-block:: python

   >>> for s in vocab.apply_to(samples):
   ...   print(s)
   ...
   {'ws': [2, 1], 'i': 10, 'label': 1}
   {'ws': [2, 1, 3], 'i': 20, 'label': 1}
   {'ws': [3], 'i': 30, 'label': 0}

Iterators
---------

There are two iterators provided in this library: `ShuffleIterator` and `BatchIterator`.
They are used to perform shuffling and batching respectively.

Shuffling
^^^^^^^^^

To shuffle, we need to pass a ``Sequence[Sample]`` to `ShuffleIterator`. We can easily
convert an ``Iterable[Sample]`` to ``Sequence[Sample]`` by converting it to a `list`.

.. code-block:: python

   >>> samples = list(vocab.apply_to(samples))  # now we have a sequence
   >>> from text2array import ShuffleIterator
   >>> iterator = ShuffleIterator(samples, key=lambda s: len(s['ws']))
   >>> len(iterator)
   3
   >>> for s in iterator:
   ...   print(s)
   ...
   {'ws': [2, 1], 'i': 10, 'label': 1}
   {'ws': [3], 'i': 30, 'label': 0}
   {'ws': [2, 1, 3], 'i': 20, 'label': 1}

The iterator above shuffles the samples but also tries to keep samples with similar lengths
closer. This is useful for NLP where we want to shuffle but also minimize padding in each
batch. If a very short sample ends up in the same batch as a very long one, there would be
a lot of wasted entries for padding. Sorting noisily by length can help mitigate this issue.
This approach is inspired by `AllenNLP <https://github.com/allenai/allennlp>`_. Note that
(1) ``iterator`` is an ``Iterable[Sample]`` and (2) shuffling is done whenever ``iterator``
is iterated over.

Batching
^^^^^^^^

To do batching, pass an ``Iterable[Sample]`` to `BatchIterator`. Since `ShuffleIterator`
is an ``Iterable[Sample]``, it is thus possible passing it to perform shuffling and
batching sequentially on each iteration.

.. code-block:: python

   >>> from text2array import BatchIterator, ShuffleIterator
   >>> iterator = ShuffleIterator(samples, key=lambda s: len(s['ws']))
   >>> iterator = BatchIterator(iterator, batch_size=2)
   >>> len(iterator)
   2
   >>> for s in iterator:
   ...   print(s)
   ...
   <text2array.batches.Batch object at 0x10ddbc358>
   <text2array.batches.Batch object at 0x10ddbc390>

When iterated over, `BatchIterator` produces `Batch` objects, which will be explained next.

Batch
-----

A `Batch` is just a ``Sequence[Sample]``, but it has a `~Batch.to_array` method to convert
samples in that batch to an array. The nice thing is sequential fields are automatically
padded, **no matter how deeply nested they are**.

.. code-block:: python

   >>> from pprint import pprint
   >>> samples = [
   ...   {'ws': ['john', 'talks'], 'cs': [list('john'), list('talks')]},
   ...   {'ws': ['john', 'loves', 'mary'], 'cs': [list('john'), list('loves'), list('mary')]},
   ...   {'ws': ['mary'], 'cs': [list('mary')]}
   ... ]
   >>> vocab = Vocab.from_samples(samples, options=dict(ws=dict(min_count=2)))
   >>> samples = list(vocab.apply_to(samples))
   >>> dict(vocab['ws'])
   {'<pad>': 0, '<unk>': 1, 'john': 2, 'mary': 3}
   >>> pprint(dict(vocab['cs']))
   {'<pad>': 0,
   '<unk>': 1,
   'a': 3,
   'h': 5,
   'j': 4,
   'l': 7,
   'm': 9,
   'n': 6,
   'o': 2,
   'r': 10,
   's': 8,
   'y': 11}
   >>> iterator = BatchIterator(samples, batch_size=2)
   >>> it = iter(iterator)
   >>> batch = next(it)
   >>> arr = batch.to_array()
   >>> arr['ws']
   array([[2, 1, 0],
          [2, 1, 3]])
   >>> arr['cs']
   array([[[ 4,  2,  5,  6,  0],
           [ 1,  3,  7,  1,  8],
           [ 0,  0,  0,  0,  0]],

          [[ 4,  2,  5,  6,  0],
           [ 7,  2,  1,  1,  8],
           [ 9,  3, 10, 11,  0]]])

Note how `Batch.to_array` returns a ``Mapping[FieldName, np.ndarray]`` object, and
sequential fields are automatically padded.
