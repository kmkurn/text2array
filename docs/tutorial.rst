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

After creating samples, we need to build a vocabulary. A vocabulary holds an ordered set
of string values for each field. Building a vocabulary from scratch is tedious. So, it's
easier to build the vocabulary from some samples. The `Vocab` class can be used for
this purpose.

.. doctest:: tutorial

    >>> from text2array import Vocab
    >>> samples = [
    ...   {'ws': ['john', 'talks'], 'i': 10, 'label': 'pos'},
    ...   {'ws': ['john', 'loves', 'mary'], 'i': 20, 'label': 'pos'},
    ...   {'ws': ['mary'], 'i': 30, 'label': 'neg'}
    ... ]
    >>> vocab = Vocab.from_samples(samples, options={'ws': dict(min_count=2)})
    >>> list(vocab.keys())
    ['ws', 'label']
    >>> vocab['ws']
    StringStore(['<pad>', '<unk>', 'john', 'mary'], default='<unk>')
    >>> vocab['label']
    StringStore(['<unk>', 'pos', 'neg'], default='<unk>')
    >>> 'john' in vocab['ws'], 'talks' in vocab['ws']
    (True, False)
    >>> vocab['ws'].index('john'), vocab['ws'].index('talks')
    (2, 1)

There are several things to note:

#. Vocabularies are only created for fields which contain `str` values.
#. Non-sequence fields do not have a padding token in the vocabulary.
#. Out-of-vocabulary words are assigned a single ID for unknown words.

`Vocab.from_samples` accepts an ``Iterable[Sample]``, which means it does not care
if all the samples fit in the memory. You can pass an iterable that streams the
samples from disk if you like. See the documentation to see other arguments that
it accepts to customize vocabulary creation.

Converting strings in samples to integers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a vocabulary is built, we need convert strings in our samples with it. This
conversion means mapping all field values according to the vocabulary. Continuing
from the previous example:

.. doctest:: tutorial

   >>> for s in vocab.stoi(samples):
   ...   print(s)
   ...
   {'ws': [2, 1], 'i': 10, 'label': 1}
   {'ws': [2, 1, 3], 'i': 20, 'label': 1}
   {'ws': [3], 'i': 30, 'label': 2}

Iterators
---------

There are two iterators provided in this library: `ShuffleIterator` and `BatchIterator`.
They are used to perform shuffling and batching respectively.

Shuffling
^^^^^^^^^

To shuffle, we need to pass a ``Sequence[Sample]`` to `ShuffleIterator`. We can easily
convert an ``Iterable[Sample]`` to ``Sequence[Sample]`` by converting it to a `list`.

.. doctest:: tutorial

   >>> samples = list(vocab.stoi(samples))  # now we have a sequence
   >>> from random import Random
   >>> from text2array import ShuffleIterator
   >>> iterator = ShuffleIterator(samples, key=lambda s: len(s['ws']), rng=Random(1234))
   >>> len(iterator)
   3
   >>> for s in iterator:
   ...   print(s)
   ...
   {'ws': [3], 'i': 30, 'label': 2}
   {'ws': [2, 1, 3], 'i': 20, 'label': 1}
   {'ws': [2, 1], 'i': 10, 'label': 1}

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

.. doctest:: tutorial

   >>> from text2array import Batch, BatchIterator, ShuffleIterator
   >>> iterator = ShuffleIterator(samples, key=lambda s: len(s['ws']))
   >>> iterator = BatchIterator(iterator, batch_size=2)
   >>> len(iterator)
   2
   >>> for s in iterator:
   ...   assert isinstance(s, Batch)
   ...

When iterated over, `BatchIterator` produces `Batch` objects, which will be explained next.

Batch
-----

A `Batch` is just a ``MutableSequence[Sample]``, but it has a `~Batch.to_array` method to convert
samples in that batch to an array. The nice thing is sequential fields are automatically
padded, **no matter how deeply nested they are**.

.. doctest:: tutorial

   >>> samples = [
   ...   {'ws': ['john', 'talks'], 'cs': [list('john'), list('talks')]},
   ...   {'ws': ['john', 'loves', 'mary'], 'cs': [list('john'), list('loves'), list('mary')]},
   ...   {'ws': ['mary'], 'cs': [list('mary')]}
   ... ]
   >>> vocab = Vocab.from_samples(samples, options={'ws': dict(min_count=2), 'cs': dict(min_count=2)})
   >>> samples = list(vocab.stoi(samples))
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
   <BLANKLINE>
          [[ 4,  2,  5,  6,  0],
           [ 7,  2,  1,  1,  8],
           [ 9,  3, 10, 11,  0]]])

Note how `Batch.to_array` returns a ``Mapping[FieldName, np.ndarray]`` object, and
sequential fields are automatically padded.
