text2array
==========

*Convert your NLP text dataset to arrays!*

**text2array** helps you process your NLP text dataset into Numpy ndarray objects that are
ready to use for e.g. neural network inputs. **text2array** handles data shuffling,
batching, padding, converting into arrays. Say goodbye to these tedious works!

Installation
------------

**text2array** requires at least Python 3.6 and can be installed via pip::

    pip install text2array

Usage overview
--------------

.. code-block:: python

    >>> from text2array import Dataset, Vocab
    >>>
    >>> samples = [
    ...   {'ws': ['john', 'talks']},
    ...   {'ws': ['john', 'loves', 'mary']},
    ...   {'ws': ['mary']}
    ... ]
    >>>
    >>> # Create a Dataset
    >>> dataset = Dataset(samples)
    >>> len(dataset)
    3
    >>> dataset[1]
    {'ws': ['john', 'loves', 'mary']}
    >>>
    >>> # Create a Vocab
    >>> vocab = Vocab.from_samples(dataset)
    >>> list(vocab['ws'])
    ['<pad>', '<unk>', 'john', 'mary']
    >>> # 'talks' and 'loves' are out-of-vocabulary because they occur only once
    >>> vocab['ws']['john']
    2
    >>> vocab['ws']['talks']  # unknown word is mapped to '<unk>'
    1
    >>>
    >>> # Applying vocab to the dataset
    >>> list(dataset)
    [{'ws': ['john', 'talks']}, {'ws': ['john', 'loves', 'mary']}, {'ws': ['mary']}]
    >>> dataset.apply_vocab(vocab)
    >>> list(dataset)
    [{'ws': [2, 1]}, {'ws': [2, 1, 3]}, {'ws': [3]}]
    >>>
    >>> # Shuffle, create batches of size 2, convert to array
    >>> batches = dataset.shuffle().batch(2)
    >>> batch = next(batches)
    >>> arr = batch.to_array()
    >>> arr['ws']
    array([[2, 1],
           [3, 0]])
    >>> batch = next(batches)
    >>> arr = batch.to_array()
    >>> arr['ws']
    array([[2, 1, 3]])

Detailed tutorial
-----------------

Sample
++++++

``Sample`` is just a ``Mapping[FieldName, FieldValue]``, where ``FieldName = str`` and
``FieldValue = Union[float, int, Sequence['FieldValue']``. It is easiest to use a ``dict``
to represent a sample, but you can essentially use any object you like as long as it
implements ``Mapping[FieldName, FieldValue]`` (which can be ensured by subclassing from
this type).

Dataset
+++++++

There are actually 2 classes for a dataset. ``Dataset`` is what you'd use normally. It
implements ``Sequence[Sample]`` so it requires all the samples to fit in the memory. To
create a ``Dataset`` object, pass an object of type ``Sequence[Sample]`` as an argument.

.. code-block:: python

    >>> from text2array import Dataset
    >>> samples = [
    ...   {'ws': ['john', 'talks']},
    ...   {'ws': ['john', 'loves', 'mary']},
    ...   {'ws': ['mary']}
    ... ]
    >>>
    >>> # Create a Dataset
    >>> dataset = Dataset(samples)
    >>> len(dataset)
    3
    >>> dataset[1]
    {'ws': ['john', 'loves', 'mary']}

If the samples can't fit in the memory, use ``StreamDataset`` instead. It implements
``Iterable[Sample]`` and streams the samples one by one, only when iterated over. To
instantiate, pass an ``Iterable[Sample]`` object.

.. code-block:: python

    >>> from text2array import StreamDataset
    >>> samples = [
    ...   {'ws': ['john', 'talks']},
    ...   {'ws': ['john', 'loves', 'mary']},
    ...   {'ws': ['mary']}
    ... ]
    >>> class Stream:
    ...   def __init__(self, seq):
    ...     self.seq = seq
    ...   def __iter__(self):
    ...     return iter(self.seq)
    ...
    >>> dataset = StreamDataset(Stream(samples))  # simulate a stream of samples
    >>> list(dataset)
    [{'ws': ['john', 'talks']}, {'ws': ['john', 'loves', 'mary']}, {'ws': ['mary']}]

Note that because ``StreamDataset`` is an iterable, you can't ask for its length nor access
by index, but it can be iterated over.

Shuffling dataset
^^^^^^^^^^^^^^^^^

``StreamDataset`` cannot be shuffled because shuffling requires all the elements to be
accessible by index. So, only ``Dataset`` can be shuffled. There are 2 ways to shuffle.
First, using ``shuffle`` method, which shuffles the dataset randomly without any
constraints. Second, using ``shuffle_by`` which accepts a ``Callable[[Sample], int]``
and use that to shuffle by performing a noisy sorting.

.. code-block:: python

    >>> from text2array import Dataset
    >>> samples = [
    ...   {'ws': ['john', 'talks']},
    ...   {'ws': ['john', 'loves', 'mary']},
    ...   {'ws': ['mary']}
    ... ]
    >>> dataset = Dataset(samples)
    >>> dataset.shuffle_by(lambda s: len(s['ws']))

The example above shuffles the dataset but also tries to keep samples with similar lengths
closer. This is useful for NLP where we want to shuffle but also minimize padding in each
batch. If a very short sample ends up in the same batch as a very long one, there would be
a lot of wasted entries for padding. Sorting noisily by length can help mitigate this issue.
This approach is inspired by `AllenNLP <https://github.com/allenai/allennlp>`_. Note that
both ``shuffle`` and ``shuffle_by`` returns the dataset object itself so method chaining
is possible.

Batching dataset
^^^^^^^^^^^^^^^^

To split up a dataset into batches, use the ``batch`` method, which takes the batch size
as an argument.

.. code-block:: python

    >>> from text2array import Dataset
    >>> samples = [
    ...   {'ws': ['john', 'talks']},
    ...   {'ws': ['john', 'loves', 'mary']},
    ...   {'ws': ['mary']}
    ... ]
    >>> dataset = Dataset(samples)
    >>> for batch in dataset.batch(2):
    ...   print('batch:', list(batch))
    ...
    batch: [{'ws': ['john', 'talks']}, {'ws': ['john', 'loves', 'mary']}]
    batch: [{'ws': ['mary']}]

The method returns an ``Iterator[Batch]`` object so it can be iterated only once. If you want
the batches to have exactly the same size, i.e. dropping the last one if it's smaller than
batch size, use ``batch_exactly`` instead. The two methods are also available for
``StreamDataset``. Note that before batching, you might want to map all those strings
into integer IDs first, which is explained in the next section.

Applying vocabulary
^^^^^^^^^^^^^^^^^^^

A vocabulary should implement ``Mapping[FieldName, Mapping[FieldValue, FieldValue]]``.
Then, call ``apply_vocab`` method with the vocabulary as an argument. This is best
explained with an example.

.. code-block:: python

    >>> from pprint import pprint
    >>> from text2array import Dataset
    >>> samples = [
    ...   {'ws': ['john', 'talks'], 'i': 10, 'label': 'pos'},
    ...   {'ws': ['john', 'loves', 'mary'], 'i': 20, 'label': 'pos'},
    ...   {'ws': ['mary'], 'i': 30, 'label': 'neg'}
    ... ]
    >>> dataset = Dataset(samples)
    >>> vocab = {
    ...   'ws': {'john': 0, 'talks': 1, 'loves': 2, 'mary': 3},
    ...   'i': {10: 5, 20: 10, 30: 15}
    ... }
    >>> dataset.apply_vocab(vocab)
    >>> pprint(list(dataset))
    [{'i': 5, 'label': 'pos', 'ws': [0, 1]},
     {'i': 10, 'label': 'pos', 'ws': [0, 2, 3]},
     {'i': 15, 'label': 'neg', 'ws': [3]}]

Note that the vocabulary is only applied to fields whose name is contained in the
vocabulary. Although not shown above, the vocabulary application still works even if
the field value is a deeply nested sequence. Note that ``apply_vocab`` is available
for ``StreamDataset`` as well.

Vocabulary
++++++++++

Creating a vocabulary object from scratch is tedious. So, it's common to learn the vocabulary
from a dataset. The ``Vocab`` class can be used for this purpose.

.. code-block:: python

    >>> samples = [
    ...   {'ws': ['john', 'talks'], 'i': 10, 'label': 'pos'},
    ...   {'ws': ['john', 'loves', 'mary'], 'i': 20, 'label': 'pos'},
    ...   {'ws': ['mary'], 'i': 30, 'label': 'neg'}
    ... ]
    >>> vocab = Vocab.from_samples(samples)
    >>> vocab.keys()
    dict_keys(['ws', 'label'])
    >>> dict(vocab['ws'])
    {'<pad>': 0, '<unk>': 1, 'john': 2, 'mary': 3}
    >>> dict(vocab['label'])
    {'<unk>': 0, 'pos': 1}
    >>> vocab['ws']['john'], vocab['ws']['talks']
    (2, 1)

There are several things to note:

#. Vocabularies are only created for fields which contain ``str`` values.
#. Words that occur only once are not included in the vocabulary.
#. Non-sequence fields do not have a padding token in the vocabulary.
#. Out-of-vocabulary words are assigned a single ID for unknown words.

``Vocab.from_samples`` actually accepts an ``Iterable[Sample]``, which means a ``Dataset``
or a ``StreamDataset`` can be passed as well. See the docstring to see other arguments
that it accepts to customize vocabulary creation.

Batch
+++++

Both ``batch`` and ``batch_exactly`` methods return ``Iterator[Batch]`` where ``Batch``
implements ``Sequence[Sample]``. This is true even for ``StreamDataset``. So, although
all samples may not all fit in the memory, a batch of them should. Given a ``Batch``
object, it can be converted into Numpy's ndarray by ``to_array`` method. Note that normally
you'd want to apply the vocabulary beforehand to ensure all values contain only ints or floats.

.. code-block:: python

    >>> from text2array import Dataset
    >>> samples = [
    ...   {'ws': ['john', 'talks'], 'i': 10, 'label': 'pos'},
    ...   {'ws': ['john', 'loves', 'mary'], 'i': 20, 'label': 'pos'},
    ...   {'ws': ['mary'], 'i': 30, 'label': 'neg'}
    ... ]
    >>> dataset = Dataset(samples)
    >>> vocab = Vocab.from_samples(dataset)
    >>> dict(vocab['ws'])
    {'<pad>': 0, '<unk>': 1, 'john': 2, 'mary': 3}
    >>> dict(vocab['label'])
    {'<unk>': 0, 'pos': 1}
    >>> dataset.apply_vocab(vocab)
    >>> batches = dataset.batch(2)
    >>> batch = next(batches)
    >>> arr = batch.to_array()
    >>> arr.keys()
    dict_keys(['ws', 'i', 'label'])
    >>> arr['ws']
    array([[2, 1, 0],
           [2, 1, 3]])
    >>> arr['i']
    array([10, 20])
    >>> arr['label']
    array([1, 1])

Note that ``to_array`` returns a ``Mapping[FieldName, np.ndarray]`` object, and sequential
fields are automatically padded.
