text2array
==========

*Convert your NLP text dataset to arrays!*

.. image:: https://travis-ci.org/kmkurn/text2array.svg?branch=master
   :target: https://travis-ci.org/kmkurn/text2array

.. image:: https://coveralls.io/repos/github/kmkurn/text2array/badge.svg?branch=master
   :target: https://coveralls.io/github/kmkurn/text2array?branch=master

.. image:: https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg
   :target: http://spacemacs.org

**text2array** helps you process your NLP text dataset into Numpy ndarray objects that are
ready to use for e.g. neural network inputs. **text2array** handles data shuffling,
batching, padding, and converting into arrays. Say goodbye to these tedious works!

Installation
------------

**text2array** requires at least Python 3.6 and can be installed via pip::

    $ pip install text2array

Overview
--------

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
    >>> 'john' in vocab['ws']
    True
    >>> vocab['ws']['john']
    2
    >>> 'talks' in vocab['ws']
    False
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
    array([[3, 0, 0],
           [2, 1, 3]])
    >>> batch = next(batches)
    >>> arr = batch.to_array()
    >>> arr['ws']
    array([[2, 1]])

Tutorial
--------

Sample
++++++

``Sample`` is just a ``Mapping[FieldName, FieldValue]``, where ``FieldName = str`` and
``FieldValue = Union[float, int, str, Sequence['FieldValue']``. It is easiest to use a
``dict`` to represent a sample, but you can essentially use any object you like as long
as it implements ``Mapping[FieldName, FieldValue]`` (which can be ensured by subclassing
from this type).

Dataset
+++++++

There are actually 2 classes for a dataset. ``Dataset`` is what you would normally use. It
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

Because ``StreamDataset`` is an iterable, you can't ask for its length nor access
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
This approach is inspired by `AllenNLP <https://github.com/allenai/allennlp>`_. Both
``shuffle`` and ``shuffle_by`` returns the dataset object itself so method chaining
is possible. See the docstring for more details.

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
the batch size, use ``batch_exactly`` instead. The two methods are also available for
``StreamDataset``. Before batching, you might want to map all those strings
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
the field value is a deeply nested sequence. Method ``apply_vocab`` is available
for ``StreamDataset`` as well.

Vocabulary
++++++++++

Creating a vocabulary object from scratch is tedious. So, it's common to learn the vocabulary
from a dataset. The ``Vocab`` class can be used for this purpose.

.. code-block:: python

    >>> from text2array import Vocab
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
    >>> 'john' in vocab['ws'], 'talks' in vocab['ws']
    (True, False)
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
samples may not all fit in the memory, a batch of them should. Given a ``Batch``
object, it can be converted into Numpy's ndarray by ``to_array`` method. Normally,
you'd want to apply the vocabulary beforehand to ensure all values contain only ints or floats.

.. code-block:: python

    >>> from text2array import Dataset, Vocab
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
fields are automatically padded. One of the nice things is that the field can be deeply
nested and the padding just works!

.. code-block:: python

    >>> from pprint import pprint
    >>> from text2array import Dataset, Vocab
    >>> samples = [
    ...   {'ws': ['john', 'talks'], 'cs': [list('john'), list('talks')]},
    ...   {'ws': ['john', 'loves', 'mary'], 'cs': [list('john'), list('loves'), list('mary')]},
    ...   {'ws': ['mary'], 'cs': [list('mary')]}
    ... ]
    >>> dataset = Dataset(samples)
    >>> vocab = Vocab.from_samples(dataset)
    >>> dataset.apply_vocab(vocab)
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
    >>> batches = dat.batch(2)
    >>> batch = next(batches)
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

So, you can go crazy and have a field representing a document hierarchically as paragraphs,
sentences, words, and characters, and it will be padded correctly.

Contributing
------------

Pull requests are welcome! To start contributing, make sure to install all the dependencies.

::

    $ pip install -r requirements.txt

Next, setup the pre-commit hook.

::

    $ ln -s ../../pre-commit.sh .git/hooks/pre-commit

Tests and the linter can be run with ``pytest`` and ``flake8`` respectively. The latter also
runs ``mypy`` for type checking.

License
-------

MIT
