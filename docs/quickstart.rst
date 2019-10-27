Installation
============

**text2array** requires at least Python 3.6 and can be installed via pip::

    $ pip install text2array

Overview
========

.. doctest:: quickstart

    >>> ### Import required modules ###
    >>> from random import Random
    >>> from text2array import BatchIterator, ShuffleIterator, Vocab
    >>>
    >>> ### Create samples ###
    >>> samples = [
    ...   {'ws': ['john', 'talks']},
    ...   {'ws': ['john', 'loves', 'mary']},
    ...   {'ws': ['mary']}
    ... ]
    >>>
    >>> ### Create a Vocab ###
    >>> vocab = Vocab.from_samples(samples, options={'ws': dict(min_count=2)})
    >>> vocab['ws']
    StringStore(['<pad>', '<unk>', 'john', 'mary'], default='<unk>')
    >>> # 'talks' and 'loves' are out-of-vocabulary because they occur only once
    >>> 'john' in vocab['ws']
    True
    >>> vocab['ws'].index('john')
    2
    >>> 'talks' in vocab['ws']
    False
    >>> vocab['ws'].index('talks')  # unknown word is mapped to '<unk>'
    1
    >>>
    >>> ### Applying vocab to samples ###
    >>> samples
    [{'ws': ['john', 'talks']}, {'ws': ['john', 'loves', 'mary']}, {'ws': ['mary']}]
    >>> samples = list(vocab.stoi(samples))
    >>> samples
    [{'ws': [2, 1]}, {'ws': [2, 1, 3]}, {'ws': [3]}]
    >>>
    >>> ### Shuffle, create batches of size 2, convert to array ###
    >>> iterator = BatchIterator(ShuffleIterator(samples, rng=Random(123)), batch_size=2)
    >>> len(iterator)
    2
    >>> for i, batch in enumerate(iterator):
    ...     print(f'Batch #{i+1}')
    ...     arr = batch.to_array()
    ...     print(repr(arr['ws']))
    ...
    Batch #1
    array([[3, 0, 0],
           [2, 1, 3]])
    Batch #2
    array([[2, 1]])
