Installation
============

**text2array** requires at least Python 3.6 and can be installed via pip::

    $ pip install text2array

Overview
========

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
