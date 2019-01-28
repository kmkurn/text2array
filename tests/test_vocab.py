from collections.abc import Mapping, Sequence

import pytest

from text2array import Dataset, Vocab


def test_vocab_from_dataset():
    ss = [{
        'i': 1,
        'w': 'three',
        't': 'three',
    }, {
        'i': 2,
        'w': 'two',
        't': 'two',
    }, {
        'i': 3,
        'w': 'one',
        't': 'one',
    }, {
        'i': 4,
        'w': 'two',
        't': 'two',
    }, {
        'i': 5,
        'w': 'three',
        't': 'three',
    }, {
        'i': 6,
        'w': 'three',
        't': 'three',
    }]
    vocab = Vocab.from_dataset(Dataset(ss))
    itos = '<pad> <unk> three two'.split()

    assert isinstance(vocab.of('w'), Sequence)
    assert len(vocab.of('w')) == len(itos)
    for i in range(len(itos)):
        vocab.of('w')[i] == itos[i]

    assert isinstance(vocab.of('w').stoi, Mapping)
    assert len(vocab.of('w').stoi) == len(vocab.of('w'))
    assert set(vocab.of('w').stoi) == set(vocab.of('w'))
    for i, s in enumerate(vocab.of('w')):
        assert vocab.of('w').stoi[s] == i
    assert vocab.of('w').stoi['foo'] == vocab.of('w').stoi['<unk>']
    assert vocab.of('w').stoi['bar'] == vocab.of('w').stoi['<unk>']

    assert isinstance(vocab.of('t'), Sequence)
    assert isinstance(vocab.of('t').stoi, Mapping)
    with pytest.raises(RuntimeError) as exc:
        vocab.of('i')
    assert "no vocabulary found for field name 'i'" in str(exc.value)
