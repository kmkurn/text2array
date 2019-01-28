from collections.abc import Mapping, Sequence

from text2array import Dataset, Vocab


def test_vocab_from_dataset(samples):
    ss = [{
        'w': 'three'
    }, {
        'w': 'two'
    }, {
        'w': 'one'
    }, {
        'w': 'two'
    }, {
        'w': 'three'
    }, {
        'w': 'three'
    }]
    for s, s_ in zip(ss, samples):
        s.update(s_)

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
