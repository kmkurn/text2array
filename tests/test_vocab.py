from collections.abc import Mapping, Sequence

import pytest

from text2array import Dataset, Vocab


class TestFromDataset():
    def test_ok(self):
        dat = Dataset([{'w': 'c'}, {'w': 'b'}, {'w': 'a'}, {'w': 'b'}, {'w': 'c'}, {'w': 'c'}])
        vocab = Vocab.from_dataset(dat)
        itos = '<pad> <unk> c b'.split()

        assert isinstance(vocab.of('w'), Sequence)
        assert len(vocab.of('w')) == len(itos)
        for i in range(len(itos)):
            assert vocab.of('w')[i] == itos[i]

        assert isinstance(vocab.of('w').stoi, Mapping)
        assert len(vocab.of('w').stoi) == len(vocab.of('w'))
        assert set(vocab.of('w').stoi) == set(vocab.of('w'))
        for i, s in enumerate(vocab.of('w')):
            assert vocab.of('w').stoi[s] == i

        assert vocab.of('w').stoi['foo'] == vocab.of('w').stoi['<unk>']
        assert vocab.of('w').stoi['bar'] == vocab.of('w').stoi['<unk>']

    def test_has_vocab_for_all_str_fields(self):
        dat = Dataset([{'w': 'b', 't': 'b'}, {'w': 'b', 't': 'b'}])
        vocab = Vocab.from_dataset(dat)
        assert isinstance(vocab.of('t'), Sequence)
        assert isinstance(vocab.of('t').stoi, Mapping)

    def test_no_vocab_for_non_str(self):
        vocab = Vocab.from_dataset(Dataset([{'i': 10}, {'i': 20}]))
        with pytest.raises(RuntimeError) as exc:
            vocab.of('i')
        assert "no vocabulary found for field name 'i'" in str(exc.value)
