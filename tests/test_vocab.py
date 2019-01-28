from collections.abc import Mapping, Sequence

import pytest

from text2array import Vocab


class TestFromSamples():
    def test_ok(self):
        ss = [{'w': 'c'}, {'w': 'b'}, {'w': 'a'}, {'w': 'b'}, {'w': 'c'}, {'w': 'c'}]
        vocab = Vocab.from_samples(ss)

        assert isinstance(vocab, Mapping)
        assert len(vocab) == 1
        assert list(vocab) == ['w']

        itos = '<pad> <unk> c b'.split()
        assert isinstance(vocab['w'], Sequence)
        assert len(vocab['w']) == len(itos)
        for i in range(len(itos)):
            assert vocab['w'][i] == itos[i]

        assert isinstance(vocab['w'].stoi, Mapping)
        assert len(vocab['w'].stoi) == len(vocab['w'])
        assert set(vocab['w'].stoi) == set(vocab['w'])
        for i, s in enumerate(vocab['w']):
            assert vocab['w'].stoi[s] == i

        assert vocab['w'].stoi['foo'] == vocab['w'].stoi['<unk>']
        assert vocab['w'].stoi['bar'] == vocab['w'].stoi['<unk>']

    def test_has_vocab_for_all_str_fields(self):
        ss = [{'w': 'b', 't': 'b'}, {'w': 'b', 't': 'b'}]
        vocab = Vocab.from_samples(ss)
        assert isinstance(vocab['t'], Sequence)
        assert isinstance(vocab['t'].stoi, Mapping)

    def test_no_vocab_for_non_str(self):
        vocab = Vocab.from_samples([{'i': 10}, {'i': 20}])
        with pytest.raises(RuntimeError) as exc:
            vocab['i']
        assert "no vocabulary found for field name 'i'" in str(exc.value)

    def test_seq(self):
        ss = [{'ws': ['a', 'c', 'c']}, {'ws': ['b', 'c']}, {'ws': ['b']}]
        vocab = Vocab.from_samples(ss)

        itos = '<pad> <unk> c b'.split()

        assert isinstance(vocab['ws'], Sequence)
        assert len(vocab['ws']) == len(itos)
        for i in range(len(itos)):
            assert vocab['ws'][i] == itos[i]

        assert isinstance(vocab['ws'].stoi, Mapping)
        assert len(vocab['ws'].stoi) == len(vocab['ws'])
        assert set(vocab['ws'].stoi) == set(vocab['ws'])
        for i, s in enumerate(vocab['ws']):
            assert vocab['ws'].stoi[s] == i

    def test_seq_of_seq(self):
        ss = [{
            'cs': [['c', 'd'], ['a', 'd']]
        }, {
            'cs': [['c'], ['b'], ['b', 'd']]
        }, {
            'cs': [['d', 'c']]
        }]
        vocab = Vocab.from_samples(ss)

        itos = '<pad> <unk> d c b'.split()

        assert isinstance(vocab['cs'], Sequence)
        assert len(vocab['cs']) == len(itos)
        for i in range(len(itos)):
            assert vocab['cs'][i] == itos[i]

        assert isinstance(vocab['cs'].stoi, Mapping)
        assert len(vocab['cs'].stoi) == len(vocab['cs'])
        assert set(vocab['cs'].stoi) == set(vocab['cs'])
        for i, s in enumerate(vocab['cs']):
            assert vocab['cs'].stoi[s] == i

    def test_empty_samples(self):
        vocab = Vocab.from_samples([])
        assert len(vocab) == 0

    def test_empty_field_values(self):
        with pytest.raises(ValueError) as exc:
            Vocab.from_samples([{'w': []}])
        assert 'field values must not be an empty sequence' in str(exc.value)

    def test_min_count(self):
        ss = [{
            'w': 'c',
            't': 'c'
        }, {
            'w': 'b',
            't': 'b'
        }, {
            'w': 'a',
            't': 'a'
        }, {
            'w': 'b',
            't': 'b'
        }, {
            'w': 'c',
            't': 'c'
        }, {
            'w': 'c',
            't': 'c'
        }]
        vocab = Vocab.from_samples(ss, ve_kwargs={'w': dict(min_count=3)})
        assert list(vocab['w']) == ['<pad>', '<unk>', 'c']
        assert list(vocab['t']) == ['<pad>', '<unk>', 'c', 'b']
