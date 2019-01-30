from collections.abc import Mapping

import pytest

from text2array import Vocab


class TestFromSamples():
    def test_ok(self):
        ss = [{'w': 'c'}, {'w': 'b'}, {'w': 'a'}, {'w': 'b'}, {'w': 'c'}, {'w': 'c'}]
        vocab = Vocab.from_samples(ss)

        assert isinstance(vocab, Mapping)
        assert len(vocab) == 1
        assert list(vocab) == ['w']
        with pytest.raises(KeyError):
            vocab['ws']

        # TODO for non-sequential field, maybe no padding?
        itos = '<pad> <unk> c b'.split()
        assert isinstance(vocab['w'], Mapping)
        assert len(vocab['w']) == len(itos)
        assert list(vocab['w']) == itos
        for i, w in enumerate(itos):
            assert w in vocab['w']
            assert vocab['w'][w] == i

        assert 'foo' not in vocab['w']
        assert vocab['w']['foo'] == vocab['w']['<unk>']
        assert 'bar' not in vocab['w']
        assert vocab['w']['bar'] == vocab['w']['<unk>']

    def test_has_vocab_for_all_str_fields(self):
        ss = [{'w': 'b', 't': 'b'}, {'w': 'b', 't': 'b'}]
        vocab = Vocab.from_samples(ss)
        assert vocab.get('w') is not None
        assert vocab.get('t') is not None

    def test_no_vocab_for_non_str(self):
        vocab = Vocab.from_samples([{'i': 10}, {'i': 20}])
        with pytest.raises(KeyError) as exc:
            vocab['i']
        assert "no vocabulary found for field name 'i'" in str(exc.value)

    def test_seq(self):
        ss = [{'ws': ['a', 'c', 'c']}, {'ws': ['b', 'c']}, {'ws': ['b']}]
        vocab = Vocab.from_samples(ss)
        assert list(vocab['ws']) == '<pad> <unk> c b'.split()

    def test_seq_of_seq(self):
        ss = [{
            'cs': [['c', 'd'], ['a', 'd']]
        }, {
            'cs': [['c'], ['b'], ['b', 'd']]
        }, {
            'cs': [['d', 'c']]
        }]
        vocab = Vocab.from_samples(ss)
        assert list(vocab['cs']) == '<pad> <unk> d c b'.split()

    def test_empty_samples(self):
        vocab = Vocab.from_samples([])
        assert not vocab

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
        vocab = Vocab.from_samples(ss, options={'w': dict(min_count=3)})
        assert 'b' not in vocab['w']
        assert 'b' in vocab['t']

    def test_no_unk(self):
        vocab = Vocab.from_samples([{'w': 'a', 't': 'a'}], options={'w': dict(unk=None)})
        assert '<unk>' not in vocab['w']
        assert '<unk>' in vocab['t']
        with pytest.raises(KeyError) as exc:
            vocab['w']['foo']
        assert "'foo' not found in vocabulary" in str(exc.value)

    def test_no_pad(self):
        vocab = Vocab.from_samples([{'w': 'a', 't': 'a'}], options={'w': dict(pad=None)})
        assert '<pad>' not in vocab['w']
        assert '<pad>' in vocab['t']

    def test_max_size(self):
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
        vocab = Vocab.from_samples(ss, options={'w': dict(max_size=1)})
        assert 'b' not in vocab['w']
        assert 'b' in vocab['t']

    def test_iterator_is_passed(self):
        ss = [{
            'ws': ['b', 'c'],
            'w': 'c'
        }, {
            'ws': ['c', 'b'],
            'w': 'c'
        }, {
            'ws': ['c'],
            'w': 'c'
        }]
        vocab = Vocab.from_samples(iter(ss))
        assert 'b' in vocab['ws']
        assert 'c' in vocab['ws']
        assert 'c' in vocab['w']
