from collections.abc import Mapping

import pytest

from text2array import Vocab


class TestFromSamples():
    # TODO think about what to test in each test case
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
        assert 'w' in vocab
        assert 't' in vocab

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
        assert list(vocab['w']) == '<pad> <unk> c'.split()
        assert list(vocab['t']) == '<pad> <unk> c b'.split()

    def test_no_unk(self):
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
        vocab = Vocab.from_samples(ss, options={'w': dict(unk=None)})
        assert list(vocab['w']) == '<pad> c b'.split()
        assert list(vocab['t']) == '<pad> <unk> c b'.split()
        with pytest.raises(KeyError) as exc:
            vocab['w']['foo']
        assert "'foo' not found in vocabulary" in str(exc.value)

    def test_no_pad(self):
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
        vocab = Vocab.from_samples(ss, options={'w': dict(pad=None)})
        assert list(vocab['w']) == '<unk> c b'.split()
        assert list(vocab['t']) == '<pad> <unk> c b'.split()

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
        assert list(vocab['w']) == '<pad> <unk> c'.split()
        assert list(vocab['t']) == '<pad> <unk> c b'.split()

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
