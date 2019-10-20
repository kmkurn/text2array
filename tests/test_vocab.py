from typing import Iterable, MutableMapping

from tqdm import tqdm
import pytest

from text2array import StringStore, Vocab


class TestFromSamples():
    @staticmethod
    def from_samples(ss, **kwargs):
        return Vocab.from_samples(ss, pbar=tqdm(disable=True), **kwargs)

    def test_ok(self):
        ss = [{'w': 'c'}, {'w': 'b'}, {'w': 'a'}, {'w': 'b'}, {'w': 'c'}, {'w': 'c'}]
        vocab = self.from_samples(ss)

        assert isinstance(vocab, MutableMapping)
        assert len(vocab) == 1
        assert list(vocab) == ['w']

        assert isinstance(vocab['w'], StringStore)
        assert vocab['w'].default == '<unk>'
        assert list(vocab['w']) == '<unk> c b a'.split()

        vocab['ws'] = StringStore()
        assert set(vocab) == {'w', 'ws'}
        del vocab['ws']
        assert list(vocab) == ['w']

    def test_has_vocab_for_all_str_fields(self):
        ss = [{'w': 'b', 't': 'b'}, {'w': 'b', 't': 'b'}]
        vocab = self.from_samples(ss)
        assert vocab.get('w') is not None
        assert vocab.get('t') is not None

    def test_no_vocab_for_non_str(self):
        vocab = self.from_samples([{'i': 10}, {'i': 20}])
        with pytest.raises(KeyError) as exc:
            vocab['i']
        assert "no vocabulary found for field name 'i'" in str(exc.value)

    def test_seq(self):
        ss = [{'ws': ['a', 'c', 'c']}, {'ws': ['b', 'c']}, {'ws': ['b']}]
        vocab = self.from_samples(ss)
        assert list(vocab['ws']) == '<pad> <unk> c b a'.split()

    def test_seq_of_seq(self):
        ss = [{
            'cs': [['c', 'd'], ['a', 'd']]
        }, {
            'cs': [['c'], ['b'], ['b', 'd']]
        }, {
            'cs': [['d', 'c']]
        }]
        vocab = self.from_samples(ss)
        assert list(vocab['cs']) == '<pad> <unk> d c b a'.split()

    def test_empty_field_values(self):
        vocab = self.from_samples([{'w': []}])
        with pytest.raises(KeyError):
            vocab['w']

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
        vocab = self.from_samples(ss, options={'w': dict(min_count=3)})
        assert 'a' not in vocab['w']
        assert 'b' not in vocab['w']
        assert 'c' in vocab['w']
        assert 'a' in vocab['t']
        assert 'b' in vocab['t']
        assert 'c' in vocab['t']

    def test_no_unk(self):
        vocab = self.from_samples([{'w': 'a', 't': 'a'}], options={'w': dict(unk=None)})
        assert vocab['w'].default is None
        assert '<unk>' not in vocab['w']
        assert '<unk>' in vocab['t']

    def test_no_pad(self):
        vocab = self.from_samples([{'w': ['a'], 't': ['a']}], options={'w': dict(pad=None)})
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
        vocab = self.from_samples(ss, options={'w': dict(max_size=1)})
        assert 'a' not in vocab['w']
        assert 'b' not in vocab['w']
        assert 'c' in vocab['w']
        assert 'a' in vocab['t']
        assert 'b' in vocab['t']
        assert 'c' in vocab['t']

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
        vocab = self.from_samples(iter(ss))
        assert 'b' in vocab['ws']
        assert 'c' in vocab['ws']
        assert 'c' in vocab['w']


class TestStoi:
    def test_samples_to_indices(self):
        ss = [{
            'ws': ['a', 'c', 'c'],
            'i': 1
        }, {
            'ws': ['b', 'c'],
            'i': 2
        }, {
            'ws': ['b'],
            'i': 3
        }]
        vocab = Vocab({'ws': StringStore('abc')})
        ss_ = vocab.stoi(ss)
        assert isinstance(ss_, Iterable)
        assert list(ss_) == [{
            'ws': [0, 2, 2],
            'i': 1
        }, {
            'ws': [1, 2],
            'i': 2
        }, {
            'ws': [1],
            'i': 3
        }]

    def test_value_is_not_str(self):
        ss = [{'ws': [0, 1, 2]}]
        vocab = Vocab({'ws': StringStore('abc')})
        assert list(vocab.stoi(ss)) == ss


class TestItos:
    def test_samples_to_strings(self):
        ss = [{
            'ws': ['a', 'c', 'c'],
            'i': 1
        }, {
            'ws': ['b', 'c'],
            'i': 2
        }, {
            'ws': ['b'],
            'i': 3
        }]
        vocab = Vocab({'ws': StringStore('abc')})
        assert list(vocab.itos(vocab.stoi(ss))) == ss

    def test_value_is_str(self):
        ss = [{'ws': ['a', 'b', 'c']}]
        vocab = Vocab({'ws': StringStore('abc')})
        assert list(vocab.itos(ss)) == ss
