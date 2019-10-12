from typing import Iterable, Mapping

from tqdm import tqdm
import pytest

from text2array import Vocab


class TestFromSamples():
    @staticmethod
    def from_samples(ss, **kwargs):
        return Vocab.from_samples(ss, pbar=tqdm(disable=True), **kwargs)

    def test_ok(self):
        ss = [{'w': 'c'}, {'w': 'b'}, {'w': 'a'}, {'w': 'b'}, {'w': 'c'}, {'w': 'c'}]
        vocab = self.from_samples(ss)

        assert isinstance(vocab, Mapping)
        assert len(vocab) == 1
        assert list(vocab) == ['w']
        with pytest.raises(KeyError):
            vocab['ws']

        itos = '<unk> c b'.split()
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
        assert list(vocab['ws']) == '<pad> <unk> c b'.split()

    def test_seq_of_seq(self):
        ss = [{
            'cs': [['c', 'd'], ['a', 'd']]
        }, {
            'cs': [['c'], ['b'], ['b', 'd']]
        }, {
            'cs': [['d', 'c']]
        }]
        vocab = self.from_samples(ss)
        assert list(vocab['cs']) == '<pad> <unk> d c b'.split()

    def test_empty_samples(self):
        vocab = self.from_samples([])
        assert not vocab

    def test_empty_field_values(self):
        with pytest.raises(ValueError) as exc:
            self.from_samples([{'w': []}])
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
        vocab = self.from_samples(ss, options={'w': dict(min_count=3)})
        assert 'b' not in vocab['w']
        assert 'b' in vocab['t']

    def test_no_unk(self):
        vocab = self.from_samples([{'w': 'a', 't': 'a'}], options={'w': dict(unk=None)})
        assert '<unk>' not in vocab['w']
        assert '<unk>' in vocab['t']
        with pytest.raises(KeyError) as exc:
            vocab['w']['foo']
        assert "'foo' not found in vocabulary" in str(exc.value)

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
        vocab = self.from_samples(iter(ss))
        assert 'b' in vocab['ws']
        assert 'c' in vocab['ws']
        assert 'c' in vocab['w']


def test_apply_to_samples():
    ss = [{'ws': ['a', 'c', 'c'], 'i': 1}, {'ws': ['b', 'c'], 'i': 2}, {'ws': ['b'], 'i': 3}]
    vocab = Vocab({'ws': {'a': 1, 'b': 2, 'c': 3}})
    ss_ = vocab.apply_to(ss)
    assert isinstance(ss_, Iterable)
    assert list(ss_) == [{'ws': [1, 3, 3], 'i': 1}, {'ws': [2, 3], 'i': 2}, {'ws': [2], 'i': 3}]


def test_apply_to_stream(stream_cls):
    ss = stream_cls([{'ws': ['a', 'c', 'c']}, {'ws': ['b', 'c']}, {'ws': ['b']}])
    vocab = Vocab({'ws': {'a': 1, 'b': 2, 'c': 3}})
    ss_ = vocab.apply_to(ss)
    assert isinstance(ss_, Iterable)
    assert list(ss_) == [{'ws': [1, 3, 3]}, {'ws': [2, 3]}, {'ws': [2]}]


def test_apply_to_value_not_found():
    ss = [{'ws': ['a']}]
    vocab = Vocab({'ws': {'b': 2}})
    with pytest.raises(KeyError) as exc:
        list(vocab.apply_to(ss))
    assert "value 'a' not found in vocab" in str(exc.value)


def test_eq_not_same():
    ss = [{'ws': ['b', 'c']}, {'ws': ['c', 'c', 'b']}]
    vocab1 = TestFromSamples.from_samples(ss, options={'ws': {'pad': None, 'unk': None}})
    vocab2 = TestFromSamples.from_samples(ss, options={'ws': {'unk': None}})
    assert vocab1 != vocab2


def test_eq_with_dict():
    ss = [{'ws': ['b', 'c']}, {'ws': ['c', 'c', 'b']}]
    vocab1 = TestFromSamples.from_samples(ss, options={'ws': {'pad': None, 'unk': None}})
    vocab2 = Vocab({'ws': {'c': 0, 'b': 1}})
    assert vocab1 != vocab2
