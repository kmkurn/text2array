from typing import Iterable, Iterator

import pytest

from text2array import Batch, StreamDataset, Vocab


def test_init(stream):
    dat = StreamDataset(stream)
    assert isinstance(dat, Iterable)
    assert list(dat) == list(stream)


def test_init_stream_non_iterable():
    with pytest.raises(TypeError) as exc:
        StreamDataset(5)
    assert '"stream" is not iterable' in str(exc.value)


def test_can_be_iterated_twice(stream_dataset):
    dat_lst1 = list(stream_dataset)
    dat_lst2 = list(stream_dataset)
    assert len(dat_lst1) == len(dat_lst2)
    assert len(dat_lst2) > 0


def test_batch(stream_cls):
    stream_dat = StreamDataset(stream_cls([{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}]))
    bs = stream_dat.batch(2)
    assert isinstance(bs, Iterator)
    bs_lst = list(bs)
    assert len(bs_lst) == 3
    assert all(isinstance(b, Batch) for b in bs_lst)
    dat = list(stream_dat)
    assert list(bs_lst[0]) == [dat[0], dat[1]]
    assert list(bs_lst[1]) == [dat[2], dat[3]]
    assert list(bs_lst[2]) == [dat[4]]


def test_batch_size_evenly_divides(stream_dataset):
    bs = stream_dataset.batch(1)
    dat = list(stream_dataset)
    bs_lst = list(bs)
    assert len(bs_lst) == len(dat)
    for i in range(len(bs_lst)):
        assert list(bs_lst[i]) == [dat[i]]


def test_batch_exactly(stream_cls):
    stream_dat = StreamDataset(stream_cls([{'i': 3}, {'i': 1}, {'i': 2}, {'i': 5}, {'i': 4}]))
    bs = stream_dat.batch_exactly(2)
    assert isinstance(bs, Iterator)
    bs_lst = list(bs)
    assert len(bs_lst) == 2
    assert all(isinstance(b, Batch) for b in bs_lst)
    dat = list(stream_dat)
    assert list(bs_lst[0]) == [dat[0], dat[1]]
    assert list(bs_lst[1]) == [dat[2], dat[3]]


def test_batch_nonpositive_batch_size(stream_dataset):
    with pytest.raises(ValueError) as exc:
        next(stream_dataset.batch(0))
    assert 'batch size must be greater than 0' in str(exc.value)

    with pytest.raises(ValueError) as exc:
        next(stream_dataset.batch_exactly(0))
    assert 'batch size must be greater than 0' in str(exc.value)


class TestApplyVocab:
    def test_ok(self, stream_cls):
        dat = StreamDataset(
            stream_cls([{
                'w': 'a',
                'ws': ['a', 'b'],
                'cs': [['a', 'b'], ['b', 'a']],
                'i': 10,
                'j': 20
            }, {
                'w': 'b',
                'ws': ['a', 'a'],
                'cs': [['b', 'b'], ['b', 'a']],
                'i': 10,
                'j': 20
            }]))
        vocab = {
            'w': {
                'a': 0,
                'b': 1
            },
            'ws': {
                'a': 2,
                'b': 3
            },
            'cs': {
                'a': 4,
                'b': 5
            },
            'j': {
                20: 2
            }
        }
        dat.apply_vocab(vocab)
        assert list(dat) == [{
            'w': 0,
            'ws': [2, 3],
            'cs': [[4, 5], [5, 4]],
            'i': 10,
            'j': 2
        }, {
            'w': 1,
            'ws': [2, 2],
            'cs': [[5, 5], [5, 4]],
            'i': 10,
            'j': 2
        }]

    def test_key_error(self, stream_cls):
        dat = StreamDataset(stream_cls([{'w': 'a'}]))
        vocab = {'w': {'b': 0}}
        dat.apply_vocab(vocab)
        with pytest.raises(KeyError) as exc:
            list(dat)
        assert "value 'a' not found in vocab" in str(exc.value)

        dat = StreamDataset(stream_cls([{'w': 10}]))
        vocab = {'w': {11: 0}}
        dat.apply_vocab(vocab)
        with pytest.raises(KeyError) as exc:
            list(dat)
        assert "value 10 not found in vocab" in str(exc.value)

    def test_with_vocab_object(self, stream_cls):
        dat = StreamDataset(
            stream_cls([{
                'ws': ['a', 'b'],
                'cs': [['a', 'c'], ['c', 'b', 'c']]
            }, {
                'ws': ['b'],
                'cs': [['b']]
            }]))
        v = Vocab.from_samples(dat)
        dat.apply_vocab(v)
        assert list(dat) == [{
            'ws': [v['ws']['a'], v['ws']['b']],
            'cs': [[v['cs']['a'], v['cs']['c']], [v['cs']['c'], v['cs']['b'], v['cs']['c']]]
        }, {
            'ws': [v['ws']['b']],
            'cs': [[v['cs']['b']]]
        }]
