from typing import MutableSet, Sequence

import pytest

from text2array import StringStore


class TestAsSequence:
    def test_ok(self):
        store = StringStore('abbccc')
        assert isinstance(store, Sequence)
        assert len(store) == 3
        assert store[0] == 'a'
        assert store[1] == 'b'
        assert store[2] == 'c'

    def test_index_out_of_range(self):
        store = StringStore()
        with pytest.raises(IndexError) as excinfo:
            store[0]
        assert 'index out of range' in str(excinfo.value)

    def test_value_not_exist(self):
        store = StringStore()
        assert store.default is None
        with pytest.raises(ValueError) as excinfo:
            store.index('d')
        assert "cannot find 'd'" in str(excinfo.value)


class TestAsMutableSet:
    def test_ok(self):
        store = StringStore('abbccc')
        assert isinstance(store, MutableSet)
        assert all(c in store for c in 'abc')

    def test_add(self):
        store = StringStore()
        for c in 'abbccc':
            store.add(c)
        assert list(store) == list('abc')

    def test_discard(self):
        store = StringStore('abbccc')
        store.discard('a')
        store.discard('d')
        assert list(store) == list('bc')


def test_default():
    store = StringStore('abb', default='b')
    assert store.default == 'b'
    assert store.index('c') == store.index(store.default)


def test_eq():
    store1 = StringStore('abbccc')
    store2 = StringStore('abbccc')
    assert store1 == store2
    assert store1 != list('abc')

    store1.discard('c')
    assert store1 != store2

    store3 = StringStore('abbccc', default='b')
    assert store2 != store3
