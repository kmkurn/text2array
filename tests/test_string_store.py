from text2array import StringStore


def test_add():
    store = StringStore()
    for tok in 'a b b c c c'.split():
        store.add(tok)
    assert list(store.items()) == [('a', 0), ('b', 1), ('c', 2)]
