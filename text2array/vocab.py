# Copyright 2019 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter, UserDict, defaultdict
from typing import Counter as CounterT, Dict, Iterable, Iterator, Mapping, \
    MutableMapping, Optional, Sequence, Set

from ordered_set import OrderedSet
from tqdm import tqdm

from .samples import FieldName, FieldValue, Sample


class Vocab(UserDict, MutableMapping[FieldName, 'StringStore']):
    """A dictionary from field names to `StringStore` objects as the field's vocabulary."""
    def __getitem__(self, name: FieldName) -> 'StringStore':
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError(f"no vocabulary found for field name '{name}'")

    def stoi(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        """Convert strings in the given samples to integers according to this vocabulary.

        This conversion means mapping all the (nested) string field values to integers
        according to the mapping specified by the `StringStore` object of that field.
        Field names with no entry in the vocabulary are ignored. Note that the actual
        conversion is lazy; it is not performed until the resulting iterable is iterated over.

        Args:
            samples (~typing.Iterable[Sample]): Samples to convert.

        Returns:
            ~typing.Iterable[Sample]: Converted samples.
        """
        return map(self._apply_to_sample, samples)

    def itos(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        """Convert integers in the given samples to strings according to this vocabulary.

        This method is essentially the inverse of `~Vocab.stoi`.

        Args:
            samples (~typing.Iterable[Sample]): Samples to convert.

        Returns:
            ~typing.Iterable[Sample]: Converted samples.
        """
        return map(lambda s: self._apply_to_sample(s, index=False), samples)

    @classmethod
    def from_samples(
            cls,
            samples: Iterable[Sample],
            pbar: Optional[tqdm] = None,
            options: Optional[Mapping[FieldName, dict]] = None,
    ) -> 'Vocab':
        """Make an instance of this class from an iterable of samples.

        A vocabulary is only made for fields whose value is a string token or a (nested)
        sequence of string tokens. It is important that ``samples`` be a true iterable, i.e.
        it can be iterated more than once. No exception is raised when this is violated.

        Args:
            samples (~typing.Iterable[Sample]): Iterable of samples.
            pbar: Instance of `tqdm <https://pypi.org/project/tqdm>`_ for displaying
                a progress bar.
            options: Mapping from field names to dictionaries to control the creation of
                the vocabularies. Recognized dictionary keys are:

                * ``min_count`` (`int`): Exclude tokens occurring fewer than this number
                  of times from the vocabulary (default: 1).
                * ``pad`` (`str`): String token to represent padding tokens. If ``None``,
                  no padding token is added to the vocabulary. Otherwise, it is the
                  first entry in the vocabulary (index is 0). Note that if the field has no
                  sequential values, no padding is added. String field values are *not*
                  considered sequential (default: ``<pad>``).
                * ``unk`` (`str`): String token to represent unknown tokens with. If
                  ``None``, no unknown token is added to the vocabulary. This means when
                  querying the vocabulary with such token, an error is raised. Otherwise,
                  it is the first entry in the vocabulary *after* ``pad``, if any (index is
                  either 0 or 1) (default: ``<unk>``).
                * ``max_size`` (`int`): Maximum size of the vocabulary, excluding ``pad``
                  and ``unk``. If ``None``, no limit on the vocabulary size. Otherwise, at
                  most, only this number of most frequent tokens are included in the
                  vocabulary. Note that ``min_count`` also sets the maximum size implicitly.
                  So, the size is limited by whichever is smaller. (default: ``None``).

        Returns:
            Vocab: Vocabulary instance.
        """
        if pbar is None:  # pragma: no cover
            pbar = tqdm(samples, desc='Counting', unit='sample')
        if options is None:
            options = {}

        counter: Dict[FieldName, CounterT[str]] = defaultdict(Counter)
        seqfield: Set[FieldName] = set()
        for s in samples:
            for name, value in s.items():
                if cls._needs_vocab(value):
                    counter[name].update(cls._flatten(value))
                if isinstance(value, Sequence) and not isinstance(value, str):
                    seqfield.add(name)
            pbar.update()
        pbar.close()

        m = {}
        for name, c in counter.items():
            opts = options.get(name, {})

            # Padding and unknown tokens
            pad = opts.get('pad', '<pad>')
            unk = opts.get('unk', '<unk>')
            inits = []
            if name in seqfield and pad is not None:
                inits.append(pad)
            if unk is not None:
                inits.append(unk)

            store = StringStore(inits, default=unk)

            min_count = opts.get('min_count', 1)
            max_size = opts.get('max_size')
            n = len(store)
            for tok, freq in c.most_common():
                if freq < min_count or (max_size is not None and len(store) - n >= max_size):
                    break
                store.add(tok)
            m[name] = store

        return cls(m)

    @classmethod
    def _needs_vocab(cls, val: FieldValue) -> bool:
        if isinstance(val, str):
            return True
        if isinstance(val, Sequence):
            return False if not val else cls._needs_vocab(val[0])
        return False

    @classmethod
    def _flatten(cls, xs) -> Iterator[str]:
        if isinstance(xs, str):
            yield xs
            return

        # must be an iterable, due to how we use this function
        for x in xs:
            yield from cls._flatten(x)

    def _apply_to_sample(self, sample: Sample, index: bool = True) -> Sample:
        fn = self._index_value if index else self._get_value
        s = {}
        for name, value in sample.items():
            try:
                store = self[name]
            except KeyError:
                s[name] = value
            else:
                s[name] = fn(store, value)
        return s

    @classmethod
    def _index_value(cls, store: 'StringStore', value: FieldValue) -> FieldValue:
        if isinstance(value, str):
            return store.index(value)
        if not isinstance(value, Sequence):
            return value

        return [cls._index_value(store, v) for v in value]

    @classmethod
    def _get_value(cls, store: 'StringStore', value: FieldValue) -> FieldValue:
        if not isinstance(value, Sequence):
            return store[value]
        if isinstance(value, str):
            return value

        return [cls._get_value(store, v) for v in value]


class StringStore(OrderedSet):
    """An ordered set of strings, with an optional default value for unknown strings.

    This class implements both `~typing.MutableSet` and `~typing.Sequence` with `str`
    as its contents.

    Example:

        >>> from text2array import StringStore
        >>> store = StringStore('abb', default='a')
        >>> list(store)
        ['a', 'b']
        >>> store.add('b')
        1
        >>> store.add('c')
        2
        >>> list(store)
        ['a', 'b', 'c']
        >>> store.index('a')
        0
        >>> store.index('b')
        1
        >>> store.index('d')
        0

    Args:
        initial: Initial elements of the store.
        default: Default string as a representation of unknown strings, i.e. those that
            do not exist in the store.
    """
    def __init__(
            self,
            initial: Optional[Sequence[str]] = None,
            default: Optional[str] = None,
    ) -> None:
        super().__init__(initial)
        self.default = default

    def index(self, s: str) -> int:
        try:
            return super().index(s)
        except KeyError:
            if self.default is not None:
                return super().index(self.default)
            raise ValueError(f"cannot find '{s}'")

    def __eq__(self, o) -> bool:
        if not isinstance(o, StringStore):
            return False
        return self.default == o.default and super().__eq__(o)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({list(self)!r}, default={self.default!r})'
