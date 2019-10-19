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
    """A dictionary that maps field names to their actual vocabularies.

    This class does not hold the str-to-int (or int-to-str) mapping directly, but rather it
    stores a mapping from field names to the corresponding str-to-int (or int-to-str) mappings.
    The latter are the actual vocabulary for that particular field name. In other words, the
    actual vocabulary for each field name is namespaced by the field name and an instance of this
    class handles all of them.

    Args:
        m: Mapping from field names to its str-to-int (or int-to-str) mapping.
    """

    def __getitem__(self, name: FieldName) -> 'StringStore':
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError(f"no vocabulary found for field name '{name}'")

    def to_indices(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        """Apply this vocabulary to the given samples.

        Applying a vocabulary means mapping all the (nested) field values to the corresponding
        values according to the mapping specified by the vocabulary. Field names that have
        no entry in the vocabulary are ignored. Note that the actual application is not
        performed until the resulting iterable is iterated over.

        Args:
            samples (~typing.Iterable[Sample]): Apply vocabulary to these samples.

        Returns:
            ~typing.Iterable[Sample]: Samples to which the vocabulary has been applied.
        """
        return map(self._index_sample, samples)

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
        A `Vocab` object returned from this method maps field names to `StringStore`, which
        is a mapping from `str` to `int` with minor enhancements.

        Args:
            samples (~typing.Iterable[Sample]): Iterable of samples.
            pbar: Instance of `tqdm <https://pypi.org/project/tqdm>`_ for displaying
                a progress bar.
            options: Mapping from field names to dictionaries to control the creation of
                the str-to-int mapping. Recognized dictionary keys are:

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

            store = StringStore(inits, unk_token=unk)

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

    def _index_sample(self, sample: Sample) -> Sample:
        s = {}
        for name, value in sample.items():
            try:
                store = self[name]
            except KeyError:
                s[name] = value
            else:
                s[name] = self._index_value(store, value)
        return s

    @classmethod
    def _index_value(cls, store: 'StringStore', value: FieldValue) -> FieldValue:
        if isinstance(value, str):
            return store.index(value)
        if not isinstance(value, Sequence):
            return value

        return [cls._index_value(store, v) for v in value]


class StringStore(OrderedSet):
    """An ordered collection of string.

    This class represents an ordered collection of string. This class is also a mapping
    whose keys and values are the strings and their indices in the ordering.

    Args:
        initials: Initial elements of the collection.
        unk_token: Use this token as a representation of strings that do not exist in
            the collection.
    """

    def __init__(
            self,
            initials: Optional[Sequence[str]] = None,
            unk_token: Optional[str] = None,
    ) -> None:
        super().__init__(initials)
        self.unk_token = unk_token

    def index(self, s: str) -> int:
        try:
            return super().index(s)
        except KeyError:
            if self.unk_token is not None:
                return super().index(self.unk_token)
            raise ValueError(f"cannot find '{s}'")

    def __eq__(self, o) -> bool:
        if not isinstance(o, StringStore):
            return False
        return self.unk_token == o.unk_token and super().__eq__(o)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({list(self)!r}, unk_token={self.unk_token!r})'
