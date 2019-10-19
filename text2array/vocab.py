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

from collections import Counter, OrderedDict, UserDict, defaultdict
from typing import Any, Counter as CounterT, Dict, Iterable, Iterator, Mapping, \
    MutableMapping, Optional, Sequence, Set, Union

from tqdm import tqdm

from .samples import FieldName, FieldValue, Sample


class Vocab(UserDict, MutableMapping[FieldName, Union[Mapping[str, int], Mapping[int, str]]]):
    """A dictionary that maps field names to their actual vocabularies.

    This class does not hold the str-to-int (or int-to-str) mapping directly, but rather it
    stores a mapping from field names to the corresponding str-to-int (or int-to-str) mappings.
    The latter are the actual vocabulary for that particular field name. In other words, the
    actual vocabulary for each field name is namespaced by the field name and an instance of this
    class handles all of them.

    Args:
        m: Mapping from field names to its str-to-int (or int-to-str) mapping.
    """

    def __init__(
            self, m: Mapping[FieldName, Union[Mapping[str, int], Mapping[int, str]]]) -> None:
        super().__init__(m)

    def __getitem__(self, name: FieldName) -> Union[Mapping[str, int], Mapping[int, str]]:
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError(f"no vocabulary found for field name '{name}'")

    def apply_to(self, samples: Iterable[Sample]) -> Iterable[Sample]:
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
        return map(self._apply_to_sample, samples)

    def invert(self) -> 'Vocab':
        """Invert the stored vocabularies.

        Inversion here means changing from str-to-int to int-to-str and vice versa.
        This method is useful to map from integer samples back to their string
        representations.

        Returns:
            Vocab: New vocabulary with the mappings inverted.
        """
        return Vocab({name: self._invert_mapping(vb) for name, vb in self.items()})

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

            store = StringStore(initials=inits, unk_token=unk)

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

    def _apply_to_sample(self, sample: Sample) -> Sample:
        s = {}
        for name, val in sample.items():
            try:
                vb = self[name]
            except KeyError:
                s[name] = val
            else:
                s[name] = self._apply_vb_to_val(vb, val)
        return s

    @classmethod
    def _apply_vb_to_val(
            cls,
            vb: Mapping[FieldValue, FieldValue],
            val: FieldValue,
    ) -> FieldValue:
        if isinstance(val, str) or not isinstance(val, Sequence):
            try:
                return vb[val]
            except KeyError:
                raise KeyError(f'value {val!r} not found in vocab')

        return [cls._apply_vb_to_val(vb, v) for v in val]

    @staticmethod
    def _invert_mapping(d: Mapping[Any, Any]) -> dict:
        return {v: k for k, v in d.items()}


class StringStore(Mapping[str, int]):
    """An ordered collection of string.

    This class represents an ordered collection of string. This class is also a mapping
    whose keys and values are the strings and their indices in the ordering.

    Example:

        >>> from text2array import StringStore
        >>> store = StringStore(initials=['a', 'b'], unk_token='b')
        >>> for w in 'a b b c c c'.split():
        ...     store.add(w)
        ...
        >>> list(store.items())
        [('a', 0), ('b', 1), ('c', 2)]
        >>> store['d']
        1

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
        if initials is None:
            initials = []

        self._initials = initials
        self._unk_token = unk_token

        self._store: Dict[str, int] = OrderedDict()
        for s in initials:
            self.add(s)

    def add(self, s: str) -> None:
        """Add a string to the collection.

        Args:
            s: String to add.
        """
        if s not in self:
            self._store[s] = len(self)

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __getitem__(self, s: str) -> int:
        try:
            return self._store[s]
        except KeyError:
            if self._unk_token is not None:
                return self._store[self._unk_token]
            raise KeyError(f"'{s}' not found in vocabulary")

    def __contains__(self, s) -> bool:
        return s in self._store

    def __eq__(self, o) -> bool:
        if not isinstance(o, StringStore):
            return False
        return list(self) == list(o)
