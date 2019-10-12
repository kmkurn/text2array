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

from collections import Counter, OrderedDict, defaultdict
from typing import Counter as CounterT, Dict, Iterable, Iterator, Mapping, \
    Optional, Sequence, Set

from tqdm import tqdm

from .samples import FieldName, FieldValue, Sample


class Vocab(Mapping[FieldName, Mapping[str, int]]):
    """Namespaced vocabulary storing the mapping from field names to their actual vocabulary.

    A vocabulary does not hold the str-to-int mapping directly, but rather it stores a mapping
    from field names to the corresponding str-to-int mappings. These mappings are the actual
    vocabulary for that particular field name. In other words, the actual vocabulary for each
    field name is namespaced by the field name and all of them are handled this :class:`Vocab`
    object.

    Args:
        m: Mapping from field names to its str-to-int mapping.
    """

    def __init__(self, m: Mapping[FieldName, Mapping[str, int]]) -> None:
        self._m = m

    def __len__(self) -> int:
        return len(self._m)

    def __iter__(self) -> Iterator[FieldName]:
        return iter(self._m)

    def __getitem__(self, name: FieldName) -> Mapping[str, int]:
        try:
            return self._m[name]
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
        return _VocabAppliedSamples(self, samples)

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
                  of times from the vocabulary (default: 2).
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
            store: dict = OrderedDict()
            opts = options.get(name, {})

            # Padding and unknown tokens
            pad = opts.get('pad', '<pad>')
            unk = opts.get('unk', '<unk>')
            if name in seqfield and pad is not None:
                store[pad] = len(store)
            if unk is not None:
                store[unk] = len(store)

            min_count = opts.get('min_count', 2)
            max_size = opts.get('max_size')
            n = len(store)
            for tok, freq in c.most_common():
                if freq < min_count or (max_size is not None and len(store) - n >= max_size):
                    break
                store[tok] = len(store)

            unk_id = None if unk is None else store[unk]
            m[name] = _StringStore(store, unk_id=unk_id)

        return cls(m)

    @classmethod
    def _needs_vocab(cls, val: FieldValue) -> bool:
        if isinstance(val, str):
            return True
        if isinstance(val, Sequence):
            if not val:
                raise ValueError('field values must not be an empty sequence')
            return cls._needs_vocab(val[0])
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


class _VocabAppliedSamples(Iterable[Sample]):
    def __init__(self, vocab: Vocab, samples: Iterable[Sample]) -> None:
        self._vocab = vocab
        self._samples = samples

    def __iter__(self) -> Iterator[Sample]:
        for s in self._samples:
            yield self._vocab._apply_to_sample(s)


class _StringStore(Mapping[str, int]):
    def __init__(self, m: Mapping[str, int], unk_id: Optional[int] = None) -> None:
        assert unk_id is None or unk_id >= 0
        self._m = m
        self._unk_id = unk_id

    def __len__(self) -> int:
        return len(self._m)

    def __iter__(self) -> Iterator[str]:
        return iter(self._m)

    def __getitem__(self, s: str) -> int:
        try:
            return self._m[s]
        except KeyError:
            if self._unk_id is not None:
                return self._unk_id
            raise KeyError(f"'{s}' not found in vocabulary")

    def __contains__(self, s) -> bool:
        return s in self._m

    def __eq__(self, o) -> bool:
        if not isinstance(o, _StringStore):
            return False
        return self._m == o._m and self._unk_id == o._unk_id
