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
"""Convert your NLP text data to arrays!"""

__version__ = "0.2.1"
__all__ = [
    "Sample",
    "Batch",
    "Vocab",
    "StringStore",
    "BatchIterator",
    "BucketIterator",
    "ShuffleIterator",
]

from .batches import Batch
from .samples import Sample
from .iterators import BatchIterator, BucketIterator, ShuffleIterator
from .vocab import StringStore, Vocab
