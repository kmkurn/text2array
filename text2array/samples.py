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

from typing import Mapping, Sequence, Union

# TODO remove these "type ignore" once mypy supports recursive types
# see: https://github.com/python/mypy/issues/731
FieldName = str
FieldValue = Union[float, int, str, Sequence['FieldValue']]  # type: ignore
Sample = Mapping[FieldName, FieldValue]  # type: ignore
