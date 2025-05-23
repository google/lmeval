# Copyright 2024 Google LLC
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

from lmeval.custom_model import CustomModel
from lmeval.enums import Modality, FileType
from pydantic import Field


class Media(CustomModel):
    "media represent a unified format for files to be stored in benchmark and used in questions"
    modality: Modality
    filetype: FileType
    filename: str = Field(default="")
    size: int  = Field(default=0)
    # used to track if the file is stored in the benchmark archive
    is_stored: bool = Field(default=False)

    # used to move files during saving or single generation
    # its empty when loaded from a benchmark archive
    original_path: str = Field(default="")

    # used to pass the content during evaluation from benchmark archive
    content: bytes = Field(default_factory=bytes, exclude=True)

    def __str__(self) -> str:
        return str(self.filetype.value)
