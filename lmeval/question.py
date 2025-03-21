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

from pydantic import Field
from pathlib import Path
from hashlib import blake2b
from typing import Any, List, Dict

from collections import defaultdict
from .custom_model import CustomModel
from .media import Media
from .models import LMAnswer
from .enums import Modality, FileType


class QuestionSource(CustomModel):
    name: str
    url: str = Field(default="")
    description: str = Field(default="")

    def __str__(self) -> str:
        return str(self.name)


class Question(CustomModel):
    # we need it for (de)serialization - automated added by Task.add()
    id: int = Field(default=-1)
    question: str | None = Field(default=None)
    language: str = Field(default="en")

    # answer
    answer: str | None = Field(default=None)
    # additional answers e.g. for multiple choice questions
    additional_answers: List[str] = Field(default_factory=list)

    # multiple choice questions only
    # Added at benchmark generation time
    choices: List[str] = Field(default_factory=list)  # alternative choices for multiple choice questions
    # dynamically computed at first prompt rendering time we keep track per prompt_version hence the dicts
    answer_letter: str = Field(default="")
    letters: str = Field(default="")
    multi_choices: str = Field(default="")
    letter_mapping: dict = Field(default_factory=dict,
                                 description="Keep track of which letter is associated with which answer")
    original_letters: List[str] = Field(default_factory=list,
                                        description="Keep track of the original letters for: [anwer] + additional_answers + choices")

    # cache template rendering keyed by prompt version to ensure consistency
    # accross model evaluations.
    # In particular this is used for multi_choices prompts to ensure that the same
    # answer order is used for all models.
    prompt_cache: dict = Field(default_factory=dict)

    # question source and extra media
    source: QuestionSource = Field(default=None)
    medias: List[Media] = Field(default_factory=list)

    # messages to be sent to the llm model
    messages: List[dict] = Field(default_factory=list)
    # metadata for the question and for the evaluation
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # tools to be used by the llm model
    tools: List[dict] | None = None

    # can't use model_ : it is reserved for pydantic
    # lm_answers[prompt_version][model_version, LMAnswer]
    lm_answers: Dict[str, Dict[str, LMAnswer]] = Field(default_factory=dict)


    def add_media(self, path: str| Path, filetype: FileType = FileType.auto,
                  modality: Modality = Modality.auto):
        """add an media to the question, it is saved to the media folder
        of the benchmark archive"""
        # check if it exists

        path = Path(path)
        assert path.exists(), f"file {path} does not exist"

        FILETYPES = {".jpg": [FileType.jpg, Modality.image],
                     ".jpeg": [FileType.jpg, Modality.image],
                     ".png": [FileType.png, Modality.image],
                     ".mp4": [FileType.mp4, Modality.video],
                     ".py": [FileType.python, Modality.code],
                    }

        # auto detection if needed
        if filetype == FileType.auto or modality == Modality.auto:
            suffix = path.suffix.lower()
            if suffix in FILETYPES:
                filetype, modality = FILETYPES[suffix]
            else:
                raise ValueError(f"unsupported media format {path.suffix}")

        # compute hash
        hash = self._compute_file_hash(path)
        filename = f"{hash}{path.suffix}"

        media = Media(filetype=filetype,
                      modality=modality,
                      filename=filename,
                      size=path.stat().st_size,
                      original_path=str(path))
        self.medias.append(media)


    def _compute_file_hash(self, path: Path,
                           digest_size: int = 16) -> str:
        h = blake2b(digest_size=digest_size)
        h.update(path.read_bytes())
        return h.hexdigest()
    
class GroupedQuestion(Question):
    metadata: Dict[str, Any]
    question_set: List[Question] 