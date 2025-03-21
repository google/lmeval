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

from ..custom_model import CustomModel
from ..models import LMModel, LMAnswer
from ..question import Question

from ..enums import ScorerType, Modality


class Scorer(CustomModel):
    name: str = Field(default='')
    description: str = Field(default='')
    type: ScorerType | str  # the enum type of the scorer used for serialization
    modality: Modality  # what type of answer this scorer is for e.g text or multimodal

    # optional fields used only by specific scorers
    regex: str = Field(default='')
    model: LMModel | None = Field(default=None)

    def _score(self,
               model_answer: LMAnswer,
               question: Question,
               task,
               debug: bool = False) -> float:
        "Function to be implemented by each scorer"
        raise NotImplementedError

    def score(self, model_answer: LMAnswer, question: Question, task) -> float:
        """Return the score for the model answer to the question

        notes:
         - if there is an error in the model answer, return -1.0 so we can filter it out
        """
        # always return -1 if the model answer is an error
        if model_answer.iserror:
            return -1.0  # so we can filter
        # if punting, return 0
        if model_answer.ispunting:
            return 0.0
        return self._score(model_answer, question, task)

    def _cleanup(self, txt: str) -> str:
        "Clean up text for comparison"
        txt = txt.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        txt = ' '.join(txt.split()).strip()  # remove multiple spaces
        return txt

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)
