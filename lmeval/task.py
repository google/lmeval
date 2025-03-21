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

from typing_extensions import Unpack
from litellm import ConfigDict
from pydantic import Field
from typing import List

from lmeval.custom_model import CustomModel
from lmeval.question import Question, GroupedQuestion
from lmeval.enums import TaskType, TaskLevel, MultiShotStrategy, Modality
from lmeval.scorers import Scorer, ContainTextInsensitive, ContainAnswerLetterInsensitive
from lmeval.scorers import BooleanAnswerScorer

class Task(CustomModel):
    name: str
    description: str = Field(default='')

    # metadata
    type: TaskType
    modality: Modality = Field(default=Modality.text)
    level: TaskLevel = Field(default=TaskLevel.basic)

    # scorers
    num_shots: int = Field(default=1)  # how many answers
    multi_short_scoring_strategy: MultiShotStrategy = Field(default=MultiShotStrategy.single)
    scorer: Scorer
    additional_scorers: List[Scorer] = Field(default_factory=list)

    # questions
    # Potentially exclude if scalability and write custom code
    questions: List[Question | GroupedQuestion] = Field(default_factory=list)

    def add_question(self, question: Question) -> int:
        """Add a question to the task

        Args:
            question (Question): question to add

        Returns:
            int: id of the question
        """
        question.id = len(self.questions)
        self.questions.append(question)
        return question.id

    def get_question(self, id: int) -> Question:
        """Get a question by id

        Args:
            id (int): id of the question

        Returns:
            Question: question
        """
        for q in self.questions:
            if q.id == id:
                return q
        return None

    def delete_question(self, id: int) -> bool:
        """Delete a question by id

        Args:
            id (int): id of the question

        Returns:
            bool: True if the question was deleted
        """
        for q in self.questions:
            if q.id == id:
                self.questions.remove(q)
                return True
        return False

    def __str__(self) -> str:
        return f"<{self.type} Task: {self.name}>"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return len(self.questions)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.name == other.name  # Assuming 'name' uniquely identifies a task
        return False

    def __hash__(self):
        return hash(self.name)  # Again, assuming 'name' uniquely identifies a task

# syntactic sugar
class TextGenerationTask(Task):
    """Text generation task"""
    scorer: Scorer  = Field(default_factory=ContainTextInsensitive)
    type: TaskType = Field(default=TaskType.text_generation)

class MultiChoicesTask(Task):
    """Multiple choices task"""
    scorer: Scorer = Field(default_factory=ContainAnswerLetterInsensitive)
    type: TaskType = Field(default=TaskType.multiple_choices)

class YesNoTask(Task):
    """Yes/No task"""
    scorer: Scorer = Field(default_factory=BooleanAnswerScorer)
    type: TaskType = Field(default=TaskType.boolean)

class BooleanQuestion(Task):
    """Boolean question task"""
    scorer: Scorer = Field(default_factory=BooleanAnswerScorer)
    type:  TaskType = Field(default=TaskType.boolean)