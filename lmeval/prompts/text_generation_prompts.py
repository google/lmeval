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

from ..enums import TaskType
from .prompt import Prompt

class QuestionOnlyPrompt(Prompt):
    def __init__(self,
                template: str = "{{question.question}}",
                name: str = "Question Only",
                description: str = "Ask the model to answer without any additional context",
                task_type = TaskType.text_generation,
                url: str = '',
                version: str = '1.0'):

        super().__init__(name=name, description=description,
                        task_type=task_type, template=template, url=url,
                        version=version)

class ChainOfThoughtsPrompt(Prompt):

    def __init__(self,
                 template: str = "{{question.question}} Let's think step by step and then answer the question starting with Answer:",
                 name: str = "Chain of Thoutghts",
                 description: str = "Ask the model to answere the question using a single word",
                 task_type = TaskType.text_generation,
                 url: str = '',
                 version: str = '1.0'):
        super().__init__(name=name, description=description,
                        task_type=task_type, template=template, url=url,
                        version=version)


class SingleWordAnswerPrompt(Prompt):
    def __init__(self,
                 template: str = "answer the following question {{question.question}} using a single word. Don't add any extra characters or words.",
                 name: str = "Single Word Answer",
                 description: str = "Ask the model to answere the question using a single word",
                 task_type = TaskType.text_generation,
                 url: str = '',
                 version: str = '1.0'):

        super().__init__(name=name, description=description,
                        task_type=task_type, template=template, url=url,
                        version=version)