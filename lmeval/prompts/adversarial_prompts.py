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

from .prompt import Prompt
from ..enums import TaskType

class AdversarialChainOfThoughtsPrompt(Prompt):
    def __init__(self,
                template: str = "{{question.question}} Let's think step by step and answer the question starting with Answer:",
                name: str = "Adversarial Chain of Thoughts",
                description: str = "Ask the model to think steps by steps before answering the question",
                task_type = TaskType.text_generation,
                url: str = '',
                version: str = '1.0'):

        super().__init__(name=name, description=description,
                        task_type=task_type, template=template, url=url,
                        version=version)
