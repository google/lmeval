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

import os
from lmeval.models.openai import OpenAIModel
from lmeval import get_scorer, ScorerType, Question, Task, TaskType
from lmeval import Category
from lmeval.evaluator import Evaluator, EvalTask
from lmeval.prompts import SingleWordAnswerPrompt
from .tests_utils import eval_single_text_generation, eval_batch_text_generation, eval_image_analysis

def test_openai_single_text_generation():
    eval_single_text_generation(OpenAIModel())

def test_openai_batch_text_generation():
    eval_batch_text_generation(OpenAIModel())

def test_openai_image_analysis():
    eval_image_analysis(OpenAIModel())
