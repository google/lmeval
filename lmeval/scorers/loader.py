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

from tabulate import tabulate
from .scorer import Scorer

from .dummy_scorer import Always1Scorer, Always0Scorer
from .boolean_answer import BooleanAnswerScorer
from .llm_rater import LLMRater
from .multiple_choices import ContainAnswerLetterInsensitive, ContainAnswerLettersInsensitive
from .exact_text import TextExactInsensitive, TextExactSensitive
from .regex import TextSensitiveRegex, TextInsensitiveRegex
from .contain_text import ContainTextSensitive, ContainTextInsensitive
from .punt_detector import PuntDetector
from ..enums import ScorerType


_SCORERS = {
    ScorerType.always_1: Always1Scorer,
    ScorerType.always_0: Always0Scorer,

    # boolean answer
    ScorerType.boolean_answer: BooleanAnswerScorer,

    # multiple choices
    ScorerType.contains_answer_letter_insensitive: ContainAnswerLetterInsensitive,
    ScorerType.contains_answer_letters_insensitive: ContainAnswerLettersInsensitive,
    # exact text
    ScorerType.text_exact_sensitive: TextExactSensitive,
    ScorerType.text_exact_insensitive: TextExactInsensitive,

    # contain text
    ScorerType.contain_text_sensitive: ContainTextSensitive,
    ScorerType.contain_text_insensitive: ContainTextInsensitive,

    # regex
    ScorerType.text_regex_sensitive: TextSensitiveRegex,
    ScorerType.text_regex_insensitive: TextInsensitiveRegex,

    # safety
    ScorerType.punt_detector: PuntDetector,

    # LLM autorater
    ScorerType.llm_rater: LLMRater,
}

def get_scorer(scorer_type: ScorerType | str) -> Scorer:
    "get scorer object for a given scorer type"
    if scorer_type in _SCORERS:
        return _SCORERS[scorer_type]()
    else:
        raise ValueError(f"Unimplemented scorer {scorer_type}")

def list_scorers():
    "list all available scorers"
    rows = []
    for s in _SCORERS.values():
        si = s()
        rows.append([si.name, si.__class__.__name__ , si.modality.name, si.description])
    print(tabulate(rows, headers=["Name", "Class Name", "Modality", "Description"]))

def add_scorer(t: ScorerType | str, scorer: Scorer):
    "add a custom scorer"
    if t in _SCORERS:
        raise ValueError(f"Scorer {t} already exists")
    _SCORERS[t] = scorer

def update_scorer(t: ScorerType | str, scorer: Scorer):
    "update a custom scorer"
    if t not in _SCORERS:
        raise ValueError(f"Scorer {t} does not exist")
    _SCORERS[t] = scorer
