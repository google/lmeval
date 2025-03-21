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

from enum import Enum

class StepType(Enum):
  lmgeneration = "lmgeneration"
  lmrating = "lmrating"
  retrieval = "retrieval"
  ranking = "ranking"
  scoring = "scoring"

# [Benchmark]

class Modality(Enum):
  auto = "auto"  # used for auto detection in question media add
  text = "text"
  image = "image"
  code = "code"
  audio = "audio"
  video = "video"
  document = "document"
  multimodal = "multimodal"   # used for scorers that can handle multiple modalities

class BenchmarkLicence(Enum):
  cc_40_by = "cc-40-by"
  cc_40_by_nc = "cc-40-by-nc"


# [Scorers]
class Regex(Enum):
  "common regex patterns"
  boolean_answer = r'answer: +(.{2,3})'
  multi_choice = r'answer: +(.)'


class ScorerType(Enum):
  # dummy scorers
  always_0 = "always_0"
  always_1 = "always_1"

  # boolean scorers
  boolean_answer = "boolean_answer"

  # mutiple choices scorers
  # this one is for when there is a single correct answer
  contains_answer_letter_insensitive = "contains_answer_letter_insensitive"
  # this one is for when there are multiple correct answers
  contains_answer_letters_insensitive = "contains_answer_letters_insensitive"

  # simple text matching scorers
  text_exact_sensitive = "text_exact_sensitive"
  text_exact_insensitive = "text_exact_insensitive"

  # text containt scorers
  contain_text_sensitive = "contain_text_sensitive"
  contain_text_insensitive = "contain_text_insensitive"

  # regex text matching scorers
  text_regex_sensitive = "text_regex_sensitive"
  text_regex_insensitive = "text_regex_insensitive"

  # llm based scorers
  punt_detector = "punt_detector"
  llm_rater = "llm_rater"

  # NLP scorers
  rouge = "rouge"
  blue = "blue"
  meteor = "meteor"
  mauve = "mauve"



# [question]
class FileType(Enum):
    auto = "auto"
    jpeg = "jepg"
    jpg = "jpeg"  # we need this to pass the right filetype to models
    png = "png"
    gif = "gif"
    mp4 = "mp4"
    mp3 = "mp3"
    wav = "wav"
    txt = "txt"
    python = "py"
    pdf = "pdf"
    html = "html"
    json = "json"


# [Tasks]
class TaskType(Enum):
    boolean = "boolean"
    multiple_choices = "multiple_choices"
    multiple_choices_multiple_answers = "multiple_choices_multiple_answers"
    text_generation = "text_generation"
    completion = "completion"
    grouped_completion = "grouped_completion"

    image_generation = "image_generation"
    python_generation = "python_generation"

    red_text_generation = "red_text_generation"


class TaskLevel(Enum):
  basic = "basic"
  intermediate = "intermediate"
  advanced = "advanced"


class MultiShotStrategy(Enum):
  single = "single"  # no multi-shot scoring
  average = "average"   # take average score
  max = "max" # take the max score
  majority = "majority"  # take the majority score