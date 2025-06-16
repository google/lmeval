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

"""Mock model for testing."""

from collections.abc import Generator, Iterable
from typing import Dict, List, Tuple

from lmeval.logger import log
from lmeval.media import Media
from lmeval.models.gemini import GeminiModel
from lmeval.models.lmmodel import LMModel, LMAnswer


class MockModel(LMModel):
  """Mock model for testing.

  Attributes:
    request_response: Dictionary of request to response.
    default_response: Default response if no matching request is found.
  """

  def __init__(
      self,
      model_version: str,
      request_response: Dict[str, str] | None = None,
      default_response: str | None = None,
  ):
    super().__init__(
        name=" ".join(model_version.split("-")).capitalize(),
        publisher="Google",
        version_string=model_version,
    )
    req  = request_response if request_response is not None else {}
    self.runtime_vars["request_response"] = req

    if default_response is not None:
      self.runtime_vars["default_response"] = default_response

  def set_request_response(self, value: Dict[str, str]):
    """Set the request to response mapping."""
    self.runtime_vars["request_response"] = value

  def set_default_response(self, response: str):
    self.runtime_vars["default_response"] = response

  def generate_text(
      self,
      prompt: str,
      medias: List[Media] | Media = [],
      temperature: float | None = 0.0,
      max_tokens: int = 4096,
      completions: int = 1,
      top_p: float = 0.0,
      seed: int = 0) -> LMAnswer:
    # print(f"generate_text: {prompt}")
    id = "mock"
    request_response = self.runtime_vars["request_response"]
    if prompt in request_response:
      answer = self._build_answer(request_response[prompt],
                                 generation_time=0.2,
                                 iserror=False,
                                 error_reason="",
                                 prompt=prompt, id=id)
    elif "default_response" in self.runtime_vars:
        default_response = self.runtime_vars["default_response"]
        answer = self._build_answer(default_response,
                                   generation_time=0.2,
                                   iserror=False,
                                   error_reason="",
                                   prompt=prompt, id=id)
    else:
        answer = self._build_answer("", generation_time=0.2, iserror=True,
                                    id=id, error_reason="No matching request found")

    return answer

  def batch_generate_text(
      self, prompts: list[str], medias: list[list[Media] | Media] = [],
      temperature: float | None = 0.0, max_tokens:int = 4096,
      completions: int = 1, top_p: float = 0.0,
      seed: int = 0) -> Generator[Tuple[int, LMAnswer], None, None]:
    log.info(f"mock-batch_generate_text: {len(prompts)} prompts")
    for i, prompt in enumerate(prompts):
      yield i, self.generate_text(prompt, medias[i], temperature, completions, top_p, seed)


class MockGeminiModel(MockModel):
  """Use batch_generate_text from GeminiModel, generate_text from MockModel."""
  batch_max_workers: int = 2

  def __init__(
      self,
      model_version: str,
      request_response: Dict[str, str] | None = None,
      default_response: str | None = None,
      batch_max_workers: int = 2,
  ):
    super().__init__(model_version, request_response, default_response)
    self.batch_max_workers = batch_max_workers
