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

"""Base class for LM models."""
from __future__ import annotations
from time import time
from collections.abc import Generator, Iterable
from typing import Any, Dict, Optional, Tuple
from pydantic import Field
import base64
from ..custom_model import CustomModel
from ..enums import Modality, ScorerType, StepType, MultiShotStrategy, TaskType
from ..media import Media


class LMModel(CustomModel):
    name: str = Field(default='')
    publisher: str = Field(default='')
    modalities: list[Modality] = [Modality.text]  # list of supported modalities
    version_string: str  = Field(default='') # exact string - used for caching gpt-4-1234
    isunsafe: bool = Field(default=False)  # mark if the safety filters are disabled

    # non serialized runtime variables
    runtime_vars: dict = Field(default={}, exclude=True) # ! do not remove exclude=True will leak keys

    # allow to customize model prompt separation tokens
    prompt_prefix: str = Field(default='')
    prompt_suffix: str = Field(default='')

    def generate_text(self,
                      prompt: str,
                      medias: list[Media] | Media = None,
                      temperature: float = 0.0,
                      max_tokens: int = 4096,
                      completions: int = 1) -> LMAnswer:
        raise NotImplementedError

    def generate_image(self,
                       prompt: str,
                       medias: Optional[list[Media]] = None,
                       temperature: float = 0.0,
                       completions: int = 1) -> LMAnswer:
        raise NotImplementedError

    def complete(self, messages: list[dict], temperature: float = 0.0,
                 completions: int = 1) -> LMAnswer:
        raise NotImplementedError

    def multi_complete(self, question: "GroupedQuestion", temperature: float = 0.0,
                       completions: int = 1) -> LMAnswer:
        raise NotImplementedError

    def _build_answer(self, text: str, generation_time: float,
                       iserror: bool = False, error_reason: str = '',
                       total_tokens: int = 0, prompt_tokens: int = 0,
                       completion_tokens: int = 0, isunsafe: bool = False,
                       cost: float = 0.0, prompt:str='',
                       id: str = '') -> LMAnswer:
        """Build an answer object.

        Args:
            text: the generated text
            generation_time: the time it took to generate the text
            iserror: if the generation failed
            error_reason: the reason of the error
            total_tokens: the total tokens used for the generation
            prompt_tokens: the tokens used for the prompt
            completion_tokens: the tokens used for the completion
            isunsafe: if the safety filters were disabled
            cost: the cost of the generation
            prompt: the prompt used for the generation
            id: the id of the completion as returned by the model

        """
        ts = int(time())

        # add generation as step
        step = Step(
            output=text,
            isunsafe=isunsafe,
            type=StepType.lmgeneration,
            # FIXME: support multiple shots
            shots=1,
            MultiShotStrategy=MultiShotStrategy.single,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            iserror=iserror,
            error_reason=error_reason,
            cost=cost,
            timestamp=ts,
            execution_time=generation_time)

        # FIXME we probably need to be able to pass multiple steps before
        steps = [step]
        # FIXME multisteps
        error_step = 1 if iserror else 0

        answer = LMAnswer(id=id,
                          answer=text,
                          isunsafe=isunsafe,
                          error_step=error_step,
                          iserror=iserror,
                          error_reason=error_reason,
                          steps=steps,
                          model=self,
                          text_prompt=prompt)
        return answer

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def _img2base64(self, raw_img: bytes) -> str:
        "convert an image to base64 to send to the model"
        return base64.b64encode(raw_img).decode('utf-8')

    def batch_execute(self, tasks: list["EvalTask"], temperature: float = 0.0,
                      completions: int = 1) -> Generator[Tuple[int, LMAnswer], None, None]:
        """ Execute a batch of prompts in parallel."""
        for i, etask in enumerate(tasks):
            if etask.task.type == TaskType.completion.value:
                yield i, self.complete(etask.messages, temperature, completions, tools=etask.question.tools)
            elif etask.task.type == TaskType.grouped_completion.value:
                yield i, self.multi_complete(etask.question, temperature=temperature, completions=10)
            else:
                # normalize medias
                mds = etask.question.medias if etask.question.medias else []
                mds = mds if isinstance(mds, list) else [mds]
                yield i, self.generate_text(etask.instanciated_prompt, mds, temperature, completions)

    def batch_generate_text(
            self,
            prompts: list[str],
            medias: list[list[Media]],
            temperature: float = 0.0,
            max_tokens: int = 4096,
            completions: int = 1
    ) -> Generator[Tuple[int, LMAnswer], None, None]:
        """ Generate text answers in batches or parallel."""
        for i, prompt in enumerate(prompts):
            yield i, self.generate_text(prompt, medias[i], temperature,
                                        completions)

    def batch_generate_image(self,
                             prompts: list[str],
                             medias: list[list[Media]],
                             temperature: float = 0.0,
                             completions: int = 1) -> Generator[Tuple[int, LMAnswer], None, None]:
        """Generate image answers in batches or parallel."""
        for i, prompt in enumerate(prompts):
            yield i, self.generate_image(prompt, medias[i], temperature,
                                         completions)

class Step(CustomModel):
    output: str
    type: StepType

    # LLM specific
    isunsafe: bool = Field(default=False)
    shots: int = Field(default=1)
    shot_strategy: MultiShotStrategy = Field(default=MultiShotStrategy.single)
    total_tokens: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)

    # error tracking
    iserror: bool = Field(default=False)
    error_reason: str = Field(default='')

    # cost
    cost: float = Field(default=0.0)

    # timing
    timestamp: int = Field(default=0)
    execution_time: float = Field(default=0.0)


class LMAnswer(CustomModel):
    id : str = Field(default='',
                             description="Completions id returned by the model")

    answer: str = Field(default='', description="Final answer")

    # record if we disabled the safety settings
    isunsafe: bool = Field(default=False)

    # record if the executionn have error
    error_step: int = Field(default=0)
    iserror: bool = Field(default=False)
    error_reason: str = Field(default='')

    # record if there as a refusal
    punting_step: int = Field(default=-1)
    ispunting: bool = Field(default=False)
    punting_reason: str = Field(default='')

    # scorer
    score: float = Field(default=0.0)
    additional_scores: Dict[ScorerType | str, float] = Field(default={})
    score_raw_data: Dict[str, Any] = Field(default={})

    # executions steps
    steps: list[Step] = Field(default=[])

    # model used
    model: LMModel

    # set of answers when grouped completion is used
    answer_set: list[LMAnswer] = Field(default_factory=list)

    text_prompt: str = Field(default='',
                             description="record the actual text prompt used")

    additional_data: dict = Field(default_factory=dict,
                                  description="Additional data that are question dependent")

    raw_response: dict = Field(default={},
                               description="Raw response from the model")

    def __str__(self) -> str:
        return str(f"{self.model.name}: {self.answer}")
    