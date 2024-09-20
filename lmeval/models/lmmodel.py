"""Base class for LM models."""
from __future__ import annotations
from time import time
from collections.abc import Generator, Iterable
from typing import Any, Dict, Optional, Tuple
from pydantic import Field
import base64
from ..custom_model import CustomModel
from ..enums import Modality, ScorerType, StepType, MultiShotStrategy
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

    def generate_text(self, prompt: str, medias: list[Media] | Media = None,
                      temperature: float = 0.0,
                      completions: int = 1) -> LMAnswer:
        raise NotImplementedError

    def generate_image(self, prompt: str, medias: Optional[list[Media]] = None,
                       temperature: float = 0.0,
                       completions: int = 1) -> LMAnswer:
        raise NotImplementedError

    def _build_answer(self, text: str, generation_time: float,
                       iserror: bool = False, error_reason: str = '',
                       total_tokens: int = 0, prompt_tokens: int = 0,
                       completion_tokens: int = 0, isunsafe: bool = False,
                       cost: float = 0.0) -> LMAnswer:
        ts = int(time())

        # add generation as step
        step = Step(output=text,
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

        answer = LMAnswer(answer=text,
                          isunsafe=isunsafe,
                          error_step=error_step,
                          iserror=iserror,
                          error_reason=error_reason,
                          steps=steps,
                          model=self)
        return answer

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def _img2base64(self, raw_img: bytes) -> str:
        "convert an image to base64 to send to the model"
        return base64.b64encode(raw_img).decode('utf-8')

    def batch_generate_text(self,
                            prompts: list[str],
                            medias: list[list[Media]],
                            temperature: float = 0.0,
                            completions: int = 1) -> Generator[Tuple[int, LMAnswer], None, None]:
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

    answer: str  # final answer

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
    additional_scores: Dict[ScorerType, float] = Field(default={})

    # executions steps
    steps: list[Step] = Field(default=[])

    # model used
    model: LMModel

    def __str__(self) -> str:
        return str(f"{self.model.name}: {self.answer}")
