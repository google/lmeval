from typing import Optional
from pydantic import Field

from lmeval.custom_model import CustomModel
from lmeval.benchmark import Category
from lmeval.task import Task
from lmeval.question import Question, GroupedQuestion
from lmeval.models import LMAnswer, LMModel
from lmeval.prompts import Prompt
from lmeval.scorers import PuntDetector



class EvalTask(CustomModel):  #  Generic[M, P]):
    benchmark_name: str  # needed to propagate it to the models
    question: Question
    category: Category
    task: Task
    lm_model: LMModel  # model is reserved by pydantic, using lm_model
    prompt: Prompt
    instanciated_prompt: str = Field(default="")
    punt_detector: Optional[PuntDetector] = None

    # tracking evaluation status
    lm_answer: Optional[LMAnswer] = None

    # those are shorthand for the answer status that are copied from LMAnswer
    score: float = Field(default=0.0)
    punted: bool = Field(default=False)
    error: bool = Field(default=False)

    def __str__(self) -> str:
        return f"{self.lm_model.version_string}:{self.prompt.name} {self.category.name} / {self.task.name} / {self.question.id}"


class CompletionEvalTask(EvalTask):
    messages: list[dict]


class GroupedCompletionEvalTask(EvalTask):
    question: GroupedQuestion
