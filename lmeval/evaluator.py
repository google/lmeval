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
import json
from pydantic import Field, BaseModel
from tabulate import tabulate
from collections import defaultdict, deque
from tqdm.auto import tqdm
from typing import TypeVar, Generic, Optional
import concurrent.futures
import functools
import lmeval
import threading
import time
import traceback

from lmeval import utils
from lmeval.logger import log
from lmeval.models import LMAnswer, LMModel
from lmeval.scorers import PuntDetector
from lmeval.question import GroupedQuestion, Question
from lmeval.task import Task
from lmeval.benchmark import Benchmark, load_benchmark
from lmeval.prompts import Prompt
from lmeval.callback import Callback
from lmeval.enums import TaskType
from lmeval.evaluation_tasks import CompletionEvalTask, GroupedCompletionEvalTask, EvalTask

# generic type
P = TypeVar('P', bound='Prompt')
M = TypeVar('M', bound='LMModel')


class AnswerStatus(Enum):
    existing = 0
    planned = 1

class Evaluator():
    """
    create a plan report
    add callback so the evaluation can be monitored

    add multichoice support
    create an execution report
    figure out how to do parallel execution with asyncio

    """

    def __init__(self,
                 benchmark: str | Benchmark,
                 save_path: str = "",
                 callback: Callback | None = None,
                 use_tempfile: bool | None = None) -> None:
        "Instantiate the evaluator system for a given benchmark"  # fix docstring
        self.save_path = save_path
        if not self.save_path:
            print("Warning: save_path is not set, results will not be saved.")

        if isinstance(benchmark, str):
            if benchmark == self.save_path:
                print("benchmark_path and save_path are the same, results will be appended to the benchmark file.")
            # fixme: catch error if path is not valid
            self.benchmark: Benchmark = load_benchmark(
                benchmark, use_tempfile=use_tempfile)
        elif isinstance(benchmark, (Benchmark, lmeval.benchmark.Benchmark)):
            self.benchmark = benchmark
        else:
            raise ValueError(f"No benchmark or benchmark path provided - {type(benchmark)} provided")

        # user supplied callback for integration
        self.callback = callback

        # we need the statistics for planning?
        self.benchmark_stats = self.benchmark.get_stats()

        # tasks queues - grouped by model so we can parallelize
        self._tasks: dict[str, deque[EvalTask]] = defaultdict(deque)

        # lock for update shared values
        self._checkpoint_lock = threading.Lock()
        self.num_processed = 0
        self.num_saved = 0

    def plan(self,
             models: M | list[M],
             prompts: P | list[P],
             punt_detector: PuntDetector | None = None,
             max_evaluations_per_task: int = 100,
             display_report: bool = True):

        # stats
        total_evaluations = 0
        # category -> tasks -> prompt -> count
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))))

        # track potential issues - e.g no prompt for a task type
        track_task_prompts: dict[Task, set[str]] = {}

        # boxing models and prompts if not lists
        models_list: list[M] = models if isinstance(models, list) else [models]
        prompts_list: list[P] = prompts if isinstance(prompts, list) else [prompts]

        # initial sanity checks
        # FIXME: call benchmark.validate() when implemented
        # check each model have a different version
        versions = set()
        for model in models_list:
            versions.add(model.version_string)
        assert len(versions) == len(models_list), f"Models should have unique version strings - found {len(models_list)} models and {len(versions)} unique version strings"

        # plan the evaluations
        for category in self.benchmark.categories:
            for task in category.tasks:
                track_task_prompts[task] = set()

                for question in task.questions:
                    for prompt in prompts_list:
                        PROMT_VER = prompt.version_string()
                        # skip if prompt type does not match task type
                        if prompt.task_type != task.type:
                            continue
                        instanciated_prompt = prompt.render(question, task)

                        for model in models_list:
                            # form question->prompt->model->answer

                            # skip running questions that are already answered
                            MODEL_VER = model.version_string

                            # check if the answer already exists
                            if PROMT_VER in question.lm_answers and MODEL_VER in question.lm_answers[
                                    PROMT_VER]:
                                # Already existing results
                                stats[category.name][task.name][PROMT_VER][
                                    MODEL_VER][AnswerStatus.existing] += 1
                                continue

                            # create evaluation task and queue it
                            if task.type == TaskType.completion.value:
                                evaltask = CompletionEvalTask(
                                    benchmark_name=self.benchmark.name,
                                    question=question,
                                    category=category,
                                    task=task,
                                    lm_model=model,
                                    lm_answer=None,
                                    prompt=prompt,
                                    messages=question.messages,
                                    tools=question.tools,
                                    punt_detector=punt_detector,
                                )
                            elif task.type == TaskType.grouped_completion.value:
                                assert isinstance(question, GroupedQuestion), "Grouped completion tasks should have a GroupedQuestion"

                                evaltask = GroupedCompletionEvalTask(
                                    benchmark_name=self.benchmark.name,
                                    question=question,
                                    category=category,
                                    task=task,
                                    lm_model=model,
                                    lm_answer=None,
                                    prompt=prompt,
                                    punt_detector=punt_detector,
                                )

                            # allows to cap the number of evaluations per task
                            if stats[category.name][
                                    task.name][PROMT_VER][MODEL_VER][
                                        AnswerStatus.
                                        planned] >= max_evaluations_per_task:
                                log.debug(
                                    f"Reached max evaluations for task {task.name}, skipping the rest."
                                )
                                break

                            self._tasks[MODEL_VER].append(evaltask)

                            # tracking variables for potential errors
                            track_task_prompts[task].add(
                                prompt.version_string())

                            # stats
                            stats[category.name][task.name][PROMT_VER][
                                MODEL_VER][AnswerStatus.planned] += 1
                            total_evaluations += 1

        # find potential issues
        for tsk, plist in track_task_prompts.items():
            if len(plist) == 0:
                log.warning(
                    f"No prompt for task {tsk.name}, Add prompt of type {tsk.type}"
                )

        # report
        report = defaultdict(list)
        rows = []
        for cat, cdata in stats.items():
            for task, tdata in cdata.items():
                for model, mdata in tdata.items():
                    for prompt, pdata in mdata.items():
                        rows.append([
                            cat, task, model, prompt,
                            pdata[AnswerStatus.planned],
                            pdata[AnswerStatus.existing],
                            pdata[AnswerStatus.planned] +
                            pdata[AnswerStatus.existing]
                        ])
                        report[model].append({
                            "category":
                            cat,
                            "task":
                            task,
                            "prompt":
                            prompt,
                            "planned":
                            pdata[AnswerStatus.planned],
                            "existing":
                            pdata[AnswerStatus.existing]
                        })

        if not display_report:
            return report

        print(f"[{self.benchmark.name} evaluation planning report]")
        print(f"|-Models to evaluate: {len(models_list)}")
        print(f"|-Prompts to evaluate: {len(prompts_list)}")
        print(f"|-Total evaluations to perform: {total_evaluations}")
        print('\n')
        print(
            tabulate(rows,
                     headers=[
                         "Category", "Task", "Prompt", "Model", "Planned",
                         "Existing", "Expected Total"
                     ]))
        return report

    def execute(self,
                save_interval: int = 100,
                use_tempfile: bool | None = None) -> Benchmark:
        """Execute the evaluation plan"""
        num_models = len(self._tasks)  # dict[model_name, deque[EvalTask]]
        if not num_models:
            raise ValueError("No models need to be evaluated")
        self.num_processed = 0
        self.num_saved = 0
        display_progress = []

        def _execute_model(model_name: str, etasks: deque[EvalTask],
                           d_index: int):
            print(
                f"exec model: {model_name}, prompts: {len(etasks)}, medias: {len(etasks[0].question.medias)}"
            )
            num_executed = 0
            model = etasks[0].lm_model
            etasks = [self.prepare_task(etask) for etask in etasks]

            score = 0.0  # live stats
            count = 0
            error = 0
            punt = 0
            for index, answer in model.batch_execute(tasks=etasks):
                assert answer is not None, f"Answer generation failed for model {model_name}"
                log.debug(f"model:index: {model_name}, {index}")
                log.debug(f"model:answer: {answer.answer}")
                etask = etasks[index]
                etask.error = answer.iserror
                if etask.punt_detector:
                    punt_score = etask.punt_detector.score(
                        answer, etask.question, etask.task)
                    log.debug(f"punt_score: {punt_score}")

                    # model is punting
                    if punt_score == 1.0:
                        etask.punted = True
                        answer.ispunting = True
                        answer.punting_reason = answer.raw_response
                        answer.answer = ""
                        log.debug(f"punting detected: {answer.punting_reason}")
                        punt += 1

                etask.lm_answer = answer
                if not etask.lm_answer.ispunting:
                    self.score_answer(etask)
                num_executed += 1
                prompt_ver = etask.prompt.version_string()
                model_ver = etask.lm_model.version_string
                score += answer.score
                count += 1
                error += answer.iserror
                # add answer to benchmark
                # Only one thread at a time can write to the benchmark
                with self._checkpoint_lock:
                    bench_task = self.benchmark.get_task(
                        etask.category.name, etask.task.name)
                    bench_question: Question = bench_task.questions[
                        etask.question.id]

                    if prompt_ver not in bench_question.lm_answers:
                        bench_question.lm_answers[prompt_ver] = {}
                    bench_question.lm_answers[prompt_ver][
                        model_ver] = etask.lm_answer
                    self.num_processed += 1
                    log.debug(
                        "Added answer to benchmark (%s, %d): %s; num processed: %d, num saved: %d",
                        model_name, index, bench_question, self.num_processed, self.num_saved)

                    dp = display_progress[d_index]
                    dp["count"] = count
                    dp["error"] = error
                    dp["punt"] = punt
                    dp["score"] = score
                    if (self.num_processed >= save_interval +
                            self.num_saved) and self.save_path:
                        self.benchmark.save(self.save_path,
                                            use_tempfile=use_tempfile)
                        self.num_saved = self.num_processed
            return num_executed

        with concurrent.futures.ThreadPoolExecutor(num_models) as executor:
            futures = []
            for model_name, etasks in self._tasks.items():
                func = functools.partial(
                    _execute_model,
                    model_name=model_name,
                    etasks=etasks,
                    d_index=len(display_progress),
                )
                display_progress.append({"pbar": tqdm(desc=f"Model {model_name}",
                                                      total=len(etasks)),
                                                      "total": len(etasks),
                                                      "count": 0, "error": 0,
                                                      "punt": 0, "score": 0.0,
                                                      "shown": 0})
                futures.append(executor.submit(func))
            done = False
            while not done:
                done = True
                for i, future in enumerate(futures):
                    if future is None:
                        continue
                    elif future.done():
                        r = future.result()  # need to get result to get errors
                        log.info(f"future: {i} returned {r}")
                        futures[i] = None
                    else:
                        done = False

                    with self._checkpoint_lock:
                        dp = display_progress[i]
                        count = dp["count"]
                        shown = dp["shown"]
                        if dp["shown"] < count:
                            pbar = dp["pbar"]
                            pbar.update(count - shown)
                            dp["shown"] = count
                            pbar.set_postfix({
                                "score": dp["score"] / count,
                                "error_rate": dp["error"] / count,
                                "punt_rate": dp["punt"] / count,
                            })
                    if not done:
                        log.debug("waiting")
                        time.sleep(2)

            for dp in display_progress:
                dp["pbar"].close()

        # save benchmark one last time
        if (self.num_saved < self.num_processed) and self.save_path:
            self.benchmark.save(self.save_path)
            self.num_saved = self.num_processed

        # return benchmark so people can manipulate it after evaluation
        return self.benchmark

    @staticmethod
    def prepare_task(etask: EvalTask) -> EvalTask:
        """Prepares the prompt and other data for a given eval task."""
        if isinstance(etask, GroupedCompletionEvalTask):
            return etask
        elif isinstance(etask, CompletionEvalTask):
            # Prepare messages for CompletionEvalTask
            if not etask.messages or len(etask.messages) == 0:
                etask.messages = [
                    {
                        "role": "user",
                        "content": etask.prompt.render(etask.question, etask.task),
                    }
                ]
            instanciated_prompt = etask.messages
            etask.instanciated_prompt = json.dumps(instanciated_prompt)
        elif etask.instanciated_prompt:
            instanciated_prompt = etask.instanciated_prompt
        else:
            instanciated_prompt = etask.prompt.render(etask.question,
                                                      etask.task)
            etask.instanciated_prompt = instanciated_prompt

        log.debug(f"prompt: {instanciated_prompt}")

        # deal with question media which are not reload from the benchmark file
        if etask.question.medias:
            for media in etask.question.medias:
                if not media.content:
                    if not utils.Path(media.original_path).exists():
                        raise ValueError(
                            f"media {media.original_path} not found")
                    media.content = utils.Path(
                        media.original_path).read_bytes()

        return etask

    @staticmethod
    def generate_answer(etask: EvalTask) -> EvalTask:
        """Generate an answer for a given eval task"""
        if isinstance(etask, GroupedCompletionEvalTask):
            return etask
        elif isinstance(etask, CompletionEvalTask):
            # Prepare messages for CompletionEvalTask
            if not etask.messages or len(etask.messages) == 0:
                etask.messages = [
                    {
                        "role": "user",
                        "content": etask.prompt.render(etask.question, etask.task),
                    }
                ]
            instanciated_prompt = etask.messages
            etask.instanciated_prompt = json.dumps(instanciated_prompt)
        elif etask.instanciated_prompt:
            instanciated_prompt = etask.instanciated_prompt
        else:
            instanciated_prompt = etask.prompt.render(etask.question,
                                                      etask.task)
            etask.instanciated_prompt = instanciated_prompt

        log.debug(f"prompt: {instanciated_prompt}")

        # deal with question media which are not reload from the benchmark file
        if etask.question.medias:
            for media in etask.question.medias:
                if not media.content:
                    if not utils.Path(media.original_path).exists():
                        raise ValueError(
                            f"media {media.original_path} not found")
                    media.content = utils.Path(
                        media.original_path).read_bytes()

        # generate model answer
        if isinstance(etask, CompletionEvalTask):
            print(f"tools here: {etask.tools}")
            model_answer: LMAnswer = etask.lm_model.complete(etask.messages, tools=etask.tools)
        else:
            model_answer: LMAnswer = etask.lm_model.generate_text(
                instanciated_prompt, medias=etask.question.medias)

        log.debug(f"model:answer: {model_answer.answer}")

        etask.error = model_answer.iserror

        # punting detection
        if etask.punt_detector:
            punt_score = etask.punt_detector.score(model_answer,
                                                   etask.question, etask.task)
            log.debug(f"punt_score: {punt_score}")
            if punt_score == 1.0:
                etask.punted = True
                model_answer.ispunting = True
                model_answer.punting_reason = model_answer.answer
                model_answer.answer = ''
                log.debug(f"punting detected: {model_answer.punting_reason}")
        etask.lm_answer = model_answer
        return etask

    @staticmethod
    def score_answer(etask: EvalTask) -> EvalTask:
        """Score an answer for a given eval task"""
        assert etask.lm_answer is not None, "Cannot score an answer that has not been generated"
        assert not etask.lm_answer.ispunting, "Cannot score a punted answer"

        try:
            score = etask.task.scorer.score(etask.lm_answer, etask.question,
                                            etask.task)
            etask.lm_answer.score = score
            etask.score = score
            log.debug(f"answer score: {score}")
        except Exception as e:
            log.error(f"error scoring answer: {e}")
            traceback.print_exc()
            etask.lm_answer.iserror = True
            etask.error = True

        for scorer in etask.task.additional_scorers:
            score = scorer.score(etask.lm_answer, etask.question, etask.task)
            etask.lm_answer.additional_scores[scorer.type] = score
        return etask
