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

from collections.abc import Generator
import os
import time
import uuid
import json
import traceback
from typing import Optional, Tuple
from dotenv import load_dotenv
import litellm
from litellm import completion, completion_cost, batch_completion
from litellm import ModelResponse, CustomStreamWrapper

from ..enums import FileType, Modality
from .lmmodel import LMModel
from .lmmodel import LMAnswer
from ..media import Media
from ..logger import log
from ..question import GroupedQuestion


def update_generation_kwargs(generation_kwargs: dict, update: dict) -> dict:
    for key, value in update.items():
        if key not in generation_kwargs:
            generation_kwargs[key] = value
    return generation_kwargs


def proxy_make_model(model='gemini/gemini-2.0-flash-001',
                     proxy=None,
                     proxy_key=None,
                     max_workers=1):
    """Factory method for creating a model via llmproxy."""
    if not proxy or not proxy_key:
        load_dotenv()
    if not proxy_key:
        proxy_key = os.getenv('LITELLM_PROXY_KEY', '')
    if not proxy_key:
        log.error('Missing llm proxy key')
        return None
    if not proxy:
        proxy = os.getenv('LITELLM_PROXY', '')
    if not proxy:
        log.error('Missing proxy address')
        return None
    return LiteLLMModel(model_version=model,
                        litellm_model=f'litellm_proxy/{model}',
                        publisher=model.split('/', maxsplit=1)[0],
                        base_url=proxy,
                        api_key=proxy_key,
                        max_workers=max_workers)


class LiteLLMModel(LMModel):
    """
    Documentation at: https://docs.litellm.ai/
    """

    def __init__(self,
                 model_version: str,
                 litellm_model: str,
                 publisher: str,
                 modalities: list[Modality] = [Modality.text],
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_workers: Optional[int] = 20,
                 benchmark_name: str = "unknown"):
        """Init a LiteLLMModel compatible model

        Args:
            model_version: the name of the model as stored in the benchmark
            litellm_model: the internal name used by litellm
            publisher: the publisher name of the model as stored in the benchmark
            modalities: Which modality are supported by the model. Defaults to [Modality.text].
            base_url: Custom hosted endpoint. Defaults to "".
            api_key: Model API key. Defaults to "".
            max_workers: Number of workers to use for batch completion. Defaults to 100 (Litellm default).
        """

        # clean up the name
        name = ' '.join(model_version.split('-'))
        name = name.lower()

        super().__init__(name=name,
                         publisher=publisher,
                         version_string=model_version,
                         modalities=modalities)

        # store the litellm version name to call it in completions
        self.runtime_vars['litellm_version_string'] = litellm_model
        self.runtime_vars['api_key'] = api_key
        self.runtime_vars['base_url'] = base_url
        self.runtime_vars['is_custom'] = True if base_url else False
        self.runtime_vars['max_workers'] = max_workers
        self.runtime_vars['benchmark_name'] = benchmark_name

    def batch_generate_text(
            self,
            prompts: list[str],
            medias: list[list[Media]],
            temperature: float = 0,
            max_tokens: int = 4096,
            completions: int = 1
    ) -> Generator[Tuple[int, LMAnswer], None, None]:
        model = self.runtime_vars['litellm_version_string']
        assert len(prompts) == len(
            medias), "prompts and medias should have the same length"
        messages_batch = []
        for i, (prompt, media) in enumerate(zip(prompts, medias)):
            messages_batch.append(self._make_messages(prompt, media))

        try:
            batch_responses = self._batch_completion(model, messages_batch,
                                                     temperature, max_tokens,
                                                     completions)
        except:
            batch_responses = [None for _ in prompts]

        for i, resp in enumerate(batch_responses):
            answer = self._make_answer(resp, prompts[i])
            yield i, answer

    def generate_text(self,
                      prompt: str,
                      medias: list[Media] | Media | None = None,
                      temperature: float = 0.0,
                      max_tokens: int = 4096,
                      completions: int = 1) -> LMAnswer:
        # FIXME: finish multi-completion support
        model = self.runtime_vars['litellm_version_string']
        messages = self._make_messages(prompt, medias)

        try:
            resp = self._completion(model=model,
                                    messages=messages,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    completions=completions)
        except Exception as e:
            resp = None
            print("Can't get response from model:", e)
            print(traceback.format_exc())

        answer = self._make_answer(resp, prompt)
        return answer

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        completions: int = 1,
        max_tokens: int = 4096,
        **generation_kwargs,
    ) -> LMAnswer:
        # FIXME: finish multi-completion support
        try:
            arguments = dict(
                model=self.runtime_vars["litellm_version_string"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                completions=completions,
                **generation_kwargs,
            )
            resp = self._completion(**arguments)

        except Exception as e:
            resp = None
            print("Can't get response from model:", traceback.format_exc())

        answer = self._make_answer(resp)
        return answer

    def _make_grouped_answer(self, answers: list[LMAnswer]) -> LMAnswer:
        is_error = any([a.iserror for a in answers])
        error_reason = "".join([a.error_reason for a in answers])
        total_tokens = sum([a.steps[0].total_tokens for a in answers])
        completion_tokens = sum(
            [a.steps[0].completion_tokens for a in answers])
        total_time = sum([a.steps[0].execution_time for a in answers])
        cost = sum([a.steps[0].cost for a in answers])
        answer_id = str(uuid.uuid4())
        print(f"Length of answers: {len(answers)}")
        grouped_answer = self._build_answer(
            text="",
            generation_time=total_time,
            iserror=is_error,
            error_reason=error_reason,
            total_tokens=total_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            id=answer_id)
        grouped_answer.answer_set = answers
        return grouped_answer

    def multi_complete(self,
                       grouped_question: GroupedQuestion,
                       temperature: float = 0.0,
                       completions: int = 1,
                       max_tokens: int = 4096,
                       **generation_kwargs) -> LMAnswer:
        n_completions = grouped_question.metadata.get('n_completions', 1)
        temperature = grouped_question.metadata.get('temperature', None)
        grouped_answers = []

        for question in grouped_question.question_set:
            answer = self.complete(question.messages, temperature,
                                   n_completions, max_tokens,
                                   **generation_kwargs)
            grouped_answers.append(answer)

        return self._make_grouped_answer(grouped_answers)

    def _make_messages(
            self,
            prompt: str,
            medias: list[Media] | Media | None = None) -> list[dict]:
        "build the message to send to the model"
        if medias is None:
            medias = []
        # boxing medias if needed
        if not isinstance(medias, list):
            medias = [medias]

        # build request
        content = []

        # ! some text model struggle with arrays of message so we have to do a if
        if len(medias) > 0:
            # process media files
            for media in medias:
                # image
                if media.modality == Modality.image.value:
                    # FIXME use llmlite
                    image_base64 = self._blob2base64(media.content)
                    # FIXME one has to potentially also handle VertexAI and others (e.g. OpenAI) differently; see PDF below
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/{media.filetype};base64,{image_base64}"
                        }
                    })
                elif media.filetype == FileType.pdf.value:
                    pdf_base64 = self._blob2base64(media.content)
                    if self.publisher in ("gemini", "anthropic"):  # VertexAI uses a different API format for PDF
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_base64}",
                            }
                        })
                    else:
                        content.append({
                            "type": "file",
                            "file": {
                                "filename": "file.pdf",
                                "file_data": f"data:application/pdf;base64,{pdf_base64}"
                            }
                        })
            # text prompt
            content.append({"type": "text", "text": prompt})
            return [{"role": "user", "content": content}]
        else:
            # for model that don't support complex messages structure
            return [{"role": "user", "content": prompt}]

    def _make_answer(self,
                     resp: ModelResponse | CustomStreamWrapper | Exception
                     | None,
                     prompt: str = "") -> LMAnswer:
        iserror = False
        error_reason = ""
        raw_response = ""
        cost = 0
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_time = 0
        model_name = self.runtime_vars['litellm_version_string']
        response_id = ""

        if isinstance(resp, ModelResponse):
            response = resp
            response_id = resp.id

            log.info("response: %s", response)
            try:
                answer_contents = [c.message.content for c in response.choices]
                tool_calls = [c.message.tool_calls for c in response.choices]

                if all(r is None and tc is None
                       for r, tc in zip(answer_contents, tool_calls)):
                    raise ValueError("No response from model")

                for answer in answer_contents:
                    if answer is not None:
                        raw_response = answer
                        break

            except Exception as e:
                try:
                    iserror = True
                    error_reason = f"{error_reason} - {repr(response)} - {(e)}"
                except Exception as f:
                    iserror = True
                    error_reason = f"{error_reason} - {f}"
            total_time = time.time() - response.created
            if not iserror:
                try:
                    # compute cost for known models
                    if not self.runtime_vars['is_custom']:
                        try:
                            cost = completion_cost(response)
                        except Exception as e:
                            cost = 0
                            msg = f"Failed to get cost for model: {model_name}, error: {e}"
                            log.error(msg)
                except Exception as e:
                    msg = f"Failed to get cost for model: {model_name}, error: {e}"
                    log.error(msg)
                try:
                    total_tokens = response.usage.total_tokens
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                except Exception as e:
                    msg = f"Failed to get usage from response for {model_name}, error: {e}"
                    log.error(msg)
            else:
                iserror = True
                error_reason = f'{resp}'

        elif isinstance(resp, Exception):
            iserror = True
            error_reason = repr(resp)
        elif resp is None:
            iserror = True
            error_reason = "Batch completion failed."
        else:
            iserror = True
            error_reason = "Not implemented"

        answer = self._build_answer(text=raw_response,
                                    generation_time=total_time,
                                    iserror=iserror,
                                    error_reason=error_reason,
                                    cost=cost,
                                    total_tokens=total_tokens,
                                    completion_tokens=completion_tokens,
                                    prompt_tokens=prompt_tokens,
                                    isunsafe=self.isunsafe,
                                    prompt=prompt,
                                    id=response_id)
        if isinstance(resp, ModelResponse):
            answer.raw_response = resp.model_dump()
        return answer

    def _batch_completion(self,
                          model: str,
                          messages_batch: list[dict],
                          temperature: float = 0.0,
                          max_tokens: int = 4096,
                          completions: int = 1,
                          **generation_kwargs) -> list[ModelResponse]:

        #! we need to isolate the batch completion to allow various implementation to pass additonals parameters
        if "generation_kwargs" in self.runtime_vars:
            # do not override the generation_kwargs passed as parameter
            generation_kwargs = update_generation_kwargs(
                generation_kwargs, self.runtime_vars["generation_kwargs"])

        if "supports_system_prompt" in self.runtime_vars and not self.runtime_vars[
                "supports_system_prompt"]:
            messages_batch = [
                self._replace_system_messages(messages)
                for messages in messages_batch
            ]
            messages_batch = [
                self._merge_messages_by_role(messages)
                for messages in messages_batch
            ]

        batch_responses = batch_completion(
            model=model,
            messages=messages_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            n=completions,
            api_key=self.runtime_vars.get('api_key'),
            base_url=self.runtime_vars.get('base_url'),
            max_workers=self.runtime_vars.get('max_workers'),
            extra_headers=self._make_headers(),
            **generation_kwargs)
        return batch_responses

    def _completion(self,
                    model: str,
                    messages: list[dict],
                    temperature: float = 0.0,
                    max_tokens: int = 4096,
                    completions: int = 1,
                    **generation_kwargs) -> ModelResponse:
        if "generation_kwargs" in self.runtime_vars:
            # do not override the generation_kwargs passed as parameter
            generation_kwargs = update_generation_kwargs(
                generation_kwargs, self.runtime_vars["generation_kwargs"])

        completion_has_tool_description = False
        if "tools" in generation_kwargs and generation_kwargs[
                "tools"] is not None:
            assert len(
                generation_kwargs["tools"]) > 0, "tools should not be empty"
            completion_has_tool_description = True

        supports_tools_calling = self.runtime_vars.get("supports_tools", False)

        if not supports_tools_calling and completion_has_tool_description:
            litellm.add_function_to_prompt = True
            tools_definition = generation_kwargs.pop("tools")
            generation_kwargs["functions_unsupported_model"] = tools_definition

        if "supports_system_prompt" in self.runtime_vars and not self.runtime_vars[
                "supports_system_prompt"]:
            messages = self._replace_system_messages(messages)
            messages = self._merge_messages_by_role(messages)

        resp = completion(model=model,
                          messages=messages,
                          temperature=temperature,
                          max_tokens=max_tokens,
                          n=completions,
                          api_key=self.runtime_vars.get('api_key'),
                          base_url=self.runtime_vars.get('base_url'),
                          extra_headers=self._make_headers(),
                          **generation_kwargs)
        return resp

    def _replace_system_messages(self, messages: list[dict]) -> list[dict]:
        for m in messages:
            if m["role"] == "system":
                m["role"] = "user"
        return messages

    def _merge_messages_by_role(self, messages: list[dict]) -> list[dict]:
        current_role = messages[0]["role"]
        current_content = messages[0]["content"]

        messages_merged = []
        for m in messages:
            if m["role"] == current_role:
                current_content += "\n\n" + m["content"]
            else:
                messages_merged.append({
                    "role": current_role,
                    "content": current_content
                })
                current_role = m["role"]
                current_content = m["content"]
        messages_merged.append({
            "role": current_role,
            "content": current_content
        })
        return messages_merged

    def _make_headers(self) -> dict[str, str]:
        headers = {
            'x-lmeval-benchmark': self.runtime_vars['benchmark_name'],
        }
        return headers
