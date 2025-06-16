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
from dotenv import load_dotenv
from litellm import check_valid_key, completion, batch_completion
from litellm import ModelResponse

from ..media import Modality
from .litellm import LiteLLMModel
# try:
#     import google.generativeai as genai
# except ImportError:
#     raise ImportError("To use Gemini please install dependencies with: pip install llmeval[gemini]")

class GeminiModel(LiteLLMModel):
    def __init__(self,
                 api_key: str = "",
                 model_version: str = 'gemini-1.5-flash',
                 safety_filters: bool = False):

        modalities = [Modality.text, Modality.image, Modality.code, Modality.document]

        # as requested by litellm https://docs.litellm.ai/docs/providers/gemini
        litellm_model = f'gemini/{model_version}'

        # add if safety is on or off
        model_version = model_version + '-safe' if safety_filters else model_version + '-unsafe'

        super().__init__(model_version=model_version,
                         modalities=modalities,
                         publisher="Google",
                         litellm_model=litellm_model)

        if not api_key:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(f"Gemini API key is required either pass it explicitly or set it in the environment variable 'GENIMI_API_KEY'")

        # check api key validity
        if not check_valid_key(model=litellm_model, api_key=api_key):
            raise ValueError(f"GEMINI_API_KEY API key is invalid")
        self.runtime_vars['api_key'] = api_key


        if not safety_filters:
            self.isunsafe = True
            self.runtime_vars['safety_settings'] = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

    def _completion(self,
                    model: str,
                    messages: list[dict],
                    temperature: float = 0.0,
                    max_tokens: int = 4096,
                    completions: int = 1,
                    top_p: float = 0.0,
                    seed: int = 0) -> ModelResponse:
        #! we need to isolate the batch completion to allow various implementation to pass additonals parameters
        resp = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=completions,
            top_p=top_p,
            seed=seed,
            api_key=self.runtime_vars.get('api_key'),
            # gemini specific
            safety_settings=self.runtime_vars.get('safety_settings'))
        return resp

    def _batch_completion(self, model: str, messages_batch: list[dict],
                          temperature: float = 0.0,
                          max_tokens: int = 4096,
                          completions: int = 1,
                          top_p: float = 0.0,
                          seed: int = 0) -> list[ModelResponse]:
        batch_responses = batch_completion(
            model=model,
            messages=messages_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            n=completions,
            top_p=top_p,
            seed=seed,
            api_key=self.runtime_vars.get('api_key'),
            # gemini specific
            safety_settings=self.runtime_vars.get('safety_settings')
            )
        return batch_responses
