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
from litellm import check_valid_key, completion, batch_completion
from litellm import ModelResponse

from ..media import Modality
from .litellm import LiteLLMModel
try:
    from google.cloud import aiplatform
except ImportError:
    raise ImportError("To use Vertex please install dependencies with: pip install llmeval[vertex]")

## subclasses per provider

class VertexModel(LiteLLMModel):
    def __init__(self,
                 model_version: str = "mistral-nemo@2407",
                 location: str = "",
                 project: str = "",
                 safety_filters: bool = False):


        modalities = [Modality.text, Modality.code]
        # as requested by litellm https://docs.litellm.ai/docs/providers/gemini
        litellm_model = f'vertex_ai/{model_version}'
        publisher = model_version.split('-')[0].capitalize()
        model_version = model_version.split('-')[1].replace('@', '-')
        model_version = model_version + '-safe' if safety_filters else model_version + '-unsafe'
        super().__init__(model_version=model_version,
                         modalities=modalities,
                         publisher=publisher,
                         litellm_model=litellm_model)

        if not location:
            location = os.getenv("VERTEX_LOCATION", "")
        if not location:
            raise ValueError(f"VERTEX_LOCATION is required either pass it explicitly or set it in the environment variable 'VERTEX_LOCATION'")

        if not project:
            project = os.getenv("VERTEX_PROJECT", "")
        if not project:
            raise ValueError(f"VERTEX_PROJECT is required either pass it explicitly or set it in the environment variable 'VERTEX_PROJECT'")

        self.runtime_vars['vertex_location'] = location
        self.runtime_vars['vertex_project'] = project


    def _completion(self,
                    model: str,
                    messages: list[dict],
                    temperature: float = 0.0,
                    max_tokens: int = 4096,
                    completions: int = 1,
                    top_p: float = 0.0,
                    seed: int = 0,
                    ) -> ModelResponse:
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
            # vertex specific
            vertex_location=self.runtime_vars.get('vertex_location'),  # Vertex location
            vertex_project=self.runtime_vars.get('vertex_project'),  # Vertex project
        )
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
            # vertex specific
            vertex_location=self.runtime_vars.get('vertex_location'),  # Vertex location
            vertex_project=self.runtime_vars.get('vertex_project'),  # Vertex project
            )
        return batch_responses
