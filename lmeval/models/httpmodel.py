# Copyright 2025 Google LLC
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

# Models that use simple http JSON POST to call the LLM

import json
import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any, Dict, Optional, Tuple
from typing_extensions import override

import requests

from ..enums import Modality
from ..logger import log
from ..media import Media
from .lmmodel import LMAnswer, LMModel


class HttpBaseModel(LMModel):
    """Base class for using HTTP to call a model."""

    def __init__(self,
                 model_version: str,
                 publisher: str,
                 modalities: Optional[list[Modality]],
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_workers: Optional[int] = 100,
                 query_suffix: str = 'query',
                 suffix_separator: str = ':',
                 timeout: float = 100,
                 retries: int = 5):
        """Construct a HTTP base model.
            model_version: the name of the model as stored in the benchmark
            publisher: the publisher name of the model as stored in the benchmark
            modalities: Which modality are supported by the model. Defaults to [Modality.text].
            base_url: Custom hosted endpoint. required
            api_key: Model API key. Defaults to "".
            max_workers: Number of workers to use for batch completion. Defaults to 100 (Litellm default).
            query_suffix: The last part of url post to indicate making a query.
            suffix_separtor: The char used to separate base_url and the suffix. If '' or None, don't use.
            timeout: Seconds to timeout
            retries: num time to retry a query
        """
        assert base_url
        if modalities is None:
            modalities = [Modality.text]
        assert modalities
        # clean up the name
        name = ' '.join(model_version.split('-'))
        name = name.lower()
        super().__init__(name=name,
                         publisher=publisher,
                         version_string=model_version,
                         modalities=modalities)

        # store the litellm version name to call it in completions
        self.runtime_vars['api_key'] = api_key
        self.runtime_vars['base_url'] = base_url
        self.runtime_vars['max_workers'] = max_workers
        self.runtime_vars['query_suffix'] = query_suffix
        self.runtime_vars['header'] = self._header()
        self.runtime_vars['suffix_separator'] = suffix_separator
        if suffix_separator:
            self.runtime_vars[
                'query_uri'] = f"{base_url}{suffix_separator}{query_suffix}"
        else:
            self.runtime_vars['query_uri'] = base_url
        self.runtime_vars['timout'] = timeout
        self.runtime_vars['retries'] = retries

    def _header(self):
        api_key = self.runtime_vars.get('api_key')
        return {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def _post_query(
            self,
            query: str,
            temperature: float = 0.0,
            max_tokens: int = 4096,
            top_p: float = 0.0,
            seed: int = 0,
            extra_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Make the POST call. Override in subclass as needed.

           Only do text and single turn for now. 
        """
        # only do text for now
        query_uri = self.runtime_vars.get('query_uri')
        data = {
            'generation_config': {
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'seed': seed,
                'topP': top_p,
                'candidate_count': 1
            },
            'contents': [{
                'parts': [
                    {
                        'text': query
                    },
                ]
            }]
        }
        if extra_dict:
            data.update(extra_dict)

        log.debug('posting %s', str(data))
        for _ in range(0, self.runtime_vars['retries'] + 1):
            try:
                res = requests.post(query_uri,
                                    headers=self.runtime_vars.get('header'),
                                    json=data,
                                    timeout=self.runtime_vars.get('timeout'))
                log.debug('returned: %s', res)
                res.raise_for_status()
                return json.loads(res.text)
            except Exception as e:  # pylint: disable=broad-except
                log.warning('POST encountered error %s', repr(e))
        raise e


class SecLmModel(HttpBaseModel):
    """Model for calling SecLM using HTTP call."""

    def __init__(self,
                 project: Optional[str] = None,
                 workbench_id: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_version: str = 'SecLm',
                 publisher: str = 'GCP',
                 location: str = 'us-central1',
                 modalities: Optional[list[Modality]] = None,
                 safety_filters: bool = False,
                 max_workers: Optional[int] = 100,
                 timeout: float = 100,
                 retries: int = 5):
        if not project:
            project = os.getenv('SECLM_PROJECT', '')
        if not project:
            raise ValueError(
                'project is required either pass in explicitly or set it in the environment variable "SECLM_PROJECT"'
            )
        if not workbench_id:
            workbench_id = os.getenv('SECLM_WORKBENCH_ID', '')
        if not workbench_id:
            raise ValueError(
                'workbench_id is required either pass in explicitly or set it in the environment variable "SECLM_WORKBENCH_ID"'
            )
        if not api_key:
            api_key = os.getenv('SECLM_API_KEY')
        if not api_key:
            raise ValueError(
                'api_key is required either pass in explicitly or set it in the environment variable "SECLM_API_KEY"'
            )
        base_url = f'https://seclm.googleapis.com/v1/projects/{project}/locations/{location}/workbenches/{workbench_id}'

        super().__init__(model_version=model_version,
                         publisher=publisher,
                         modalities=modalities,
                         base_url=base_url,
                         api_key=api_key,
                         max_workers=max_workers,
                         query_suffix='query',
                         suffix_separator=':',
                         timeout=timeout,
                         retries=retries)

        self.runtime_vars['project'] = project
        self.runtime_vars['workbench_id'] = workbench_id
        self.runtime_vars['location'] = location
        if not safety_filters:
            self.isunsafe = True
            self.runtime_vars['safety_settings'] = [
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE',
                },
            ]

    @override
    def generate_text(self,
                      prompt: str,
                      medias: list[Media] | Media = None,
                      temperature: float = 0.0,
                      max_tokens: int = 4096,
                      completions: int = 1,
                      top_p: float = 0.0,
                      seed: int = 0) -> LMAnswer:
        # no media for now
        iserror = False
        error_reason = ''
        gen_time = 0

        extra_dict = {
            'safety_settings': self.runtime_vars.get('safety_settings')
        } if self.isunsafe else None

        try:
            start = time()
            res = self._post_query(prompt, temperature, max_tokens, top_p, seed, extra_dict)
            gen_time = time() - start
            text = res['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:  # pylint: disable=broad-except
            iserror = True
            error_reason = repr(e)

        return self._build_answer(text=text,
                                  generation_time=gen_time,
                                  iserror=iserror,
                                  error_reason=error_reason,
                                  prompt=prompt)

    @override
    def batch_generate_text(
            self,
            prompts: list[str],
            medias: list[list[Media]],
            temperature: float = 0,
            max_tokens: int = 4096,
            completions: int = 1,
            top_p: float = 0.0,
            seed: int = 0
    ) -> Generator[Tuple[int, LMAnswer], None, None]:
        # only handle text for now but check the size matches anyway.
        assert len(prompts) == len(medias)
        try:
            batch_responses = self._batch_completion(prompts, medias,
                                                     temperature, max_tokens,
                                                     completions, top_p, seed)
        except Exception as e:  # pylint: disable=broad-except
            error_message = f'batch generation failed {repr(e)}'
            log.error(error_message)
            batch_responses = [
                self._build_answer(text='',
                                   generation_time=0,
                                   iserror=True,
                                   error_reason=error_message)
                for _ in range(len(prompts))
            ]

        for i, answer in enumerate(batch_responses):
            yield i, answer

    def _batch_completion(self, prompts: list[str], medias: list[list[Media]],
                          temperature: float, max_tokens: int,
                          completions: int, top_p: float = 0.0,
                          seed: int = 0) -> list[LMAnswer]:
        # only do text for now
        completions = []
        with ThreadPoolExecutor(
                max_workers=self.runtime_vars.get('max_workers')) as executor:
            for i, p in enumerate(prompts):

                future = executor.submit(self.generate_text, p, medias[i],
                                         temperature, max_tokens, completions,
                                         top_p, seed)
                completions.append(future)

        return [future.result() for future in completions]
