from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import os
import logging
import json
from openai import OpenAI
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.model_name = model_name
        self.api_key = api_key

        client_args = {"api_key": self.api_key}
        client_args["base_url"] = base_url

        self.client = OpenAI(**client_args)
        logger.info(f"Initialized OpenAI-compatible client with model: {model_name}")

    def generate_completion(
        self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7
    ) -> str:
        """
        temperature: Sampling temperature (0.0-1.0)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(
                f"Error generating completion with OpenAI-compatible API: {str(e)}"
            )
            raise

    def generate_structured_output(
        self, prompt: str, response_format: Dict[str, Any], temperature: float = 0.4
    ) -> Dict[str, Any]:
        """
        temperature: Sampling temperature (0.0-1.0)
        """
        try:
            formatted_prompt = f"{prompt}\n\nRespond with a JSON object that matches this schema: {json.dumps(response_format)}"

            try:
                # Try with response_format first (OpenAI API style)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    # response_format={"type": "json_object"},
                    temperature=temperature,
                )
            except Exception as e:
                # If response_format is not supported, fall back to just parsing the response
                logger.warning(f"Falling back to manual JSON parsing: {str(e)}")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=temperature,
                )

            content = response.choices[0].message.content
            logger.info(f"{content}")
            return json.loads(content)
        except Exception as e:
            logger.error(
                f"Error generating structured output with OpenAI-compatible API: {str(e)}"
            )
            raise


def get_api_key_from_file(key_file_path: Union[str, Path]) -> str:
    path = Path(key_file_path)
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")

    with open(path, "r") as f:
        api_key = f.read().strip()

    return api_key
