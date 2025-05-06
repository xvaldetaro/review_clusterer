from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import os
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient(ABC):
    LLM_NAME: str = "base"  # Should be overridden by child classes
    
    @abstractmethod
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        temperature: Sampling temperature (0.0-1.0)
        """
        pass

    @abstractmethod
    def generate_structured_output(
        self, 
        prompt: str, 
        response_format: Dict[str, Any],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        temperature: Sampling temperature (0.0-1.0)
        """
        pass


class OpenAIClient(LLMClient):
    LLM_NAME: str = "openai"
    
    def __init__(
        self, 
        model_name: str = "gpt-4-turbo", 
        api_key: Optional[str] = None
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install it with: pip install openai"
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
            )
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI client with model: {model_name}")
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        temperature: Sampling temperature (0.0-1.0)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating completion with OpenAI: {str(e)}")
            raise
    
    def generate_structured_output(
        self, 
        prompt: str, 
        response_format: Dict[str, Any],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        temperature: Sampling temperature (0.0-1.0)
        """
        try:
            formatted_prompt = f"{prompt}\n\nRespond with a JSON object that matches this schema: {json.dumps(response_format)}"
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error generating structured output with OpenAI: {str(e)}")
            raise


class AnthropicClient(LLMClient):
    LLM_NAME: str = "anthropic"
    
    def __init__(
        self, 
        model_name: str = "claude-3-opus-20240229", 
        api_key: Optional[str] = None
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install it with: pip install anthropic"
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set"
            )
        
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Initialized Anthropic client with model: {model_name}")
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        temperature: Sampling temperature (0.0-1.0)
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating completion with Claude: {str(e)}")
            raise
    
    def generate_structured_output(
        self, 
        prompt: str, 
        response_format: Dict[str, Any],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        temperature: Sampling temperature (0.0-1.0)
        """
        try:
            tools = [{
                "name": "generate_structured_response",
                "description": "Generate a structured response based on the user's query",
                "input_schema": response_format
            }]
            
            system_prompt = (
                "You are a helpful assistant that provides accurate analysis of customer reviews. "
                "When asked to analyze reviews, you will provide structured responses using JSON."
            )
            
            response = self.client.messages.create(
                model=self.model_name,
                temperature=temperature,
                system=system_prompt,
                tools=tools,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if response.content[0].type == "tool_use":
                tool_use = response.content[0].tool_use
                return tool_use.input
            else:
                content = response.content[0].text
                try:
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = content
                    
                    return json.loads(json_str)
                except:
                    logger.warning("Failed to parse JSON from Claude response")
                    return {"error": "Failed to parse structured output", "raw_response": content}
        except Exception as e:
            logger.error(f"Error generating structured output with Claude: {str(e)}")
            raise


def get_llm_client(provider: str = "openai", **kwargs) -> LLMClient:
    if provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_api_key_from_file(key_file_path: Union[str, Path], provider: str) -> str:
    path = Path(key_file_path)
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    
    with open(path, "r") as f:
        api_key = f.read().strip()
    
    if provider.lower() == "openai" and not api_key.startswith("sk-"):
        logger.warning("OpenAI API key doesn't start with 'sk-', it may not be valid")
    elif provider.lower() == "anthropic" and not api_key.startswith("sk-ant-"):
        logger.warning("Anthropic API key doesn't start with 'sk-ant-', it may not be valid")
    
    return api_key