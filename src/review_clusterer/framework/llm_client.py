from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import os
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract base class that defines the interface for LLM APIs.
    """
    LLM_NAME: str = "base"  # Should be overridden by child classes
    
    @abstractmethod
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a text completion using the LLM.
        
        Args:
            prompt: The prompt text to send to the LLM
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            The generated completion text
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
        Generate a structured output (JSON) using the LLM.
        
        Args:
            prompt: The prompt text to send to the LLM
            response_format: Schema definition for the desired response format
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            The generated response as a Python dictionary
        """
        pass


class OpenAIClient(LLMClient):
    """
    Implementation of LLMClient for OpenAI API.
    """
    LLM_NAME: str = "openai"
    
    def __init__(
        self, 
        model_name: str = "gpt-4-turbo", 
        api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            model_name: The OpenAI model to use
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
        """
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
        Generate a text completion using OpenAI.
        
        Args:
            prompt: The prompt text to send to OpenAI
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            The generated completion text
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
        Generate a structured output (JSON) using OpenAI.
        
        Args:
            prompt: The prompt text to send to OpenAI
            response_format: Schema definition for the desired response format
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            The generated response as a Python dictionary
        """
        try:
            # Add instructions about the response format to the prompt
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
    """
    Implementation of LLMClient for Anthropic Claude API.
    """
    LLM_NAME: str = "anthropic"
    
    def __init__(
        self, 
        model_name: str = "claude-3-opus-20240229", 
        api_key: Optional[str] = None
    ):
        """
        Initialize the Anthropic Claude client.
        
        Args:
            model_name: The Claude model to use
            api_key: Anthropic API key (if None, will look for ANTHROPIC_API_KEY env var)
        """
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
        Generate a text completion using Anthropic Claude.
        
        Args:
            prompt: The prompt text to send to Claude
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            The generated completion text
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
        Generate a structured output (JSON) using Anthropic Claude.
        
        Args:
            prompt: The prompt text to send to Claude
            response_format: Schema definition for the desired response format
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            The generated response as a Python dictionary
        """
        try:
            # Construct a tool for Claude-3 to use for JSON output
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
            
            # Get tool use from response
            if response.content[0].type == "tool_use":
                tool_use = response.content[0].tool_use
                return tool_use.input
            else:
                # Try to extract JSON from text response
                content = response.content[0].text
                try:
                    # Search for JSON block in the response
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to find any JSON-like structure
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
    """
    Factory function to get an appropriate LLM client.
    
    Args:
        provider: The LLM provider to use ('openai' or 'anthropic')
        **kwargs: Additional arguments to pass to the LLM client constructor
        
    Returns:
        An instance of an LLMClient
    """
    if provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_api_key_from_file(key_file_path: Union[str, Path], provider: str) -> str:
    """
    Read an API key from a file.
    
    Args:
        key_file_path: Path to the file containing the API key
        provider: The LLM provider ('openai' or 'anthropic') to determine key format
        
    Returns:
        The API key as a string
    """
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