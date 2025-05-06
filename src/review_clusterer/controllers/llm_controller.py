from pathlib import Path
import logging
import json
import os
from typing import Dict, Any, Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

from review_clusterer.framework.llm_client import get_llm_client, get_api_key_from_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def llm_test_controller(
    provider: str = "openai",
    prompt: str = "Summarize the key points of effective customer service.",
    model_name: Optional[str] = None,
    api_key_file: Optional[Path] = None
) -> None:
    """
    Controller for testing LLM client functionality.
    
    Args:
        provider: The LLM provider to use ('openai' or 'anthropic')
        prompt: The prompt to send to the LLM
        model_name: Optional model name to use
        api_key_file: Optional path to a file containing the API key
    """
    console = Console()
    
    try:
        # Get API key
        api_key = None
        if api_key_file:
            api_key = get_api_key_from_file(api_key_file, provider)
        else:
            # Check for environment variable
            env_var = f"{provider.upper()}_API_KEY"
            if os.environ.get(env_var):
                console.print(f"[green]Using API key from {env_var} environment variable[/green]")
            else:
                console.print(f"[yellow]Warning: No API key provided and {env_var} environment variable not set[/yellow]")
        
        # Default model names by provider
        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-opus-20240229",
        }
        
        # Use the provided model or default
        model = model_name or default_models.get(provider.lower())
        
        # Initialize LLM client
        client_kwargs = {"model_name": model}
        if api_key:
            client_kwargs["api_key"] = api_key
            
        llm_client = get_llm_client(provider, **client_kwargs)
        
        # Print request details
        console.print(Panel(
            f"[bold]Provider:[/bold] {provider}\n"
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Prompt:[/bold] {prompt}",
            title="LLM Request",
            box=box.ROUNDED
        ))
        
        # Send prompt and get completion
        console.print("[cyan]Generating response...[/cyan]")
        response = llm_client.generate_completion(prompt)
        
        # Print response
        console.print(Panel(
            response,
            title="LLM Response",
            box=box.ROUNDED
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


def llm_structured_test_controller(
    provider: str = "openai",
    prompt: str = "Analyze these customer reviews for sentiment and key themes: 'The product exceeded my expectations, it's fast and reliable.', 'Terrible customer service, had to wait for hours.'",
    schema: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    api_key_file: Optional[Path] = None
) -> None:
    """
    Controller for testing LLM structured output functionality.
    
    Args:
        provider: The LLM provider to use ('openai' or 'anthropic')
        prompt: The prompt to send to the LLM
        schema: JSON schema for structured response
        model_name: Optional model name to use
        api_key_file: Optional path to a file containing the API key
    """
    console = Console()
    
    # Default schema if none provided
    if schema is None:
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"]
                },
                "key_themes": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "summary": {
                    "type": "string"
                }
            },
            "required": ["sentiment", "key_themes", "summary"]
        }
    
    try:
        # Get API key
        api_key = None
        if api_key_file:
            api_key = get_api_key_from_file(api_key_file, provider)
        
        # Default model names by provider
        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-opus-20240229",
        }
        
        # Use the provided model or default
        model = model_name or default_models.get(provider.lower())
        
        # Initialize LLM client
        client_kwargs = {"model_name": model}
        if api_key:
            client_kwargs["api_key"] = api_key
            
        llm_client = get_llm_client(provider, **client_kwargs)
        
        # Print request details
        console.print(Panel(
            f"[bold]Provider:[/bold] {provider}\n"
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Schema:[/bold]\n{json.dumps(schema, indent=2)}\n\n"
            f"[bold]Prompt:[/bold] {prompt}",
            title="LLM Structured Request",
            box=box.ROUNDED
        ))
        
        # Send prompt and get structured response
        console.print("[cyan]Generating structured response...[/cyan]")
        response = llm_client.generate_structured_output(prompt, schema)
        
        # Print response
        syntax = Syntax(
            json.dumps(response, indent=2),
            "json",
            theme="monokai",
            line_numbers=True
        )
        console.print(Panel(
            syntax,
            title="LLM Structured Response",
            box=box.ROUNDED
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")