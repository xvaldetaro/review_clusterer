from pathlib import Path
import logging
import json
import os
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

from review_clusterer.framework.llm_client import LLMClient, get_api_key_from_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


def get_llm_client(
    base_url: str,
    api_key_file: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> LLMClient:
    api_key = None
    if api_key_file:
        api_key = get_api_key_from_file(api_key_file)
    else:
        env_var = "API_KEY"
        if os.environ.get(env_var):
            console.print(
                f"[green]Using API key from {env_var} environment variable[/green]"
            )
            api_key = os.environget(env_var)
        else:
            console.print(
                f"[yellow]Warning: No API key provided and {env_var} environment variable not set[/yellow]"
            )
            api_key = "no_api_key"

    return LLMClient(base_url, api_key, model_name)


def llm_test_controller(
    base_url: str,
    prompt: str,
    model_name: str,
    api_key_file: Optional[Path] = None,
) -> None:
    try:
        llm_client = get_llm_client(base_url, api_key_file, model_name)

        # Format API endpoint information
        endpoint_info = f"[bold]API Endpoint:[/bold] {base_url}"

        console.print(
            Panel(
                f"{endpoint_info}\n"
                f"[bold]Model:[/bold] {model_name}\n"
                f"[bold]Prompt:[/bold] {prompt}",
                title="LLM Request",
                box=box.ROUNDED,
            )
        )

        console.print("[cyan]Generating response...[/cyan]")
        response = llm_client.generate_completion(prompt)

        console.print(Panel(response, title="LLM Response", box=box.ROUNDED))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


def llm_structured_test_controller(
    base_url: str,
    prompt: str,
    model_name: str,
    api_key_file: Optional[Path] = None,
    schema: Optional[dict] = None,
) -> None:
    console = Console()

    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed"],
            },
            "key_themes": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["sentiment", "key_themes", "summary"],
    }

    try:
        llm_client = get_llm_client(base_url, api_key_file, model_name)
        # Format API endpoint information
        endpoint_info = f"[bold]API Endpoint:[/bold] {base_url}"

        console.print(
            Panel(
                f"{endpoint_info}\n"
                f"[bold]Model:[/bold] {model_name}\n"
                f"[bold]Schema:[/bold]\n{json.dumps(schema, indent=2)}\n\n"
                f"[bold]Prompt:[/bold] {prompt}",
                title="LLM Structured Request",
                box=box.ROUNDED,
            )
        )

        console.print("[cyan]Generating structured response...[/cyan]")
        response = llm_client.generate_structured_output(prompt, schema)
        syntax = Syntax(
            json.dumps(response, indent=2), "json", theme="monokai", line_numbers=True
        )
        console.print(Panel(syntax, title="LLM Structured Response", box=box.ROUNDED))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
