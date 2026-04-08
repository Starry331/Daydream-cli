import sys

import click
from rich.console import Console

from daydream import __version__
from daydream.config import (
    get_default_host,
    get_default_max_tokens,
    get_default_port,
    get_default_temp,
    get_default_top_p,
)

err_console = Console(stderr=True)


def _handle_errors(func):
    """Decorator to catch common errors and print friendly messages."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            err_console.print(f"[red]Error:[/] {e}")
            raise SystemExit(1)
        except KeyboardInterrupt:
            err_console.print()
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as e:
            err_console.print(f"[red]Error:[/] {e}")
            raise SystemExit(1)

    return wrapper


@click.group()
@click.version_option(version=__version__, prog_name="daydream")
def cli():
    """Daydream — Apple Silicon local model CLI, powered by MLX."""
    from daydream.config import ensure_home

    ensure_home()


@cli.command()
@click.argument("model")
@click.argument("prompt", required=False, default=None)
@click.option("--temp", type=float, default=get_default_temp(), show_default=True, help="Sampling temperature")
@click.option("--top-p", type=float, default=get_default_top_p(), show_default=True, help="Nucleus sampling top-p")
@click.option("--max-tokens", "-m", type=int, default=get_default_max_tokens(), show_default=True, help="Max tokens to generate")
@click.option("--system", "-s", type=str, default=None, help="System prompt")
@click.option("--verbose", "-v", is_flag=True, help="Show performance metrics")
@_handle_errors
def run(model, prompt, temp, top_p, max_tokens, system, verbose):
    """Run a model — interactive chat or one-shot generation."""
    from daydream.chat import run_chat, run_oneshot

    if prompt is not None or not sys.stdin.isatty():
        run_oneshot(model, prompt=prompt, temp=temp, top_p=top_p,
                    max_tokens=max_tokens, system=system, verbose=verbose)
    else:
        run_chat(model, temp=temp, top_p=top_p, max_tokens=max_tokens,
                 system=system, verbose=verbose)


@cli.command()
@click.argument("model")
@_handle_errors
def pull(model):
    """Download a model from HuggingFace."""
    from daydream.models import pull_model
    pull_model(model)


@cli.command(name="list")
@_handle_errors
def list_models():
    """List downloaded models."""
    from daydream.models import list_models
    list_models()


@cli.command()
@click.argument("model")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@_handle_errors
def rm(model, force):
    """Remove a downloaded model."""
    from daydream.models import remove_model
    remove_model(model, force=force)


@cli.command()
@click.argument("model")
@_handle_errors
def show(model):
    """Show model information."""
    from daydream.models import show_model
    show_model(model)


@cli.command()
@click.option("--model", type=str, default=None, help="Model to preload")
@click.option("--host", default=get_default_host(), show_default=True, help="Bind address")
@click.option("--port", "-p", type=int, default=get_default_port(), show_default=True, help="Port number")
@click.option("--background", is_flag=True, help="Run server in the background")
@click.option("--foreground", is_flag=True, hidden=True)
@click.option("--detach", is_flag=True, hidden=True)
@_handle_errors
def serve(model, host, port, background, foreground, detach):
    """Start or manage the OpenAI-compatible API server."""
    from daydream.server import start_server
    start_server(
        model=model,
        host=host,
        port=port,
        detach=background or detach,
    )


@cli.command()
@_handle_errors
def ps():
    """Show running model / server status."""
    from daydream.server import show_status
    show_status()


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Force kill the server if it does not stop cleanly")
@_handle_errors
def stop(force):
    """Stop the managed background server."""
    from daydream.server import stop_server
    stop_server(force=force)


@cli.command(name="models")
@_handle_errors
def available_models():
    """List all available model names in the registry."""
    from rich.table import Table
    from daydream.registry import list_available
    from pathlib import Path

    table = Table(show_header=True, header_style="bold")
    table.add_column("NAME", style="cyan")
    table.add_column("TARGET", style="dim")
    table.add_column("SOURCE")

    for family, variant, repo_id in list_available():
        short_name = family if variant == "default" else f"{family}:{variant}"
        source = "local" if Path(repo_id).expanduser().exists() else "huggingface"
        table.add_row(short_name, repo_id, source)

    Console().print(table)
