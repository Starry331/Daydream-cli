import sys

import click
from click.core import ParameterSource
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


def _coalesce_model_reference(option_model, model_parts):
    if option_model and model_parts:
        raise ValueError("Use either a positional model or --model, not both.")
    if option_model:
        return option_model
    if not model_parts:
        return None

    parts = [part.strip() for part in model_parts if part and part.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    if all("/" not in part for part in parts):
        family = "-".join(parts[:-1])
        variant = parts[-1]
        return f"{family}:{variant}"

    return " ".join(parts)


def _parameter_was_explicit(ctx: click.Context, name: str) -> bool:
    try:
        return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
    except Exception:
        return False


def _resolve_profile_reference(model: str):
    from daydream.profiles import get_profile

    profile = get_profile(model)
    if profile is None:
        return model, None
    return profile.from_model, profile


@click.group()
@click.version_option(version=__version__, prog_name="daydream")
def cli():
    """Daydream — Apple Silicon local model CLI, powered by MLX."""
    from daydream.config import ensure_home

    ensure_home()


@cli.command()
@click.argument("name")
@click.option("-f", "--file", "file_path", type=click.Path(exists=True, dir_okay=False, path_type=str), default="Daydreamfile", show_default=True, help="Path to a Daydreamfile")
@_handle_errors
def create(name, file_path):
    """Create a custom model profile from a Daydreamfile."""
    from daydream.profiles import create_profile

    profile = create_profile(name, file_path=file_path)
    Console().print(f"[green]✓[/] Created custom model [bold]{profile.name}[/] from [bold]{profile.from_model}[/]")


@cli.command()
@click.argument("model")
@click.argument("prompt", required=False, default=None)
@click.option("--temp", type=float, default=get_default_temp(), show_default=True, help="Sampling temperature")
@click.option("--top-p", type=float, default=get_default_top_p(), show_default=True, help="Nucleus sampling top-p")
@click.option("--max-tokens", "-m", type=int, default=get_default_max_tokens(), show_default=True, help="Max tokens to generate")
@click.option("--system", "-s", type=str, default=None, help="System prompt")
@click.option("--draft", "draft_mode", flag_value="force", default=None, help="Enable the built-in Qwen3.5 draft model")
@click.option("--no-draft", "draft_mode", flag_value="off", help="Disable draft-model acceleration")
@click.option("--verbose", "-v", is_flag=True, help="Show performance metrics")
@click.pass_context
@_handle_errors
def run(ctx, model, prompt, temp, top_p, max_tokens, system, draft_mode, verbose):
    """Run a model — interactive chat or one-shot generation."""
    from daydream.chat import run_chat, run_oneshot

    resolved_model, profile = _resolve_profile_reference(model)
    if profile is not None:
        if not _parameter_was_explicit(ctx, "temp") and "temperature" in profile.parameters:
            temp = float(profile.parameters["temperature"])
        if not _parameter_was_explicit(ctx, "top_p") and "top_p" in profile.parameters:
            top_p = float(profile.parameters["top_p"])
        if not _parameter_was_explicit(ctx, "max_tokens") and "max_tokens" in profile.parameters:
            max_tokens = int(profile.parameters["max_tokens"])
        if not _parameter_was_explicit(ctx, "system") and profile.system:
            system = profile.system
        initial_effort = str(profile.parameters.get("effort", "default"))
    else:
        initial_effort = "default"

    if prompt is not None or not sys.stdin.isatty():
        run_oneshot(
            resolved_model,
            prompt=prompt,
            temp=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            system=system,
            verbose=verbose,
            initial_effort=initial_effort,
            draft_mode=draft_mode,
            display_name=profile.name if profile else model,
        )
    else:
        run_chat(
            resolved_model,
            temp=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            system=system,
            verbose=verbose,
            initial_effort=initial_effort,
            draft_mode=draft_mode,
            display_name=profile.name if profile else model,
        )


@cli.command()
@click.argument("model")
@_handle_errors
def pull(model):
    """Download a model from HuggingFace."""
    from daydream.models import pull_model

    resolved_model, _ = _resolve_profile_reference(model)
    pull_model(resolved_model)


@cli.command()
@click.argument("source", required=False, default=None)
@click.argument("destination", required=False, default=None)
@_handle_errors
def cp(source, destination):
    """Create an alias for a model."""
    from daydream.registry import copy_alias

    if source and destination:
        repo_id = copy_alias(source, destination)
        Console().print(f"[green]✓[/] Alias [bold]{destination}[/] -> {repo_id}")
        return

    from daydream.interactive import interactive_select, interactive_input
    from daydream.models import downloaded_models

    models = downloaded_models()
    if not models:
        err_console.print("[red]Error:[/] No downloaded models found.")
        raise SystemExit(1)

    selected = interactive_select("Select a model:", models)
    if selected is None:
        return

    short_name, repo_id = selected
    alias = interactive_input(f"Enter alias for [bold]{short_name}[/]:")
    if not alias:
        return

    copy_alias(short_name, alias)
    Console().print(f"[green]✓[/] Alias [bold]{alias}[/] -> {repo_id}")


@cli.command()
@click.argument("alias", required=False, default=None)
@_handle_errors
def unalias(alias):
    """Remove a model alias."""
    from daydream.registry import list_user_aliases, remove_alias

    if alias:
        repo_id = remove_alias(alias)
        Console().print(f"[green]✓[/] Removed alias [bold]{alias}[/] (was -> {repo_id})")
        return

    from daydream.interactive import interactive_select

    aliases = list_user_aliases()
    if not aliases:
        err_console.print("[dim]No user-defined aliases found.[/dim]")
        return

    selected = interactive_select("Select alias to remove:", aliases)
    if selected is None:
        return

    alias_name, repo_id = selected
    remove_alias(alias_name)
    Console().print(f"[green]✓[/] Removed alias [bold]{alias_name}[/] (was -> {repo_id})")


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
    from daydream.profiles import delete_profile
    from daydream.models import remove_model

    if delete_profile(model):
        Console().print(f"[green]✓[/] Removed custom model [bold]{model}[/]")
        return
    remove_model(model, force=force)


@cli.command()
@click.argument("model")
@_handle_errors
def show(model):
    """Show model information."""
    from daydream.profiles import get_profile, show_profile
    from daydream.models import show_model

    if get_profile(model) is not None:
        show_profile(model)
        return
    show_model(model)


@cli.command()
@click.argument("model_parts", nargs=-1, required=False)
@click.option("--model", type=str, default=None, help="Model to preload")
@click.option("--host", default=get_default_host(), show_default=True, help="Bind address")
@click.option("--port", "-p", type=int, default=get_default_port(), show_default=True, help="Port number")
@click.option("--max-tokens", type=int, default=None, help="Max tokens per response")
@click.option("--draft", "draft_mode", flag_value="force", default=None, help="Force-enable draft-model acceleration")
@click.option("--no-draft", "draft_mode", flag_value="off", help="Disable draft-model acceleration")
@click.option("--background", is_flag=True, help="Run server in the background")
@click.option("--foreground", is_flag=True, hidden=True)
@click.option("--detach", is_flag=True, hidden=True)
@click.pass_context
@_handle_errors
def serve(ctx, model_parts, model, host, port, max_tokens, draft_mode, background, foreground, detach):
    """Start or manage the OpenAI-compatible API server."""
    from daydream.server import start_server
    resolved_model = _coalesce_model_reference(model, model_parts)
    kwargs = dict(
        model=resolved_model,
        host=host,
        port=port,
        detach=background or detach,
        draft_mode=draft_mode,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    start_server(**kwargs)


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
    from daydream.profiles import list_profiles
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

    for profile in list_profiles():
        table.add_row(profile.name, profile.from_model, "profile")

    Console().print(table)
