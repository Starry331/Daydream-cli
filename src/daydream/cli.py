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


_SPECULATIVE_TO_DRAFT_MODE: dict[str, str | None] = {
    # --speculative METHOD → the internal `draft_mode` token that the
    # chat/server layers already understand. "auto" → None (= "no
    # explicit user choice, let the family default apply").
    "auto": None,
    "mtp": "mtp",
    "draft": "force",
    "lookup": "lookup",
    "none": "off",
}


def _reconcile_speculative_flags(speculative_method: str | None, draft_mode: str | None) -> str | None:
    """Translate the new --speculative METHOD into the existing
    `draft_mode` token. The explicit --speculative wins if set;
    otherwise the older --draft / --no-draft / --lookup aliases
    survive untouched.
    """
    if speculative_method is None:
        return draft_mode
    normalized = speculative_method.strip().lower()
    if normalized not in _SPECULATIVE_TO_DRAFT_MODE:
        return draft_mode
    return _SPECULATIVE_TO_DRAFT_MODE[normalized]


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
@click.option(
    "--speculative",
    "speculative_method",
    type=click.Choice(["auto", "mtp", "draft", "lookup", "none"], case_sensitive=False),
    default=None,
    help=(
        "Speculative decoding method. auto=picks best for model; "
        "mtp=model-native MTP/NextN (Qwen3.6, Phase 2); draft=small "
        "external draft model; lookup=prompt-ngram, no extra model; "
        "none=disable."
    ),
)
@click.option("--draft", "draft_mode", flag_value="force", default=None, help="Alias for --speculative draft")
@click.option("--no-draft", "draft_mode", flag_value="off", help="Alias for --speculative none")
@click.option("--lookup", "draft_mode", flag_value="lookup", help="Alias for --speculative lookup")
@click.option(
    "--draft-model",
    "draft_model_override",
    type=str,
    default=None,
    help="Override the draft repo (e.g. mlx-community/Qwen3.5-0.8B-MLX-4bit, mlx-community/Qwen3.5-2B-4bit)",
)
@click.option("--num-draft-tokens", "num_draft_tokens_override", type=int, default=None, help="Override draft tokens per step (default: 2-3 depending on family)")
@click.option(
    "--context-length",
    "-c",
    "context_length",
    type=str,
    default=None,
    help="Context window: default (auto) / 4k / 8k / 16k / 32k / 64k / 128k / custom int",
)
@click.option("--verbose", "-v", is_flag=True, help="Show performance metrics")
@click.pass_context
@_handle_errors
def run(ctx, model, prompt, temp, top_p, max_tokens, system, speculative_method, draft_mode, draft_model_override, num_draft_tokens_override, context_length, verbose):
    """Run a model — interactive chat or one-shot generation."""
    from daydream.chat import run_chat, run_oneshot
    from daydream.config import set_default_context_length

    # Reconcile --speculative METHOD with the older --draft / --no-draft
    # / --lookup aliases. The explicit --speculative wins if set.
    draft_mode = _reconcile_speculative_flags(speculative_method, draft_mode)

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

    # Persist the --context-length flag so subsequent runs (and the
    # chat REPL's /context display) reflect the user's choice.
    if context_length is not None:
        set_default_context_length(context_length)

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
            draft_model_override=draft_model_override,
            num_draft_tokens_override=num_draft_tokens_override,
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
            draft_model_override=draft_model_override,
            num_draft_tokens_override=num_draft_tokens_override,
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
@click.option(
    "--speculative",
    "speculative_method",
    type=click.Choice(["auto", "mtp", "draft", "lookup", "none"], case_sensitive=False),
    default=None,
    help="Speculative decoding method (see `daydream run --help` for details)",
)
@click.option("--draft", "draft_mode", flag_value="force", default=None, help="Alias for --speculative draft")
@click.option("--no-draft", "draft_mode", flag_value="off", help="Alias for --speculative none")
@click.option("--lookup", "draft_mode", flag_value="lookup", help="Alias for --speculative lookup")
@click.option(
    "--draft-model",
    "draft_model_override",
    type=str,
    default=None,
    help="Override the draft repo (e.g. mlx-community/Qwen3.5-0.8B-MLX-4bit, mlx-community/Qwen3.5-2B-4bit)",
)
@click.option("--num-draft-tokens", "num_draft_tokens_override", type=int, default=None, help="Override draft tokens per step")
@click.option(
    "--context-length",
    "-c",
    "context_length",
    type=str,
    default=None,
    help="Context window: default (auto) / 4k / 8k / 16k / 32k / 64k / 128k / custom int",
)
@click.option("--background", is_flag=True, help="Run server in the background")
@click.option("--foreground", is_flag=True, hidden=True)
@click.option("--detach", is_flag=True, hidden=True)
@click.pass_context
@_handle_errors
def serve(ctx, model_parts, model, host, port, max_tokens, speculative_method, draft_mode, draft_model_override, num_draft_tokens_override, context_length, background, foreground, detach):
    """Start or manage the OpenAI-compatible API server."""
    from daydream.config import set_default_context_length
    from daydream.server import start_server

    draft_mode = _reconcile_speculative_flags(speculative_method, draft_mode)

    if context_length is not None:
        set_default_context_length(context_length)

    resolved_model = _coalesce_model_reference(model, model_parts)
    kwargs = dict(
        model=resolved_model,
        host=host,
        port=port,
        detach=background or detach,
        draft_mode=draft_mode,
        draft_model_override=draft_model_override,
        num_draft_tokens_override=num_draft_tokens_override,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    start_server(**kwargs)


# ── daydream mtp install / uninstall / status ──────────────────────
@cli.group()
def mtp():
    """Manage the optional Qwen3.6 MTP / NextN acceleration."""


@mtp.command("install")
@click.option(
    "--trunk",
    default=None,
    help="Base trunk repo to attach MTP to (default: mlx-community/Qwen3.6-27B-4bit)",
)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Download the full ~16 GB MTPLX repo (bundled trunk + MTP head). "
         "Use this when the thin install fails with a quantization-shape mismatch.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Reinstall over an existing install.",
)
@_handle_errors
def mtp_install(trunk, full, force):
    """Install MTP head. Thin (~338 MB, symlinked trunk) by default; --full pulls everything."""
    from daydream.backends.mtp_install import (
        DEFAULT_BASE_TRUNK,
        DEFAULT_SIDECAR_REPO,
        THIN_INSTALL_DOWNLOAD_BYTES,
        find_thin_install,
        thin_install,
        thin_uninstall,
    )
    from daydream.models import is_model_available_locally, pull_model
    from daydream.utils import format_size

    base_trunk = trunk or DEFAULT_BASE_TRUNK
    console = Console()

    existing = find_thin_install(base_trunk)
    if existing is not None and not existing.broken_trunk and not force:
        console.print(
            f"[green]✓[/] MTP already installed at [bold]{existing.install_dir}[/]"
        )
        console.print(
            f"[dim]Download: {format_size(existing.bytes_downloaded)} · "
            f"Trunk symlinks: {format_size(existing.bytes_symlinked)}.[/dim]"
        )
        console.print("[dim]Use --force to reinstall.[/dim]")
        return
    if force and existing is not None:
        thin_uninstall(base_trunk=base_trunk)

    if not full and not is_model_available_locally(base_trunk):
        console.print(f"[yellow]Trunk `{base_trunk}` not cached.[/]")
        console.print("[dim]Pulling it first (one-time, ~16 GB)...[/dim]")
        pull_model(base_trunk, register_alias=True)

    if full:
        console.print(
            f"Full install from [cyan]{DEFAULT_SIDECAR_REPO}[/] [dim](~16 GB)[/dim]"
        )
    else:
        size = format_size(THIN_INSTALL_DOWNLOAD_BYTES)
        console.print(
            f"Thin install from [cyan]{DEFAULT_SIDECAR_REPO}[/] [dim](~{size})[/dim]"
        )
    install = thin_install(base_trunk=base_trunk, full=full)
    console.print(f"[green]✓[/] Installed at [bold]{install.install_dir}[/]")
    console.print(f"[dim]Run: daydream run {base_trunk} --speculative mtp[/dim]")


@mtp.command("uninstall")
@click.option(
    "--trunk",
    default=None,
    help="Base trunk repo whose MTP attachment to remove",
)
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt")
@_handle_errors
def mtp_uninstall(trunk, yes):
    """Remove the MTP head download (frees ~338 MB; the trunk is kept)."""
    from daydream.backends.mtp_install import (
        DEFAULT_BASE_TRUNK,
        find_thin_install,
        thin_uninstall,
    )
    from daydream.utils import format_size

    base_trunk = trunk or DEFAULT_BASE_TRUNK
    console = Console()

    existing = find_thin_install(base_trunk)
    if existing is None:
        console.print(f"[dim]No MTP install found for `{base_trunk}`.[/dim]")
        return

    if not yes:
        if not click.confirm(
            f"Remove MTP install at {existing.install_dir} "
            f"({format_size(existing.bytes_downloaded)} freed; trunk kept)?",
            default=False,
        ):
            console.print("[dim]Cancelled.[/dim]")
            return

    freed = thin_uninstall(base_trunk=base_trunk)
    console.print(
        f"[green]✓[/] Removed MTP install · freed [bold]{format_size(freed)}[/]"
    )


@mtp.command("status")
@_handle_errors
def mtp_status():
    """Show whether MTP is installed locally."""
    from daydream.backends.mtp import mtplx_available, mtplx_version
    from daydream.backends.mtp_install import (
        DEFAULT_BASE_TRUNK,
        find_thin_install,
    )
    from daydream.utils import format_size

    console = Console()

    if not mtplx_available():
        console.print("[yellow]MTP runtime (mtplx): NOT importable in this venv[/]")
        console.print(
            "[dim]mtplx is bundled with Daydream on Apple Silicon. "
            "If you're not on Apple Silicon, MTP is unavailable. "
            "Otherwise reinstall: pip install --force-reinstall daydream[/dim]"
        )
    else:
        console.print(f"[green]✓[/] MTP runtime [bold]mtplx {mtplx_version()}[/] (bundled)")

    install = find_thin_install(DEFAULT_BASE_TRUNK)
    if install is None:
        console.print("[yellow]MTP head: not installed[/]")
        console.print("[dim]Install with: daydream mtp install[/dim]")
        return

    console.print(f"[green]✓[/] MTP head installed at [bold]{install.install_dir}[/]")
    console.print(f"  [dim]Base trunk:[/]     {install.base_trunk}")
    console.print(f"  [dim]Sidecar repo:[/]   {install.sidecar_repo}")
    console.print(f"  [dim]Downloaded:[/]     {format_size(install.bytes_downloaded)}")
    console.print(f"  [dim]Trunk symlinks:[/] {format_size(install.bytes_symlinked)} (shared with trunk)")
    if install.broken_trunk:
        console.print(
            "[red]⚠[/]  [yellow]Trunk symlinks are broken[/] "
            "[dim](the base trunk was removed). Run "
            "`daydream pull " + install.base_trunk + "` to repair, or "
            "`daydream mtp uninstall` to clean up.[/dim]"
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


@cli.command()
@_handle_errors
def link():
    """Launch or install linked AI tools."""
    from daydream.interactive import interactive_menu, interactive_select
    from daydream.links import get_menu_items, launch_tool

    items = get_menu_items()
    selected = interactive_menu("Daydream Link", items)
    if selected is None:
        return

    if selected["key"] == "chat":
        from daydream.models import downloaded_models
        models = downloaded_models()
        if not models:
            err_console.print("[red]Error:[/] No downloaded models found. Run [bold]daydream pull <model>[/bold] first.")
            raise SystemExit(1)
        model = interactive_select("Select a model:", models)
        if model is None:
            return
        short_name, _ = model
        from daydream.chat import run_chat
        run_chat(short_name)
        return

    launch_tool(selected)


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
