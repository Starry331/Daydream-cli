"""OpenAI-compatible API server wrapping mlx-lm, and ps status check."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rich.console import Console

from daydream.config import (
    SERVER_LOG_FILE,
    SERVER_STATE_FILE,
    ensure_home,
    get_default_host,
    get_default_port,
)
from daydream.models import ensure_runtime_model, is_fixture_model
from daydream.profiles import get_profile
from daydream.registry import reverse_lookup

console = Console()


def _api_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def _print_client_setup(host: str, port: int, repo_id: str | None, *, profile_name: str | None = None) -> None:
    console.print("[bold]Client setup[/]")
    console.print(f"  Base URL: {_api_base_url(host, port)}")
    console.print("  API key:  daydream-local  [dim](any non-empty value works)[/dim]")
    if profile_name:
        console.print(f"  Custom:   {profile_name}  [dim](Daydream profile defaults are active)[/dim]")
    if repo_id:
        console.print(f"  Model:    {repo_id}")
        short = reverse_lookup(repo_id)
        if short and short != repo_id:
            console.print(f"  Alias:    {short}  [dim](Daydream CLI only)[/dim]")
    else:
        console.print("  Model:    choose one from /v1/models after the server is ready")
    console.print()


def _build_server_args(
    *,
    model=None,
    host="127.0.0.1",
    port=11434,
    temp: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    chat_template_args: dict | None = None,
) -> argparse.Namespace:
    """Build the argparse.Namespace that mlx_lm.server.ModelProvider expects."""
    return argparse.Namespace(
        model=model,
        adapter_path=None,
        host=host,
        port=port,
        allowed_origins=["*"],
        draft_model=None,
        num_draft_tokens=3,
        trust_remote_code=False,
        log_level="INFO",
        chat_template="",
        use_default_chat_template=False,
        temp=temp,
        top_p=top_p,
        top_k=0,
        min_p=0.0,
        max_tokens=max_tokens,
        chat_template_args=chat_template_args or {},
        decode_concurrency=32,
        prompt_concurrency=8,
        prefill_step_size=2048,
        prompt_cache_size=10,
        prompt_cache_bytes=None,
        pipeline=False,
    )


def _fixture_response_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).lower()
        if "hello" in content:
            return "Hello from Daydream."
        return "Daydream is running in offline fixture mode."
    return "Hello from Daydream."


def _run_fixture_server(host: str, port: int, repo_id: str) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"status": "ok"})
                return

            if self.path != "/v1/models":
                self._send_json({"error": "not found"}, status=404)
                return

            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": repo_id,
                            "object": "model",
                            "owned_by": "daydream",
                        }
                    ],
                }
            )

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json({"error": "not found"}, status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload.get("messages", [])
            text = _fixture_response_text(messages)
            self._send_json(
                {
                    "id": "chatcmpl-daydream-fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": repo_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": sum(len(str(m.get("content", "")).split()) for m in messages),
                        "completion_tokens": len(text.split()),
                        "total_tokens": sum(len(str(m.get("content", "")).split()) for m in messages) + len(text.split()),
                    },
                }
            )

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _build_opener():
    import urllib.request

    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _read_json_url(url: str, *, timeout: float = 2.0) -> dict | None:
    import urllib.error

    opener = _build_opener()
    try:
        with opener.open(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def _is_server_healthy(host: str, port: int, *, timeout: float = 2.0) -> bool:
    health_url = f"http://{host}:{port}/health"
    return _read_json_url(health_url, timeout=timeout) is not None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_server_state(
    *,
    pid: int,
    host: str,
    port: int,
    model: str | None,
    log_file: str | None,
    profile: str | None = None,
) -> None:
    ensure_home()
    previous = _load_server_state() or {}
    if log_file is None and previous.get("pid") == pid:
        log_file = previous.get("log_file")
    payload = {
        "pid": pid,
        "host": host,
        "port": port,
        "model": model,
        "profile": profile,
        "log_file": log_file,
        "started_at": time.time(),
    }
    SERVER_STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_server_state() -> dict | None:
    if not SERVER_STATE_FILE.exists():
        return None
    try:
        data = json.loads(SERVER_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _clear_server_state() -> None:
    try:
        SERVER_STATE_FILE.unlink()
    except FileNotFoundError:
        pass


def _spawn_background_server(*, model: str | None, host: str, port: int) -> None:
    ensure_home()
    if _is_server_healthy(host, port, timeout=0.5):
        console.print(f"[yellow]Server already running on http://{host}:{port}[/]")
        show_status()
        return

    command = [sys.executable, "-m", "daydream", "serve", "--foreground", "--host", host, "--port", str(port)]
    if model:
        command.extend(["--model", model])

    SERVER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SERVER_LOG_FILE.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
        )

    profile_obj = get_profile(model) if model else None
    profile_name = profile_obj.name if profile_obj else None
    _write_server_state(
        pid=process.pid,
        host=host,
        port=port,
        model=model,
        log_file=str(SERVER_LOG_FILE),
        profile=profile_name,
    )

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if _is_server_healthy(host, port, timeout=0.2):
            console.print(f"[green]✓[/] Server started in background (pid {process.pid})")
            console.print(f"[dim]Endpoint:[/] http://{host}:{port}")
            console.print(f"[dim]Log:[/]      {SERVER_LOG_FILE}")
            return

        if process.poll() is not None:
            _clear_server_state()
            raise ValueError(f"Background server exited immediately. Check {SERVER_LOG_FILE}")

        time.sleep(0.1)

    console.print(f"[green]✓[/] Server process started in background (pid {process.pid})")
    console.print(f"[dim]Endpoint:[/] http://{host}:{port}")
    console.print(f"[dim]Log:[/]      {SERVER_LOG_FILE}")
    console.print("[dim]Status:[/]   warming up")


def start_server(*, model=None, host="127.0.0.1", port=11434, detach: bool = False) -> None:
    """Start the OpenAI-compatible API server."""
    profile = get_profile(model) if model else None
    source_model = profile.from_model if profile else model
    repo_id = ensure_runtime_model(source_model, auto_pull=True, register_alias=True) if source_model else None

    if detach:
        _spawn_background_server(model=profile.name if profile else repo_id, host=host, port=port)
        return

    short = profile.name if profile else reverse_lookup(repo_id) or model if repo_id else None
    _write_server_state(
        pid=os.getpid(),
        host=host,
        port=port,
        model=repo_id,
        log_file=None,
        profile=profile.name if profile else None,
    )

    console.print(f"[bold cyan]Daydream[/] server listening on [bold]http://{host}:{port}[/]")
    if short:
        console.print(f"Model: [bold]{short}[/] ({repo_id})")
    if profile and profile.system:
        console.print("[dim]System style in Daydreamfile is CLI-only. API clients should send their own system message.[/dim]")
    console.print(f"API:   [dim]{_api_base_url(host, port)}/chat/completions[/]")
    console.print()
    _print_client_setup(host, port, repo_id, profile_name=profile.name if profile else None)

    try:
        if repo_id and is_fixture_model(repo_id):
            _run_fixture_server(host, port, repo_id)
            return

        from mlx_lm.server import ModelProvider, run as mlx_server_run

        profile_parameters = profile.parameters if profile else {}
        chat_template_args = {}
        if profile_parameters.get("effort") == "instant":
            chat_template_args = {"enable_thinking": False}
        elif profile_parameters.get("effort") in {"short", "long"}:
            chat_template_args = {"enable_thinking": True}

        args = _build_server_args(
            model=repo_id,
            host=host,
            port=port,
            temp=float(profile_parameters.get("temperature", 0.0)),
            top_p=float(profile_parameters.get("top_p", 1.0)),
            max_tokens=int(profile_parameters.get("max_tokens", 512)),
            chat_template_args=chat_template_args,
        )
        provider = ModelProvider(args)
        mlx_server_run(host, port, provider)
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")
    finally:
        state = _load_server_state()
        if state and state.get("pid") == os.getpid():
            _clear_server_state()


def show_status() -> None:
    """Check if a Daydream/mlx-lm server is running."""
    state = _load_server_state() or {}
    host = state.get("host") or get_default_host()
    port = int(state.get("port") or get_default_port())

    if not _is_server_healthy(host, port, timeout=2):
        pid = state.get("pid")
        if isinstance(pid, int) and _pid_is_running(pid):
            console.print(f"[yellow]◐[/] Server process running but not ready yet (pid {pid})")
            console.print(f"  Endpoint: http://{host}:{port}")
            log_file = state.get("log_file")
            if log_file:
                console.print(f"  Log: {log_file}")
            return

        _clear_server_state()
        console.print(f"[dim]No server running on {host}:{port}[/dim]")
        return

    console.print("[green]●[/] Server running")
    console.print(f"  Endpoint: http://{host}:{port}")
    console.print(f"  OpenAI:   {_api_base_url(host, port)}")
    pid = state.get("pid")
    if isinstance(pid, int):
        console.print(f"  PID: {pid}")
    log_file = state.get("log_file")
    if log_file:
        console.print(f"  Log: {log_file}")
    profile = state.get("profile")
    if profile:
        console.print(f"  Custom: {profile}")

    models_url = f"http://{host}:{port}/v1/models"
    data = _read_json_url(models_url, timeout=2) or {}
    models = data.get("data", [])
    if not models:
        console.print("  [dim]No model loaded[/dim]")
        return

    for model in models:
        model_id = model.get("id", "unknown")
        short = reverse_lookup(model_id) or model_id
        console.print(f"  Model: [bold]{short}[/] ({model_id})")
    console.print("  API key: any non-empty value")


def stop_server(*, force: bool = False) -> None:
    """Stop the managed background server."""
    state = _load_server_state() or {}
    pid = state.get("pid")
    host = state.get("host") or get_default_host()
    port = int(state.get("port") or get_default_port())

    if not isinstance(pid, int):
        if _is_server_healthy(host, port, timeout=0.5):
            raise ValueError(
                "A server is responding, but it is not managed by Daydream. Stop it manually."
            )
        console.print(f"[dim]No server running on {host}:{port}[/dim]")
        return

    if not _pid_is_running(pid):
        _clear_server_state()
        console.print(f"[dim]No server running on {host}:{port}[/dim]")
        return

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not _pid_is_running(pid):
            _clear_server_state()
            console.print(f"[green]✓[/] Stopped server on http://{host}:{port}")
            return
        time.sleep(0.1)

    if force:
        os.kill(pid, signal.SIGKILL)
        _clear_server_state()
        console.print(f"[green]✓[/] Force-stopped server on http://{host}:{port}")
        return

    raise ValueError(
        f"Server pid {pid} did not stop in time. Run `daydream stop --force`."
    )
