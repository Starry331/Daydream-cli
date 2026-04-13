"""OpenAI-compatible API server wrapping mlx-lm, and ps status check."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rich.console import Console

from daydream import engine
from daydream.chat import (
    _ReasoningParser,
    _clean_final_output_from_reasoning_leak,
    _extract_answer_labeled_text,
    _finalize_output_text,
    _recover_final_output,
)
from daydream.config import (
    SERVER_LOG_FILE,
    SERVER_STATE_FILE,
    ensure_home,
    get_default_host,
    get_default_max_tokens,
    get_default_port,
    get_default_temp,
    get_default_top_p,
)
from daydream.models import ensure_runtime_model, is_fixture_model, is_model_available_locally, pull_model
from daydream.profiles import get_profile
from daydream.registry import reverse_lookup
from daydream.speculative import (
    default_draft_for_model,
    default_num_draft_tokens,
    draft_model_for_family,
)

console = Console()


def _finalize_runtime_response_text(
    candidate_text: str,
    parser: _ReasoningParser,
) -> tuple[str, str]:
    combined = "\n".join(
        part.strip()
        for part in (candidate_text, parser.reasoning_text)
        if part and part.strip()
    )
    labeled = _extract_answer_labeled_text(combined)
    if labeled:
        content = labeled.lstrip("* ").strip()
        reasoning = combined
        for marker in (
            "*Final Answer:*",
            "*Answer:*",
            "*Response:*",
            "Final Answer:",
            "Final Answer：",
            "Answer:",
            "Answer：",
            "Response:",
            "Response：",
            "回答：",
            "回答:",
            "答案：",
            "答案:",
            "回复：",
            "回复:",
        ):
            reasoning = reasoning.replace(f"{marker} {content}", "")
            reasoning = reasoning.replace(f"{marker}{content}", "")
        if content and reasoning.strip().endswith(content):
            reasoning = reasoning[: reasoning.rfind(content)].rstrip()
        return content, reasoning.strip()
    content = _finalize_output_text(candidate_text, candidate_text, parser)
    if not content and parser.saw_reasoning:
        content = _clean_final_output_from_reasoning_leak(parser.reasoning_text).strip()
    content = content.lstrip("\n")
    if parser.saw_reasoning and content:
        content = content.rstrip()
    return content, parser.reasoning_text.strip()


def _iter_sse_text_chunks(text: str, *, chunk_size: int = 24):
    if not text:
        return
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for line in normalized.split("\n"):
        if not line:
            yield "\n"
            continue
        start = 0
        while start < len(line):
            end = min(len(line), start + chunk_size)
            yield line[start:end]
            start = end
        yield "\n"


def _api_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def _print_client_setup(
    host: str,
    port: int,
    repo_id: str | None,
    *,
    profile_name: str | None = None,
    draft_model: str | None = None,
    num_draft_tokens: int | None = None,
) -> None:
    console.print("[bold]Client setup[/]")
    console.print(f"  Base URL: {_api_base_url(host, port)}")
    console.print("  API key:  daydream-local  [dim](any non-empty value works)[/dim]")
    if profile_name:
        console.print(f"  Custom:   {profile_name}  [dim](Daydream profile defaults are active)[/dim]")
    if repo_id:
        console.print(f"  Model:    {repo_id}")
        short = reverse_lookup(repo_id)
        if short and short != repo_id:
            if draft_model:
                token_note = f" · {num_draft_tokens} draft tokens" if num_draft_tokens else ""
                console.print(
                    f"  Alias:    {short}  [dim](Daydream CLI only · draft acceleration active: {draft_model}{token_note})[/dim]"
                )
            else:
                console.print(f"  Alias:    {short}  [dim](Daydream CLI only)[/dim]")
        elif draft_model:
            token_note = f" · {num_draft_tokens} draft tokens" if num_draft_tokens else ""
            console.print(
                f"  Draft:    {draft_model}  [dim](speculative acceleration active{token_note})[/dim]"
            )
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
    max_tokens: int = 4096,
    chat_template_args: dict | None = None,
    draft_model=None,
    num_draft_tokens: int = 6,
) -> argparse.Namespace:
    """Build the argparse.Namespace that mlx_lm.server.ModelProvider expects."""
    return argparse.Namespace(
        model=model,
        adapter_path=None,
        host=host,
        port=port,
        allowed_origins=["*"],
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
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
                self._send_json({"status": "ok", "model": repo_id})
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
    server.allow_reuse_address = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _run_runtime_server(
    host: str,
    port: int,
    repo_id: str,
    *,
    model,
    tokenizer,
    draft_model=None,
    temp: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    chat_template_args: dict | None = None,
) -> None:
    generation_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse_chunk(self, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.wfile.write(b"data: " + data + b"\n\n")
            self.wfile.flush()

        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"status": "ok", "model": repo_id})
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
            if not isinstance(messages, list):
                self._send_json({"error": "messages must be a list"}, status=400)
                return

            stream = bool(payload.get("stream", False))
            request_temp = float(payload.get("temperature", temp))
            request_top_p = float(payload.get("top_p", top_p))
            request_max_tokens = int(payload.get("max_tokens", max_tokens))

            prompt_cache = engine.create_prompt_cache(model, draft_model)
            parser = _ReasoningParser()
            chunks: list[str] = []
            last_response = None

            with generation_lock:
                response_iter = engine.generate_stream(
                    model,
                    tokenizer,
                    messages,
                    max_tokens=request_max_tokens,
                    temp=request_temp,
                    top_p=request_top_p,
                    chat_template_kwargs=chat_template_args or {},
                    draft_model=draft_model,
                    num_draft_tokens=default_num_draft_tokens(repo_id) if draft_model is not None else None,
                    prefill_step_size=2048,
                    prompt_cache=prompt_cache,
                )

                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    first_chunk = True
                    try:
                        for response in response_iter:
                            last_response = response
                            raw_chunk = response.text or ""
                            text, reasoning_text, _ = parser.feed(raw_chunk)

                            delta: dict = {}
                            if first_chunk:
                                delta["role"] = "assistant"
                                first_chunk = False
                            if reasoning_text:
                                delta["reasoning_content"] = reasoning_text
                            if text:
                                delta["content"] = text

                            if delta:
                                self._send_sse_chunk(
                                    {
                                        "id": "chatcmpl-daydream-stream",
                                        "object": "chat.completion.chunk",
                                        "created": 0,
                                        "model": repo_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": delta,
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                    except (BrokenPipeError, ConnectionResetError):
                        return

                    # Flush remaining parser buffer
                    flush_visible, flush_reasoning, _ = parser.flush()
                    if flush_visible or flush_reasoning:
                        flush_delta: dict = {}
                        if flush_reasoning:
                            flush_delta["reasoning_content"] = flush_reasoning
                        if flush_visible:
                            flush_delta["content"] = flush_visible
                        if flush_delta:
                            try:
                                self._send_sse_chunk(
                                    {
                                        "id": "chatcmpl-daydream-stream",
                                        "object": "chat.completion.chunk",
                                        "created": 0,
                                        "model": repo_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": flush_delta,
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            except (BrokenPipeError, ConnectionResetError):
                                return

                    finish_reason = getattr(last_response, "finish_reason", "stop") if last_response else "stop"
                    try:
                        self._send_sse_chunk(
                            {
                                "id": "chatcmpl-daydream-stream",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": repo_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish_reason,
                                    }
                                ],
                            }
                        )
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return

                for response in response_iter:
                    last_response = response
                    raw_chunk = response.text or ""
                    text, _, _ = parser.feed(raw_chunk)
                    if text:
                        chunks.append(text)

                # Flush remaining parser buffer
                flush_visible, _, _ = parser.flush()
                if flush_visible:
                    chunks.append(flush_visible)

            full_text, reasoning_text = _finalize_runtime_response_text("".join(chunks), parser)
            prompt_tokens = getattr(last_response, "prompt_tokens", 0) or 0
            generation_tokens = getattr(last_response, "generation_tokens", 0) or 0
            message = {"role": "assistant", "content": full_text}
            if reasoning_text:
                message["reasoning"] = reasoning_text
            self._send_json(
                {
                    "id": "chatcmpl-daydream",
                    "object": "chat.completion",
                    "created": 0,
                    "model": repo_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": getattr(last_response, "finish_reason", "stop") if last_response else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": generation_tokens,
                        "total_tokens": prompt_tokens + generation_tokens,
                    },
                }
            )

    server = ThreadingHTTPServer((host, port), Handler)
    server.allow_reuse_address = True
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
    draft_model: str | None,
    num_draft_tokens: int | None,
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
        "draft_model": draft_model,
        "num_draft_tokens": num_draft_tokens,
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


def _spawn_background_server(
    *,
    model: str | None,
    host: str,
    port: int,
    max_tokens: int | None = None,
    draft_model: str | None = None,
    num_draft_tokens: int | None = None,
    draft_mode: str | None = None,
) -> None:
    ensure_home()
    if _is_server_healthy(host, port, timeout=0.5):
        console.print(f"[yellow]Server already running on http://{host}:{port}[/]")
        show_status()
        return

    command = [sys.executable, "-m", "daydream", "serve", "--foreground", "--host", host, "--port", str(port)]
    if model:
        command.extend(["--model", model])
    if max_tokens is not None:
        command.extend(["--max-tokens", str(max_tokens)])
    if draft_mode == "off":
        command.append("--no-draft")
    elif draft_mode == "force":
        command.append("--draft")

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
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        log_file=str(SERVER_LOG_FILE),
        profile=profile_name,
    )

    deadline = time.time() + 15.0
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


def start_server(
    *,
    model=None,
    host="127.0.0.1",
    port=11434,
    detach: bool = False,
    max_tokens: int | None = None,
    draft_mode: str | None = None,
) -> None:
    """Start the OpenAI-compatible API server."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    profile = get_profile(model) if model else None
    source_model = profile.from_model if profile else model
    repo_id = ensure_runtime_model(source_model, auto_pull=True, register_alias=True) if source_model else None
    draft_repo_id = None
    num_draft_tokens = None
    draft_disabled_reason = None
    if repo_id and draft_mode == "force":
        candidate_draft = draft_model_for_family(repo_id)
        if candidate_draft:
            if not is_model_available_locally(candidate_draft):
                pull_model(candidate_draft, register_alias=True)
            draft_repo_id = candidate_draft
            num_draft_tokens = default_num_draft_tokens(repo_id) or 6
        else:
            console.print("[yellow]This model does not support draft acceleration.[/yellow]")

    if detach:
        _spawn_background_server(
            model=profile.name if profile else repo_id,
            host=host,
            port=port,
            max_tokens=max_tokens,
            draft_model=draft_repo_id,
            num_draft_tokens=num_draft_tokens,
            draft_mode=draft_mode,
        )
        return

    short = profile.name if profile else reverse_lookup(repo_id) or model if repo_id else None
    _write_server_state(
        pid=os.getpid(),
        host=host,
        port=port,
        model=repo_id,
        draft_model=draft_repo_id,
        num_draft_tokens=num_draft_tokens,
        log_file=None,
        profile=profile.name if profile else None,
    )

    console.print(f"[bold cyan]Daydream[/] server listening on [bold]http://{host}:{port}[/]")
    if short:
        console.print(f"Model: [bold]{short}[/] ({repo_id})")
    if draft_repo_id:
        console.print(f"Speculative: [dim]enabled[/dim] with {draft_repo_id} ({num_draft_tokens} draft tokens)")
    elif repo_id and draft_model_for_family(repo_id):
        if draft_disabled_reason:
            console.print(f"Speculative: [dim]disabled[/dim] ({draft_disabled_reason})")
        else:
            console.print("Speculative: [dim]disabled[/dim] (draft model not installed locally)")
    if profile and profile.system:
        console.print("[dim]System style in Daydreamfile is CLI-only. API clients should send their own system message.[/dim]")
    console.print(f"API:   [dim]{_api_base_url(host, port)}/chat/completions[/]")
    console.print()
    _print_client_setup(
        host,
        port,
        repo_id,
        profile_name=profile.name if profile else None,
        draft_model=draft_repo_id,
        num_draft_tokens=num_draft_tokens,
    )

    try:
        if repo_id and is_fixture_model(repo_id):
            _run_fixture_server(host, port, repo_id)
            return

        if repo_id and draft_repo_id is not None:
            loaded_model, loaded_tokenizer = engine.load_model(repo_id, ensure_available=False)
            loaded_draft_model, _ = engine.load_model(draft_repo_id, ensure_available=False)
            profile_parameters = profile.parameters if profile else {}
            chat_template_args = {}
            if profile_parameters.get("effort") == "instant":
                chat_template_args = {"enable_thinking": False}
            elif profile_parameters.get("effort") in {"short", "long"}:
                chat_template_args = {"enable_thinking": True}

            effective_max_tokens = max_tokens or int(profile_parameters.get("max_tokens", get_default_max_tokens()))
            _run_runtime_server(
                host,
                port,
                repo_id,
                model=loaded_model,
                tokenizer=loaded_tokenizer,
                draft_model=loaded_draft_model,
                temp=float(profile_parameters.get("temperature", get_default_temp())),
                top_p=float(profile_parameters.get("top_p", get_default_top_p())),
                max_tokens=effective_max_tokens,
                chat_template_args=chat_template_args,
            )
            return

        from mlx_lm.server import ModelProvider, run as mlx_server_run

        profile_parameters = profile.parameters if profile else {}
        chat_template_args = {}
        if profile_parameters.get("effort") == "instant":
            chat_template_args = {"enable_thinking": False}
        elif profile_parameters.get("effort") in {"short", "long"}:
            chat_template_args = {"enable_thinking": True}

        effective_max_tokens = max_tokens or int(profile_parameters.get("max_tokens", get_default_max_tokens()))

        def _serve_with_args(serve_draft, serve_num_tokens):
            args = _build_server_args(
                model=repo_id,
                host=host,
                port=port,
                temp=float(profile_parameters.get("temperature", get_default_temp())),
                top_p=float(profile_parameters.get("top_p", get_default_top_p())),
                max_tokens=effective_max_tokens,
                chat_template_args=chat_template_args,
                draft_model=serve_draft,
                num_draft_tokens=serve_num_tokens or 6,
            )
            provider = ModelProvider(args)
            mlx_server_run(host, port, provider)

        # First attempt: with draft model if available
        try:
            _serve_with_args(draft_repo_id, num_draft_tokens)
        except KeyboardInterrupt:
            console.print("\n[dim]Server stopped.[/dim]")
        except Exception as e:
            if draft_repo_id is None:
                # No draft was active — plain server failure
                console.print(f"[red]Server error: {e}[/red]")
                raise
            # Draft was active and server failed — disable draft and retry
            console.print(f"[yellow]Speculative decoding failed: {e}[/yellow]")
            console.print("[dim]Disabling draft model and restarting with main model only...[/dim]")
            _write_server_state(
                pid=os.getpid(), host=host, port=port, model=repo_id,
                draft_model=None, num_draft_tokens=None,
                log_file=None, profile=profile.name if profile else None,
            )
            try:
                _serve_with_args(None, None)
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
    draft_model = state.get("draft_model")
    if draft_model:
        console.print(f"  Draft: {draft_model}")

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
