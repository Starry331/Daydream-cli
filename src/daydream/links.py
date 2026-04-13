"""External tool integrations for `daydream link`."""

from __future__ import annotations

import shutil
import subprocess
import sys


def _is_installed(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _npm_global_installed(package: str) -> bool:
    try:
        result = subprocess.run(
            ["npm", "list", "-g", "--depth=0", package],
            capture_output=True, text=True, timeout=5,
        )
        return package in result.stdout
    except Exception:
        return False


TOOLS: list[dict] = [
    {
        "key": "chat",
        "name": "Chat with a model",
        "description": "Start an interactive chat with a local model",
        "check": lambda: True,
        "status": lambda: "ready",
        "run": None,  # handled specially in CLI
    },
    {
        "key": "claude",
        "name": "Launch Claude Code",
        "description": "Anthropic's agentic coding tool",
        "check": lambda: _is_installed("claude"),
        "install": "npm install -g @anthropic-ai/claude-code",
        "cmd": "claude",
    },
    {
        "key": "codex",
        "name": "Launch Codex",
        "description": "OpenAI's open-source coding agent",
        "check": lambda: _is_installed("codex"),
        "install": "npm install -g @openai/codex",
        "cmd": "codex",
    },
    {
        "key": "openclaw",
        "name": "Launch OpenClaw",
        "description": "Personal AI with 100+ skills",
        "check": lambda: _is_installed("openclaw"),
        "install": "pip install openclaw",
        "cmd": "openclaw",
    },
    {
        "key": "opencode",
        "name": "Launch OpenCode",
        "description": "Anomaly's open-source coding agent",
        "check": lambda: _is_installed("opencode"),
        "install": "go install github.com/opencode-ai/opencode@latest",
        "cmd": "opencode",
    },
    {
        "key": "aider",
        "name": "Launch Aider",
        "description": "AI pair programming in the terminal",
        "check": lambda: _is_installed("aider"),
        "install": "pip install aider-chat",
        "cmd": "aider",
    },
]


def get_menu_items() -> list[dict]:
    """Build menu items with live installation status."""
    items = []
    for tool in TOOLS:
        installed = tool["check"]()
        if tool.get("status"):
            status = tool["status"]()
        elif installed:
            status = "ready"
        elif tool.get("install"):
            status = "install"
        else:
            status = "not_installed"

        label = tool["name"]
        if not installed and tool.get("install"):
            label += " (not installed)"

        items.append({
            "key": tool["key"],
            "name": label,
            "description": tool["description"],
            "status": status,
            "installed": installed,
            "install_cmd": tool.get("install"),
            "cmd": tool.get("cmd"),
        })
    return items


def launch_tool(item: dict) -> None:
    """Launch or install+launch a tool."""
    if not item.get("cmd"):
        return

    if not item["installed"] and item.get("install_cmd"):
        from rich.console import Console
        console = Console()
        console.print(f"  Installing with: [bold]{item['install_cmd']}[/]")
        ret = subprocess.call(item["install_cmd"], shell=True)
        if ret != 0:
            console.print("[red]Installation failed.[/]")
            return
        console.print(f"[green]✓[/] Installed {item['cmd']}")

    import os
    os.execvp(item["cmd"], [item["cmd"]])
