# Daydream CLI

Daydream is a local Apple Silicon model CLI built on top of `mlx-lm`.

The goal is simple:

- Ollama-style CLI UX
- MLX-native inference
- Hugging Face model flow
- OpenAI-compatible local API for agents and coding tools

Daydream is intentionally narrow. It focuses on quantized MLX models, local terminal workflows, and a small command surface.

## Status

Current release: `v0.1.0`

Implemented commands:

- `daydream run`
- `daydream pull`
- `daydream list`
- `daydream rm`
- `daydream show`
- `daydream serve`
- `daydream ps`
- `daydream stop`
- `daydream models`

## Requirements

- Apple Silicon Mac
- Python `3.14+`
- MLX-compatible environment

Daydream only supports quantized MLX models.

Not supported:

- GGUF models
- non-quantized MLX model repos

## Installation

Clone the repo:

```bash
git clone https://github.com/Starry331/Daydream-cli.git
cd Daydream-cli
```

Create a virtual environment and install dependencies:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

Verify the CLI:

```bash
daydream --help
```

If you do not want to activate the venv, use:

```bash
./.venv/bin/daydream --help
```

## Quick Start

Run a model directly from Hugging Face:

```bash
./.venv/bin/daydream run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

Daydream will:

1. resolve the `hf.co/...` reference
2. auto-download the model if needed
3. verify it is a quantized MLX model
4. register a local alias
5. start the chat session

Single prompt:

```bash
./.venv/bin/daydream run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit "hello"
```

Pre-download only:

```bash
./.venv/bin/daydream pull hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

## Model Naming

Daydream accepts three kinds of model references.

### 1. Built-in short names

```bash
daydream run qwen3:8b
daydream run smollm2:135m
```

### 2. Hugging Face refs

```bash
daydream run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
daydream run mlx-community/Qwen3.5-9B-MLX-4bit
```

### 3. Local MLX model directories

```bash
daydream run /path/to/Qwen3.5-9B-MLX-4bit
```

Local directories are auto-registered the first time they are used.

## Chat Usage

Interactive chat:

```bash
daydream run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

Inside chat:

- `/help`
- `/reset`
- `/clear`
- `/quit`

### Multi-line input

Triple-quote block:

```text
>>> """
... explain this file:
... def add(a, b):
...     return a + b
... """
```

Backslash continuation:

```text
>>> explain this file\
... def add(a, b):\
...     return a + b
```

### Daydreaming animation

While the model is preparing its first output, Daydream shows a `Daydreaming` status animation inspired by coding-agent "Working" indicators.

## Background Server

`daydream serve` now defaults to background mode.

Start the local API server:

```bash
daydream serve --model hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

Check status:

```bash
daydream ps
```

Stop the managed server:

```bash
daydream stop
```

Force stop if needed:

```bash
daydream stop --force
```

Run the server in the foreground:

```bash
daydream serve --foreground --model hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

## OpenAI-Compatible API

Default endpoint:

```text
http://127.0.0.1:11434
```

Example request:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

This is intended for:

- local agents
- coding tools
- OpenAI-compatible clients

## Listing And Inspecting Models

List downloaded models:

```bash
daydream list
```

List built-in and discovered model names:

```bash
daydream models
```

Show model details:

```bash
daydream show qwen3:8b
daydream show hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

Remove a downloaded model from cache:

```bash
daydream rm qwen3:8b
```

## Local Registry

User config lives under:

- `~/.daydream/config.yaml`
- `~/.daydream/registry.yaml`

Model aliases can be added manually:

```yaml
qwen3.5:
  9b: mlx-community/Qwen3.5-9B-MLX-4bit
```

Then run:

```bash
daydream run qwen3.5:9b
```

Local model scan roots:

- default: `~/.daydream/models/`
- optional override: `DAYDREAM_MODELS_DIRS`

## Environment Variables

- `DAYDREAM_HOME`
- `DAYDREAM_CACHE_DIR`
- `DAYDREAM_LOCAL_MODELS_DIR`
- `DAYDREAM_MODELS_DIRS`
- `HF_HOME`
- `HF_HUB_CACHE`

Example:

```bash
export DAYDREAM_HOME=/tmp/daydream-home
export DAYDREAM_CACHE_DIR=/tmp/daydream-cache
```

## Common Workflows

### Terminal-only local chat

```bash
daydream run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

### Start background API once

```bash
daydream serve --model hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
daydream ps
```

### Pipe content into a model

```bash
cat file.py | daydream run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
```

### Use a local model directory

```bash
daydream run /path/to/My-Model-MLX-4bit
```

## Error Cases

GGUF is rejected immediately:

```bash
daydream run hf.co/bartowski/Qwen3.5-14B-GGUF
```

Expected result:

```text
GGUF models are not supported. Use a quantized MLX model instead.
```

Non-quantized MLX repos are also rejected.

## Development

Run tests:

```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

Compile-check the source tree:

```bash
PYTHONPYCACHEPREFIX=/tmp/daydream-pyc PYTHONPATH=src ./.venv/bin/python -m compileall src tests
```

## License

No license file is included yet.
