# Daydream CLI

English (default) | [中文](#chinese)

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

The examples in this README use Qwen, SmolLM, and short names as examples only. Daydream is not limited to Qwen3.5. It can run any quantized MLX model repo that `mlx-lm` can load.

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

After activation, you can use `daydream ...` directly.

If you do not want to activate the venv, use:

```bash
./.venv/bin/daydream --help
```

Optional: set a Hugging Face token for higher rate limits and faster downloads:

```bash
export HF_TOKEN=your_huggingface_token
```

Without `HF_TOKEN`, public models still download normally, but the Hub may apply lower anonymous rate limits.

## Quick Start

Run a model directly from Hugging Face:

```bash
daydream run hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
```

Daydream will:

1. resolve the `hf.co/...` reference
2. auto-download the model if needed
3. verify it is a quantized MLX model
4. register a local alias
5. start the chat session

Single prompt:

```bash
daydream run hf.co/mlx-community/SmolLM2-135M-Instruct-4bit "hello"
```

Pre-download only:

```bash
daydream pull hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
```

## Model Naming

Daydream accepts three kinds of model references.

### 1. Built-in short names

```bash
daydream run qwen3:8b
daydream run smollm2:135m
daydream run gemma3:4b
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

Example: if your MLX model directory is on the Desktop:

```bash
cd /Users/starry/Desktop/Daydreamcli
daydream run /Users/starry/Desktop/SmolLM2-135M-Instruct-4bit
```

If `/Users/starry/Desktop/Daydreamcli` is the project directory, pass the model directory path to `daydream run`, not the project path itself.

## Chat Usage

Interactive chat:

```bash
daydream run qwen3:8b
```

Inside chat:

- `/` opens the slash-command menu in the chat box
- `/effort` opens the reasoning-effort picker
- `/effort instant`
- `/effort short`
- `/effort default`
- `/effort long`
- `/help`
- `/reset`
- `/clear`
- `/t`
- `/quit`

### Slash commands and reasoning effort

When you type `/`, Daydream shows a dim inline command menu similar to coding-agent chat UIs.

`/effort` lets you manually adjust how much reasoning the model should use:

- `instant`
- `short`
- `default`
- `long`

`default` is the starting value.

If the current model likely supports reasoning-effort control, Daydream injects a small system hint for that turn. If the model does not support it, Daydream does not crash; it keeps the selected value and shows that the model may ignore it.

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

While the model is preparing its first output, Daydream shows a `Daydreaming` status animation inspired by coding-agent `Working` indicators.

## Server

`daydream serve` runs in the foreground by default.

Start the local API server in the foreground:

```bash
daydream serve --model hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
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

Run the server in the background:

```bash
daydream serve --background --model hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
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
    "model": "mlx-community/SmolLM2-135M-Instruct-4bit",
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
daydream show hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
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
daydream run qwen3:8b
```

### Start background API once

```bash
daydream serve --background --model hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
daydream ps
```

### Pipe content into a model

```bash
cat file.py | daydream run qwen3:8b
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

## Chinese

<details>
<summary>点击展开中文说明</summary>

### 项目简介

Daydream 是一个基于 `mlx-lm` 的 Apple Silicon 本地模型 CLI。

目标很明确：

- 提供接近 Ollama 的 CLI 使用体验
- 使用 MLX 原生推理
- 兼容 Hugging Face 模型工作流
- 提供 OpenAI 兼容本地 API，方便 agent 和 coding tools 接入

Daydream 目前只专注一件事：在本地终端和 agent 场景下，稳定运行量化过的 MLX 模型。

### 当前版本

当前 release：`v0.1.0`

已实现命令：

- `daydream run`
- `daydream pull`
- `daydream list`
- `daydream rm`
- `daydream show`
- `daydream serve`
- `daydream ps`
- `daydream stop`
- `daydream models`

### 环境要求

- Apple Silicon Mac
- Python `3.14+`
- 可运行 MLX 的本地环境

只支持量化过的 MLX 模型。

README 里的 Qwen、SmolLM 和短名称都只是示例，不代表 Daydream 只支持 Qwen3.5。只要是 `mlx-lm` 能加载的量化 MLX 模型仓库，Daydream 都可以运行。

不支持：

- GGUF
- 未量化的 MLX 模型仓库

### 安装

克隆仓库：

```bash
git clone https://github.com/Starry331/Daydream-cli.git
cd Daydream-cli
```

创建虚拟环境并安装：

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

验证：

```bash
daydream --help
```

激活虚拟环境后，就可以直接使用 `daydream ...`。

如果不激活虚拟环境，也可以直接：

```bash
./.venv/bin/daydream --help
```

可选：设置 Hugging Face token，可以获得更高限速和更快下载：

```bash
export HF_TOKEN=你的_huggingface_token
```

如果不设置 `HF_TOKEN`，公开模型通常也能正常下载，只是 Hugging Face 可能会对匿名请求施加更低的速率限制。

### 快速开始

直接从 Hugging Face 运行模型：

```bash
daydream run hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
```

Daydream 会自动：

1. 解析 `hf.co/...` 引用
2. 如果本地没有就自动下载
3. 校验是不是量化过的 MLX 模型
4. 自动注册本地别名
5. 启动聊天

单次提问：

```bash
daydream run hf.co/mlx-community/SmolLM2-135M-Instruct-4bit "hello"
```

只下载不运行：

```bash
daydream pull hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
```

### 模型引用方式

支持三种模型引用：

#### 1. 内置短名称

```bash
daydream run qwen3:8b
daydream run smollm2:135m
daydream run gemma3:4b
```

#### 2. Hugging Face 引用

```bash
daydream run hf.co/mlx-community/Qwen3.5-9B-MLX-4bit
daydream run mlx-community/Qwen3.5-9B-MLX-4bit
```

#### 3. 本地 MLX 模型目录

```bash
daydream run /path/to/Qwen3.5-9B-MLX-4bit
```

本地目录第一次使用时会自动注册别名。

例如，如果你的 MLX 模型目录放在桌面：

```bash
cd /Users/starry/Desktop/Daydreamcli
daydream run /Users/starry/Desktop/SmolLM2-135M-Instruct-4bit
```

如果 `/Users/starry/Desktop/Daydreamcli` 是项目目录，那么传给 `daydream run` 的应该是模型目录路径，不是项目目录路径本身。

### 聊天模式

进入交互聊天：

```bash
daydream run qwen3:8b
```

聊天内支持：

- `/`：在聊天框里打开命令菜单
- `/effort`：打开思维链长度选择器
- `/effort instant`
- `/effort short`
- `/effort default`
- `/effort long`
- `/help`
- `/reset`
- `/clear`
- `/t`
- `/quit`

#### Slash 命令与 reasoning effort

当你输入 `/` 时，Daydream 会在聊天框里用低透明度文字显示一组命令提示，风格更接近 Claude Code / Codex 这类 coding-agent 的聊天输入区。

`/effort` 可以手动调节模型这一轮的推理强度：

- `instant`
- `short`
- `default`
- `long`

默认值是 `default`。

如果当前模型大概率支持 reasoning effort 控制，Daydream 会在请求里附加一条很小的 system hint；如果模型不支持，也不会崩溃，只会提示该模型可能忽略这个设置。

#### 多行输入

三引号块输入：

```text
>>> """
... explain this file:
... def add(a, b):
...     return a + b
... """
```

反斜杠续行：

```text
>>> explain this file\
... def add(a, b):\
...     return a + b
```

#### Daydreaming 动画

模型首个 token 输出前，CLI 会显示 `Daydreaming` 动画，风格接近 coding-agent 的 `Working` 状态。

### 服务模式

`daydream serve` 默认前台运行。

前台启动本地 API 服务：

```bash
daydream serve --model hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
```

查看状态：

```bash
daydream ps
```

停止服务：

```bash
daydream stop
```

必要时强制停止：

```bash
daydream stop --force
```

如果需要后台模式：

```bash
daydream serve --background --model hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
```

### OpenAI 兼容 API

默认地址：

```text
http://127.0.0.1:11434
```

请求示例：

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/SmolLM2-135M-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

适合：

- 本地 agent
- coding tools
- OpenAI 兼容客户端

### 模型查看与管理

查看已下载模型：

```bash
daydream list
```

查看内置和已发现模型名：

```bash
daydream models
```

查看模型详情：

```bash
daydream show qwen3:8b
daydream show hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
```

删除缓存中的模型：

```bash
daydream rm qwen3:8b
```

### 本地注册表

用户配置目录：

- `~/.daydream/config.yaml`
- `~/.daydream/registry.yaml`

你也可以手动写别名：

```yaml
qwen3.5:
  9b: mlx-community/Qwen3.5-9B-MLX-4bit
```

之后可以直接：

```bash
daydream run qwen3.5:9b
```

本地模型扫描目录：

- 默认：`~/.daydream/models/`
- 可额外通过 `DAYDREAM_MODELS_DIRS` 指定

### 环境变量

- `DAYDREAM_HOME`
- `DAYDREAM_CACHE_DIR`
- `DAYDREAM_LOCAL_MODELS_DIR`
- `DAYDREAM_MODELS_DIRS`
- `HF_HOME`
- `HF_HUB_CACHE`

例如：

```bash
export DAYDREAM_HOME=/tmp/daydream-home
export DAYDREAM_CACHE_DIR=/tmp/daydream-cache
```

### 常见工作流

只在终端本地聊天：

```bash
daydream run qwen3:8b
```

启动后台 API：

```bash
daydream serve --background --model hf.co/mlx-community/SmolLM2-135M-Instruct-4bit
daydream ps
```

用管道把内容喂给模型：

```bash
cat file.py | daydream run qwen3:8b
```

使用本地模型目录：

```bash
daydream run /path/to/My-Model-MLX-4bit
```

### 错误情况

GGUF 会被立即拒绝：

```bash
daydream run hf.co/bartowski/Qwen3.5-14B-GGUF
```

预期报错：

```text
GGUF models are not supported. Use a quantized MLX model instead.
```

未量化的 MLX 仓库同样会报错。

### 开发验证

运行测试：

```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

语法检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/daydream-pyc PYTHONPATH=src ./.venv/bin/python -m compileall src tests
```

</details>

## License

No license file is included yet.
