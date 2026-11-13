# ForgeCLI

ForgeCLI is an NVIDIA Chat Completions terminal assistant with streaming responses,
saved sessions, and model-dependent function calling.

## Setup and launch

From any directory, run the setup script:

```sh
bash /path/to/ai/setup.sh
```

The script installs dependencies relative to its own location. Set
`NVIDIA_API_KEY` in the environment or in `ai/.env`, then launch:

```sh
python /path/to/ai/bin/ai.py
```

Relative file and shell-tool paths use the shell's current working directory at
launch. Use `/cd <directory>` to change the workspace during a session. ForgeCLI
stores its settings, history, logs, and sessions beside the application files.

The optional web interface is available with `--web`; `--host` and `--port`
configure its bind address and port:

```sh
python /path/to/ai/bin/ai.py --web --host 127.0.0.1 --port 1432
```

## Tool availability

Function calling is enabled only for models listed in `TOOL_CAPABLE_MODELS`.
Run `/model` to see model support or `/tools` to see the current tool status.
If the selected model does not support tools, choose one marked `yes`.

The terminal help command (`/help`) lists chat and session commands. `/api help`
lists API configuration and health commands. Errors are written to `ai/logs/forge.log`.

## Troubleshooting

- Check `NVIDIA_API_KEY` with `/api info`; the key itself is never shown there.
- Check the current model and tool support with `/model` and `/tools`.
- Use `/api health` to check endpoint connectivity.
- Run `python -m pip install -r /path/to/ai/requirements.txt` if dependencies are missing.
