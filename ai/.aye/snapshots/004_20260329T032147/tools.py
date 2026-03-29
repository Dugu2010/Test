"""Tool definitions and execution engine for AI CLI Agent."""

import os
import subprocess
import glob
import json
import traceback
from lib.utils import logger, truncate
from conf.conf import MAX_TOOL_OUTPUT_CHARS

# ──────────────────────────────────────────────
# Tool Schema Definitions (OpenAI function-call format)
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "cmd_exec",
            "description": "Execute a shell command and return stdout/stderr. Use for running programs, scripts, git, package managers, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command (optional, defaults to current dir)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (optional, default 60)"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file. Creates the file and any parent directories if they don't exist. Overwrites existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_append",
            "description": "Append content to the end of a file. Creates the file if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_delete",
            "description": "Delete a file or empty directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file or directory to delete"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List contents of a directory with file sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: current directory)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively (default: false)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a pattern or containing text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern for file names (e.g., '*.py', 'src/**/*.js')"
                    },
                    "text": {
                        "type": "string",
                        "description": "Search for files containing this text (optional)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Root directory to search from (default: current dir)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute a Python code snippet and return the output. Useful for calculations, data processing, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]


# ──────────────────────────────────────────────
# Tool Execution Functions
# ──────────────────────────────────────────────

def exec_cmd(command: str, working_dir: str = None, timeout: int = 60) -> dict:
    """Execute a shell command."""
    logger.info(f"cmd_exec: {command} (cwd={working_dir}, timeout={timeout})")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir or None,
            env={**os.environ, 'TERM': 'dumb'},
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- STDERR ---\n" + result.stderr) if output else result.stderr
        if not output:
            output = "(no output)"
        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "output": truncate(output)
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "exit_code": -1, "output": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "exit_code": -1, "output": f"Error: {str(e)}"}


def exec_file_write(path: str, content: str) -> dict:
    """Write content to a file."""
    logger.info(f"file_write: {path} ({len(content)} chars)")
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "output": f"Written {len(content)} chars to {abs_path}"}
    except Exception as e:
        return {"success": False, "output": f"Error writing file: {str(e)}"}


def exec_file_read(path: str) -> dict:
    """Read file content."""
    logger.info(f"file_read: {path}")
    try:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return {"success": False, "output": f"File not found: {abs_path}"}
        if os.path.getsize(abs_path) > 500_000:
            return {"success": False, "output": f"File too large ({os.path.getsize(abs_path)} bytes). Use cmd_exec with head/tail."}
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return {"success": True, "output": truncate(content)}
    except Exception as e:
        return {"success": False, "output": f"Error reading file: {str(e)}"}


def exec_file_append(path: str, content: str) -> dict:
    """Append to a file."""
    logger.info(f"file_append: {path} ({len(content)} chars)")
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'a', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "output": f"Appended {len(content)} chars to {abs_path}"}
    except Exception as e:
        return {"success": False, "output": f"Error appending: {str(e)}"}


def exec_file_delete(path: str) -> dict:
    """Delete a file or empty directory."""
    logger.info(f"file_delete: {path}")
    try:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            os.remove(abs_path)
            return {"success": True, "output": f"Deleted file: {abs_path}"}
        elif os.path.isdir(abs_path):
            os.rmdir(abs_path)
            return {"success": True, "output": f"Deleted directory: {abs_path}"}
        else:
            return {"success": False, "output": f"Path not found: {abs_path}"}
    except Exception as e:
        return {"success": False, "output": f"Error deleting: {str(e)}"}


def exec_list_dir(path: str = '.', recursive: bool = False) -> dict:
    """List directory contents."""
    logger.info(f"list_dir: {path} (recursive={recursive})")
    try:
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return {"success": False, "output": f"Not a directory: {abs_path}"}

        entries = []
        if recursive:
            for root, dirs, files in os.walk(abs_path):
                # Skip hidden and common ignored dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.git')]
                for name in files:
                    fp = os.path.join(root, name)
                    rel = os.path.relpath(fp, abs_path)
                    try:
                        size = os.path.getsize(fp)
                    except OSError:
                        size = 0
                    entries.append(f"  {_fmt_size(size):>8s}  {rel}")
                if len(entries) > 500:
                    entries.append("  ... (truncated, too many files)")
                    break
        else:
            for name in sorted(os.listdir(abs_path)):
                fp = os.path.join(abs_path, name)
                try:
                    size = os.path.getsize(fp)
                except OSError:
                    size = 0
                kind = 'd' if os.path.isdir(fp) else 'f'
                entries.append(f"  [{kind}] {_fmt_size(size):>8s}  {name}")

        header = f"Directory: {abs_path} ({len(entries)} items)"
        return {"success": True, "output": header + "\n" + "\n".join(entries)}
    except Exception as e:
        return {"success": False, "output": f"Error listing directory: {str(e)}"}


def exec_search_files(pattern: str = None, text: str = None, path: str = '.') -> dict:
    """Search for files by pattern or content."""
    logger.info(f"search_files: pattern={pattern}, text={text}, path={path}")
    try:
        abs_path = os.path.abspath(path)
        results = []

        if pattern:
            for match in glob.glob(os.path.join(abs_path, pattern), recursive=True):
                results.append(os.path.relpath(match, abs_path))
                if len(results) > 200:
                    results.append("... (truncated)")
                    break

        if text:
            for root, dirs, files in os.walk(abs_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv')]
                for name in files:
                    fp = os.path.join(root, name)
                    try:
                        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(100_000)
                        if text in content:
                            rel = os.path.relpath(fp, abs_path)
                            # Find line numbers
                            lines = content.split('\n')
                            for i, line in enumerate(lines, 1):
                                if text in line:
                                    results.append(f"{rel}:{i}: {line.strip()[:120]}")
                    except (OSError, UnicodeDecodeError):
                        pass
                    if len(results) > 200:
                        results.append("... (truncated)")
                        break

        if not results:
            return {"success": True, "output": "No matches found."}
        return {"success": True, "output": "\n".join(results)}
    except Exception as e:
        return {"success": False, "output": f"Error searching: {str(e)}"}


def exec_python(code: str) -> dict:
    """Execute Python code."""
    logger.info(f"python_exec: {code[:100]}...")
    try:
        import io
        import contextlib
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        local_vars = {}
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, {'__builtins__': __builtins__}, local_vars)
        output = stdout_buf.getvalue()
        errors = stderr_buf.getvalue()
        combined = output
        if errors:
            combined += ("\n--- STDERR ---\n" + errors) if combined else errors
        if not combined:
            combined = "(no output)"
        return {"success": True, "output": truncate(combined)}
    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "output": f"Error: {str(e)}\n{tb}"}


# ──────────────────────────────────────────────
# Tool Dispatcher
# ──────────────────────────────────────────────

TOOL_MAP = {
    'cmd_exec': lambda args: exec_cmd(
        args.get('command', ''),
        args.get('working_dir'),
        args.get('timeout', 60)
    ),
    'file_write': lambda args: exec_file_write(
        args.get('path', ''),
        args.get('content', '')
    ),
    'file_read': lambda args: exec_file_read(
        args.get('path', '')
    ),
    'file_append': lambda args: exec_file_append(
        args.get('path', ''),
        args.get('content', '')
    ),
    'file_delete': lambda args: exec_file_delete(
        args.get('path', '')
    ),
    'list_dir': lambda args: exec_list_dir(
        args.get('path', '.'),
        args.get('recursive', False)
    ),
    'search_files': lambda args: exec_search_files(
        args.get('pattern'),
        args.get('text'),
        args.get('path', '.')
    ),
    'python_exec': lambda args: exec_python(
        args.get('code', '')
    ),
}


def execute_tool(name: str, arguments: dict) -> dict:
    """Dispatch tool call by name."""
    logger.info(f"Executing tool: {name} with args: {json.dumps(arguments)[:200]}")
    handler = TOOL_MAP.get(name)
    if not handler:
        return {"success": False, "output": f"Unknown tool: {name}"}
    try:
        return handler(arguments)
    except Exception as e:
        logger.error(f"Tool execution error for {name}: {e}", exc_info=True)
        return {"success": False, "output": f"Tool execution failed: {str(e)}"}


def _fmt_size(size: int) -> str:
    """Format byte size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == 'B' else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
