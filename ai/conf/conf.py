"""Configuration for ForgeCLI."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_DIR = os.path.join(PROJECT_ROOT, 'tools')
LIB_DIR = os.path.join(PROJECT_ROOT, 'lib')
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'forge.log')
SESSION_DIR = os.path.join(PROJECT_ROOT, 'sessions')
UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'uploads')
WEB_DIR = os.path.join(PROJECT_ROOT, 'web')
SETTINGS_FILE = os.path.join(PROJECT_ROOT, '.forge_settings.json')

# API Configuration
API_BASE_URL = 'https://integrate.api.nvidia.com'
API_ENDPOINT = f'{API_BASE_URL}/v1/chat/completions'

# Model defaults
DEFAULT_MODEL = 'moonshotai/kimi-k2-instruct'

# All free NVIDIA endpoint models
AVAILABLE_MODELS = [
    'moonshotai/kimi-k2-instruct',
    'deepseek-ai/deepseek-v3.2',
    'z-ai/glm-4.7',
    'deepseek-ai/deepseek-v3.1-terminus',
    'moonshotai/kimi-k2-instruct-0905',
    'deepseek-ai/deepseek-v3.1',
    'stepfun-ai/step-3.5-flash',
    'microsoft/phi-3.5-mini-instruct',
    'mistralai/mistral-large-3-675b-instruct-2512',
    'google/gemma-3-27b-it',
    'meta/llama-4-maverick-17b-128e-instruct',
    'mistralai/devstral-2-123b-instruct-2512',
    'mistralai/mistral-medium-3-instruct',
    'qwen/qwen3-coder-480b-a35b-instruct',
    'qwen/qwq-32b',
    'bytedance/seed-oss-36b-instruct',
    'mistralai/magistral-small-2506',
    'moonshotai/kimi-k2-thinking',
    'nvidia/nv-embed-v1',
    'mistralai/mistral-small-3.1-24b-instruct-2503',
    'tiiuae/falcon3-7b-instruct',
    'nvidia/nv-dinov2',
    'mistralai/mistral-nemotron',
    'google/gemma-2-27b-it',
]

# Models that support function calling / tools
# Only these models get tools[] in the API payload.
# Others get plain chat (no tools parameter at all).
TOOL_CAPABLE_MODELS = {
    'moonshotai/kimi-k2-instruct',
    'deepseek-ai/deepseek-v3.2',
    'z-ai/glm-4.7',
    'deepseek-ai/deepseek-v3.1-terminus',
    'moonshotai/kimi-k2-instruct-0905',
    'deepseek-ai/deepseek-v3.1',
    'stepfun-ai/step-3.5-flash',
    'mistralai/mistral-large-3-675b-instruct-2512',
    'meta/llama-4-maverick-17b-128e-instruct',
    'mistralai/devstral-2-123b-instruct-2512',
    'mistralai/mistral-medium-3-instruct',
    'qwen/qwen3-coder-480b-a35b-instruct',
    'moonshotai/kimi-k2-thinking',
    'mistralai/mistral-nemotron',
}

# Non-chat models (embedding / vision-only). Skip entirely for chat.
NON_CHAT_MODELS = {
    'nvidia/nv-embed-v1',
    'nvidia/nv-dinov2',
}

# Model metadata for auto-selection and UI
MODEL_META = {
    'moonshotai/kimi-k2-instruct':                  {'tags': ['coding', 'reasoning', 'agentic'], 'desc': 'Top MoE model with strong reasoning and coding', 'tier': 'top'},
    'deepseek-ai/deepseek-v3.2':                     {'tags': ['reasoning', 'long-context'], 'desc': '685B reasoning LLM with sparse attention', 'tier': 'top'},
    'z-ai/glm-4.7':                                  {'tags': ['multilingual', 'coding', 'agentic'], 'desc': 'Multilingual agentic coding partner', 'tier': 'top'},
    'deepseek-ai/deepseek-v3.1-terminus':             {'tags': ['tool-calling', 'agentic', 'reasoning'], 'desc': 'Hybrid inference with strict function calling', 'tier': 'top'},
    'moonshotai/kimi-k2-instruct-0905':              {'tags': ['long-context', 'reasoning'], 'desc': 'Enhanced reasoning with longer context', 'tier': 'top'},
    'deepseek-ai/deepseek-v3.1':                     {'tags': ['reasoning', 'tool-calling'], 'desc': 'Fast reasoning with 128K context', 'tier': 'top'},
    'stepfun-ai/step-3.5-flash':                     {'tags': ['agentic', 'reasoning'], 'desc': '200B sparse MoE reasoning engine', 'tier': 'high'},
    'microsoft/phi-3.5-mini-instruct':               {'tags': ['lightweight', 'multilingual', 'fast'], 'desc': 'Lightweight multilingual LLM', 'tier': 'high'},
    'mistralai/mistral-large-3-675b-instruct-2512':  {'tags': ['agentic', 'vision', 'multimodal'], 'desc': '675B MoE VLM for agentic use', 'tier': 'high'},
    'google/gemma-3-27b-it':                         {'tags': ['vision', 'multimodal', 'reasoning'], 'desc': 'Multimodal model for image reasoning', 'tier': 'high'},
    'meta/llama-4-maverick-17b-128e-instruct':       {'tags': ['multilingual', 'general'], 'desc': 'Multimodal 128 MoE model', 'tier': 'high'},
    'mistralai/devstral-2-123b-instruct-2512':       {'tags': ['coding', 'long-context'], 'desc': 'Code model with 256K context', 'tier': 'high'},
    'mistralai/mistral-medium-3-instruct':           {'tags': ['enterprise', 'multimodal'], 'desc': 'Enterprise multimodal language model', 'tier': 'mid'},
    'qwen/qwen3-coder-480b-a35b-instruct':           {'tags': ['coding', 'agentic'], 'desc': 'Agentic coding with 256K context', 'tier': 'mid'},
    'qwen/qwq-32b':                                  {'tags': ['reasoning', 'coding', 'math'], 'desc': 'Powerful reasoning model', 'tier': 'mid'},
    'bytedance/seed-oss-36b-instruct':               {'tags': ['long-context', 'reasoning'], 'desc': 'Long-context reasoning LLM', 'tier': 'mid'},
    'mistralai/magistral-small-2506':                {'tags': ['reasoning', 'fast'], 'desc': 'High performance reasoning, edge deploy', 'tier': 'mid'},
    'moonshotai/kimi-k2-thinking':                   {'tags': ['reasoning', 'thinking', 'tool-calling'], 'desc': 'Open reasoning with 256K context', 'tier': 'mid'},
    'nvidia/nv-embed-v1':                            {'tags': ['embeddings'], 'desc': 'Text embedding model (non-chat)', 'tier': 'embed'},
    'mistralai/mistral-small-3.1-24b-instruct-2503': {'tags': ['multilingual', 'fast', 'vision'], 'desc': 'Efficient multilingual model', 'tier': 'standard'},
    'tiiuae/falcon3-7b-instruct':                    {'tags': ['coding', 'math', 'fast'], 'desc': 'Lightweight coding and reasoning', 'tier': 'standard'},
    'nvidia/nv-dinov2':                              {'tags': ['vision', 'embeddings'], 'desc': 'Visual embeddings (non-chat)', 'tier': 'embed'},
    'mistralai/mistral-nemotron':                    {'tags': ['agentic', 'tool-calling'], 'desc': 'Built for agentic workflows', 'tier': 'standard'},
    'google/gemma-2-27b-it':                         {'tags': ['general', 'coding'], 'desc': 'Text generation and code', 'tier': 'standard'},
}

# Generation defaults
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

# Rate limiting
RATE_LIMIT_RPM = 30
RATE_LIMIT_BACKOFF_BASE = 1.0
RATE_LIMIT_BACKOFF_MAX = 60.0
RATE_LIMIT_MAX_RETRIES = 5

# Context
MAX_CONTEXT_MESSAGES = 50
MAX_TOOL_OUTPUT_CHARS = 8000

# TUI
APP_NAME = 'ForgeCLI'
AI_NAME = 'ForgeAI'
APP_VERSION = '1.0.0'

# Streaming
STREAM_ENABLED = True
STREAM_REFRESH_RATE = 15

# Web UI
WEB_HOST = '0.0.0.0'
WEB_PORT = 1432
MAX_UPLOAD_SIZE = 50 * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {
    '.py', '.js', '.ts', '.html', '.css', '.json', '.yaml', '.yml',
    '.md', '.txt', '.csv', '.xml', '.toml', '.cfg', '.ini', '.sh',
    '.bash', '.zsh', '.fish', '.rs', '.go', '.java', '.c', '.cpp',
    '.h', '.hpp', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.sql', '.dockerfile', '.makefile', '.gitignore',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp',
    '.pdf', '.doc', '.docx', '.log', '.env',
}


def model_supports_tools(model: str) -> bool:
    """Check if a model supports function calling / tools."""
    return model in TOOL_CAPABLE_MODELS


def model_is_chat(model: str) -> bool:
    """Check if a model supports chat completions (not just embeddings)."""
    return model not in NON_CHAT_MODELS
