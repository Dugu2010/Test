"""Configuration for ForgeCLI."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_DIR = os.path.join(PROJECT_ROOT, 'tools')
LIB_DIR = os.path.join(PROJECT_ROOT, 'lib')
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'forge.log')
SESSION_DIR = os.path.join(PROJECT_ROOT, 'sessions')

# API Configuration
API_BASE_URL = 'https://integrate.api.nvidia.com'
API_ENDPOINT = f'{API_BASE_URL}/v1/chat/completions'

# Model defaults
DEFAULT_MODEL = 'moonshotai/kimi-k2-instruct'

# All free NVIDIA endpoint models — sorted by popularity
AVAILABLE_MODELS = [
    # ── Top-tier (>10M requests) ─────────────────────────────────────
    'moonshotai/kimi-k2-instruct',                  # 21.48M  coding, reasoning, agentic
    'deepseek-ai/deepseek-v3.2',                     # 16.73M  reasoning, long context
    'z-ai/glm-4.7',                                  # 15.53M  multilingual, agentic coding
    'deepseek-ai/deepseek-v3.1-terminus',             # 14.10M  tool calling, 128K context
    'moonshotai/kimi-k2-instruct-0905',              # 12.96M  long context, enhanced reasoning
    'deepseek-ai/deepseek-v3.1',                     # 12.72M  reasoning, tool use

    # ── High-tier (5M–10M) ───────────────────────────────────────────
    'stepfun-ai/step-3.5-flash',                     #  9.48M  agentic, sparse MoE
    'microsoft/phi-3.5-mini-instruct',               #  7.88M  lightweight multilingual
    'mistralai/mistral-large-3-675b-instruct-2512',  #  7.42M  agentic, VLM, 675B MoE
    'google/gemma-3-27b-it',                         #  6.37M  multimodal, reasoning
    'meta/llama-4-maverick-17b-128e-instruct',       #  6.19M  multilingual, 128 MoE
    'mistralai/devstral-2-123b-instruct-2512',       #  5.86M  coding, 256K context

    # ── Mid-tier (3M–5M) ─────────────────────────────────────────────
    'mistralai/mistral-medium-3-instruct',           #  4.75M  enterprise, multimodal
    'qwen/qwen3-coder-480b-a35b-instruct',           #  3.76M  agentic coding, 256K ctx
    'qwen/qwq-32b',                                  #  3.70M  reasoning, coding
    'bytedance/seed-oss-36b-instruct',               #  3.68M  long context, reasoning
    'mistralai/magistral-small-2506',                #  3.60M  reasoning, edge deploy
    'moonshotai/kimi-k2-thinking',                   #  3.57M  reasoning, 256K, tool use
    'nvidia/nv-embed-v1',                            #  3.11M  text embeddings (non-chat)

    # ── Standard-tier (<3M) ──────────────────────────────────────────
    'mistralai/mistral-small-3.1-24b-instruct-2503', #  2.35M  multilingual, efficient
    'tiiuae/falcon3-7b-instruct',                    #  2.01M  coding, math, reasoning
    'nvidia/nv-dinov2',                              #  1.24M  vision embeddings (non-chat)
    'mistralai/mistral-nemotron',                    #  891K   agentic, function calling
    'google/gemma-2-27b-it',                         #  877K   text gen, code gen
]

# Generation defaults
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

# Rate limiting
RATE_LIMIT_RPM = 30          # requests per minute budget
RATE_LIMIT_BACKOFF_BASE = 1.0
RATE_LIMIT_BACKOFF_MAX = 60.0
RATE_LIMIT_MAX_RETRIES = 5

# Context
MAX_CONTEXT_MESSAGES = 50    # max conversation turns kept
MAX_TOOL_OUTPUT_CHARS = 8000 # truncate long tool outputs

# TUI
APP_NAME = 'ForgeCLI'
AI_NAME = 'ForgeAI'
APP_VERSION = '1.0.0'

# Streaming
STREAM_ENABLED = True
STREAM_REFRESH_RATE = 15     # Live display refresh per second
