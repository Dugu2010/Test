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
DEFAULT_MODEL = 'meta/llama-3.1-405b-instruct'
AVAILABLE_MODELS = [
    'meta/llama-3.1-405b-instruct',
    'meta/llama-3.1-70b-instruct',
    'meta/llama-3.1-8b-instruct',
    'nvidia/nemotron-4-340b-instruct',
    'mistralai/mixtral-8x22b-instruct-v0.1',
    'mistralai/mistral-large-2-instruct',
    'google/gemma-2-27b-it',
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
APP_VERSION = '1.0.0'

# Streaming
STREAM_ENABLED = True
STREAM_REFRESH_RATE = 15     # Live display refresh per second
