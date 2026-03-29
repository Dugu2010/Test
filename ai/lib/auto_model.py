"""ForgeAI Auto-Model Selection """

import re
from typing import Optional, List
from conf.conf import AVAILABLE_MODELS, MODEL_META, DEFAULT_MODEL

# Intent patterns: (compiled_regex, intent_category)
_PATTERNS = [
    # Vision / image
    (re.compile(r'\b(image|photo|picture|screenshot|diagram|png|jpg|jpeg|gif|svg|look at|what.*see|describe.*image|analyze.*image|vision|upload.*image)\b', re.I), 'vision'),
    # Coding
    (re.compile(r'\b(code|function|class|def |import |bug|debug|refactor|implement|program|script|regex|api|endpoint|algorithm|syntax|compile|runtime|error.*code|traceback|exception|git|commit|merge|pull request|dockerfile|docker|kubernetes|nginx|deploy|cicd|ci/cd|test.*unit|pytest|jest|webpack|npm|pip|cargo|make|cmake)\b', re.I), 'coding'),
    # Math / reasoning
    (re.compile(r'\b(math|equation|calcul|integral|derivative|probability|statistic|proof|theorem|algebra|geometry|matrix|linear|optimize|formula|compute|solve.*equation)\b', re.I), 'math'),
    # Deep reasoning / thinking
    (re.compile(r'\b(think.*step|reason.*through|explain.*why|analyze.*depth|compare.*contrast|pros.*cons|trade.*off|evaluate|critique|deep.*dive|think.*carefully|chain.*thought|step.*by.*step.*reason)\b', re.I), 'thinking'),
    # Tool use / agentic
    (re.compile(r'\b(run.*command|execute|terminal|shell|create.*file|write.*file|delete.*file|search.*file|install|pip install|apt|brew|chmod|mkdir|find.*files?|ls |cd |cat |grep )\b', re.I), 'agentic'),
    # Long context
    (re.compile(r'\b(entire.*file|whole.*codebase|all.*files|large.*document|full.*context|summarize.*everything|read.*all|long.*text|book|paper|article)\b', re.I), 'long-context'),
    # Fast / simple
    (re.compile(r'\b(quick|fast|brief|short|simple|just|only|tl;?dr|one.?liner|yes or no|true or false)\b', re.I), 'fast'),
    # Multilingual
    (re.compile(r'\b(translat|chinese|japanese|korean|french|german|spanish|arabic|hindi|russian|portuguese|multilingual|locali[sz])\b', re.I), 'multilingual'),
]

# Intent 
_INTENT_TO_TAGS = {
    'vision':       ['vision', 'multimodal'],
    'coding':       ['coding'],
    'math':         ['reasoning', 'math'],
    'thinking':     ['thinking', 'reasoning'],
    'agentic':      ['agentic', 'tool-calling'],
    'long-context': ['long-context'],
    'fast':         ['fast', 'lightweight'],
    'multilingual': ['multilingual'],
}

# Preferred models per intent (hand-picked best choice)
_INTENT_PREFERRED = {
    'vision':       'google/gemma-3-27b-it',
    'coding':       'qwen/qwen3-coder-480b-a35b-instruct',
    'math':         'qwen/qwq-32b',
    'thinking':     'moonshotai/kimi-k2-thinking',
    'agentic':      'deepseek-ai/deepseek-v3.1-terminus',
    'long-context': 'moonshotai/kimi-k2-instruct-0905',
    'fast':         'microsoft/phi-3.5-mini-instruct',
    'multilingual': 'z-ai/glm-4.7',
}


def detect_intent(message: str) -> Optional[str]:
    """Detect the primary intent of a user message."""
    scores = {}
    for pattern, intent in _PATTERNS:
        matches = pattern.findall(message)
        if matches:
            scores[intent] = scores.get(intent, 0) + len(matches)
    if not scores:
        return None
    return max(scores, key=scores.get)


def detect_file_intent(filenames: List[str]) -> Optional[str]:
    """Detect intent from uploaded file types."""
    image_exts = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp'}
    code_exts = {'.py', '.js', '.ts', '.rs', '.go', '.java', '.c', '.cpp', '.rb', '.php', '.swift', '.kt'}
    for name in filenames:
        ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        ext = f'.{ext}'
        if ext in image_exts:
            return 'vision'
        if ext in code_exts:
            return 'coding'
    return None


def select_model(
    message: str,
    filenames: Optional[List[str]] = None,
    current_model: Optional[str] = None,
) -> str:
    """Select the best model for the given message and context.

    Returns the model name string.
    """
    # Check file-based intent first
    if filenames:
        file_intent = detect_file_intent(filenames)
        if file_intent and file_intent in _INTENT_PREFERRED:
            preferred = _INTENT_PREFERRED[file_intent]
            if preferred in AVAILABLE_MODELS:
                return preferred

    # Check message intent
    intent = detect_intent(message)
    if intent and intent in _INTENT_PREFERRED:
        preferred = _INTENT_PREFERRED[intent]
        if preferred in AVAILABLE_MODELS:
            return preferred

    # Fallback to current or default
    return current_model or DEFAULT_MODEL


def get_intent_info(message: str, filenames: Optional[List[str]] = None) -> dict:
    """Return intent detection details for UI display."""
    intent = None
    source = 'default'

    if filenames:
        file_intent = detect_file_intent(filenames)
        if file_intent:
            intent = file_intent
            source = 'file'

    if not intent:
        msg_intent = detect_intent(message)
        if msg_intent:
            intent = msg_intent
            source = 'message'

    model = select_model(message, filenames)
    meta = MODEL_META.get(model, {})

    return {
        'intent': intent or 'general',
        'source': source,
        'model': model,
        'model_desc': meta.get('desc', ''),
        'model_tags': meta.get('tags', []),
    }
