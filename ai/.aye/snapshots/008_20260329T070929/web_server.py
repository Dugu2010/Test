"""ForgeAI Web Server"""

import os
import json
import uuid
import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from lib.agent import Agent
from lib.auto_model import select_model, get_intent_info
from lib.utils import logger, safe_json_loads
from conf.conf import (
    APP_NAME, AI_NAME, APP_VERSION,
    AVAILABLE_MODELS, MODEL_META, DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
    SESSION_DIR, UPLOAD_DIR, WEB_DIR, SETTINGS_FILE,
    MAX_UPLOAD_SIZE, ALLOWED_UPLOAD_EXTENSIONS,
    API_ENDPOINT,
)

app = FastAPI(title=f"{APP_NAME} Web UI", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

_agents = {}
_lock = threading.Lock()


def _get_agent(session_id: str) -> Agent:
    with _lock:
        if session_id not in _agents:
            agent = Agent()
            agent.session_id = session_id
            agent.load_session(session_id)
            _agents[session_id] = agent
        return _agents[session_id]


def _load_settings() -> dict:
    defaults = {
        'model': DEFAULT_MODEL,
        'auto_model': True,
        'temperature': DEFAULT_TEMPERATURE,
        'max_tokens': DEFAULT_MAX_TOKENS,
        'top_p': DEFAULT_TOP_P,
        'system_prompt': '',
        'theme': 'dark',
        'api_key': '',
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def _save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")


# ── REST Endpoints ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(WEB_DIR, 'index.html')
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>ForgeAI</h1><p>web/index.html not found</p>", status_code=404)
    with open(html_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(f.read())


@app.get("/api/info")
async def api_info():
    return {
        'name': APP_NAME,
        'ai_name': AI_NAME,
        'version': APP_VERSION,
        'models_count': len(AVAILABLE_MODELS),
        'endpoint': API_ENDPOINT,
    }


@app.get("/api/health")
async def api_health():
    """Check API health and connectivity."""
    import requests as req
    api_ok = False
    try:
        base = API_ENDPOINT.rsplit('/', 1)[0]
        r = req.head(base, timeout=8)
        api_ok = r.status_code < 500
    except Exception:
        pass

    settings = _load_settings()
    return {
        'status': 'ok',
        'api_reachable': api_ok,
        'api_key_set': bool(settings.get('api_key') or os.environ.get('NVIDIA_API_KEY')),
        'version': APP_VERSION,
    }


@app.get("/api/commands")
async def api_commands():
    """List all CLI commands (mirrors help)."""
    return {
        'commands': [
            {'cmd': '/help', 'aliases': ['/h'], 'desc': 'Show help'},
            {'cmd': '/clear', 'aliases': ['/cls'], 'desc': 'Clear screen'},
            {'cmd': '/model', 'aliases': ['/m'], 'desc': 'Show or change model'},
            {'cmd': '/tools', 'aliases': [], 'desc': 'List available tools'},
            {'cmd': '/history', 'aliases': [], 'desc': 'Show conversation history'},
            {'cmd': '/new', 'aliases': [], 'desc': 'Start new session'},
            {'cmd': '/save', 'aliases': [], 'desc': 'Save current session'},
            {'cmd': '/sessions', 'aliases': [], 'desc': 'List saved sessions'},
            {'cmd': '/load', 'aliases': [], 'desc': 'Load a session'},
            {'cmd': '/session', 'aliases': [], 'desc': 'Current session info'},
            {'cmd': '/stream', 'aliases': ['/s'], 'desc': 'Toggle streaming'},
            {'cmd': '/verbose', 'aliases': ['/v'], 'desc': 'Toggle verbose'},
            {'cmd': '/stats', 'aliases': [], 'desc': 'API statistics'},
            {'cmd': '/cd', 'aliases': [], 'desc': 'Change directory'},
            {'cmd': '/api info', 'aliases': [], 'desc': 'API connection info'},
            {'cmd': '/api models', 'aliases': [], 'desc': 'List models with details'},
            {'cmd': '/api settings', 'aliases': [], 'desc': 'View/update settings'},
            {'cmd': '/api health', 'aliases': [], 'desc': 'Check API health'},
            {'cmd': '/api key', 'aliases': [], 'desc': 'View/set API key'},
            {'cmd': '/exit', 'aliases': ['/quit', '/q'], 'desc': 'Exit'},
        ]
    }


@app.get("/api/models")
async def list_models():
    models = []
    for m in AVAILABLE_MODELS:
        meta = MODEL_META.get(m, {})
        models.append({
            'id': m,
            'tags': meta.get('tags', []),
            'description': meta.get('desc', ''),
            'tier': meta.get('tier', 'standard'),
        })
    return {'models': models}


@app.get("/api/settings")
async def get_settings():
    return _load_settings()


@app.post("/api/settings")
async def update_settings(body: dict):
    current = _load_settings()
    for key in ('model', 'auto_model', 'temperature', 'max_tokens', 'top_p', 'system_prompt', 'theme', 'api_key'):
        if key in body:
            current[key] = body[key]
    _save_settings(current)
    return current


@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    try:
        for fname in sorted(os.listdir(SESSION_DIR), reverse=True):
            if fname.endswith('.json'):
                path = os.path.join(SESSION_DIR, fname)
                try:
                    with open(path) as f:
                        data = json.load(f)
                    user_msgs = [m for m in data.get('messages', []) if m.get('role') == 'user']
                    title = user_msgs[0].get('content', 'New Chat')[:80] if user_msgs else 'New Chat'
                    sessions.append({
                        'id': data.get('session_id', fname[:-5]),
                        'title': title,
                        'created': data.get('created', ''),
                        'model': data.get('model', DEFAULT_MODEL),
                        'message_count': len(user_msgs),
                    })
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
    return {'sessions': sessions}


@app.post("/api/sessions")
async def create_session():
    sid = str(uuid.uuid4())[:8]
    agent = Agent()
    agent.session_id = sid
    with _lock:
        _agents[sid] = agent
    agent.save_session()
    return {'session_id': sid}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Session not found")
    with open(path) as f:
        data = json.load(f)
    messages = [
        m for m in data.get('messages', [])
        if m.get('role') in ('user', 'assistant', 'tool')
    ]
    return {
        'session_id': data.get('session_id', session_id),
        'model': data.get('model', DEFAULT_MODEL),
        'created': data.get('created', ''),
        'messages': messages,
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)
    with _lock:
        _agents.pop(session_id, None)
    return {'deleted': True}


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Session not found")
    return FileResponse(path, filename=f"forge_{session_id}.json", media_type="application/json")


@app.post("/api/sessions/import")
async def import_session(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data = json.loads(content)
        sid = data.get('session_id', str(uuid.uuid4())[:8])
        path = os.path.join(SESSION_DIR, f"{sid}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return {'session_id': sid}
    except Exception as e:
        raise HTTPException(400, f"Invalid session file: {e}")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.size and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "File too large")

    ext = Path(file.filename).suffix.lower() if file.filename else ''
    if ext and ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(400, f"File type {ext} not allowed")

    file_id = f"{uuid.uuid4().hex[:12]}_{file.filename or 'upload'}"
    file_path = os.path.join(UPLOAD_DIR, file_id)
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    text_preview = None
    is_image = ext in ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp')
    if not is_image:
        try:
            text_preview = content.decode('utf-8', errors='replace')[:2000]
        except Exception:
            pass

    return {
        'file_id': file_id,
        'filename': file.filename,
        'size': len(content),
        'ext': ext,
        'is_image': is_image,
        'text_preview': text_preview,
        'path': file_path,
    }


@app.get("/api/uploads/{file_id}")
async def get_upload(file_id: str):
    file_path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    return FileResponse(file_path)


@app.post("/api/auto-model")
async def auto_model_detect(body: dict):
    message = body.get('message', '')
    filenames = body.get('filenames', [])
    return get_intent_info(message, filenames)


# ── WebSocket Chat ───────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            raw = await ws.receive_text()
            data = safe_json_loads(raw, {})
            if not data:
                await ws.send_json({'type': 'error', 'text': 'Invalid JSON'})
                continue

            msg_type = data.get('type', '')

            if msg_type == 'ping':
                await ws.send_json({'type': 'pong'})
                continue

            if msg_type != 'message':
                continue

            message = data.get('content', '').strip()
            session_id = data.get('session_id', str(uuid.uuid4())[:8])
            model_choice = data.get('model', 'auto')
            file_context = data.get('files', [])

            if not message and not file_context:
                await ws.send_json({'type': 'error', 'text': 'Empty message'})
                continue

            agent = _get_agent(session_id)

            settings = _load_settings()
            agent.client.temperature = settings.get('temperature', DEFAULT_TEMPERATURE)
            agent.client.max_tokens = settings.get('max_tokens', DEFAULT_MAX_TOKENS)
            agent.client.top_p = settings.get('top_p', DEFAULT_TOP_P)

            api_key = settings.get('api_key', '') or os.environ.get('NVIDIA_API_KEY', '')
            if api_key:
                agent.client.session.headers['Authorization'] = f'Bearer {api_key}'
                agent.client.api_key = api_key

            if model_choice == 'auto' and settings.get('auto_model', True):
                filenames = [f.get('filename', '') for f in file_context]
                selected_model = select_model(message, filenames, agent.client.model)
                intent_info = get_intent_info(message, filenames)
                agent.client.set_model(selected_model)
                await ws.send_json({
                    'type': 'auto_model',
                    'model': selected_model,
                    'intent': intent_info['intent'],
                })
            elif model_choice != 'auto':
                agent.client.set_model(model_choice)

            full_message = message
            if file_context:
                file_parts = []
                for fc in file_context:
                    fname = fc.get('filename', 'file')
                    preview = fc.get('text_preview', '')
                    if preview:
                        file_parts.append(f"\n--- Uploaded file: {fname} ---\n{preview}\n--- End of {fname} ---")
                    else:
                        file_parts.append(f"\n[Uploaded file: {fname} ({fc.get('ext', 'unknown')} format)]")
                full_message = message + '\n'.join(file_parts)

            loop = asyncio.get_event_loop()
            event_queue = asyncio.Queue()

            def _run_agent():
                try:
                    for event in agent.process_streaming(full_message):
                        asyncio.run_coroutine_threadsafe(
                            event_queue.put(event), loop
                        )
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(
                        event_queue.put({'type': 'error', 'text': str(e)}), loop
                    )
                finally:
                    asyncio.run_coroutine_threadsafe(
                        event_queue.put(None), loop
                    )

            thread = threading.Thread(target=_run_agent, daemon=True)
            thread.start()

            while True:
                event = await event_queue.get()
                if event is None:
                    break
                await ws.send_json(event)

            agent.save_session()

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await ws.send_json({'type': 'error', 'text': str(e)})
        except Exception:
            pass


def start_server(host: str = None, port: int = None):
    import uvicorn
    from conf.conf import WEB_HOST, WEB_PORT
    h = host or WEB_HOST
    p = port or WEB_PORT
    logger.info(f"Starting {APP_NAME} Web UI on http://{h}:{p}")
    print(f"\n\033[1;36m\U0001f525 {APP_NAME} Web UI\033[0m")
    print(f"   \033[90mPowered by {AI_NAME} v{APP_VERSION}\033[0m")
    print(f"   \033[92m\u279c\033[0m http://localhost:{p}")
    print(f"   \033[90mPress Ctrl+C to stop\033[0m\n")
    uvicorn.run(app, host=h, port=p, log_level="warning")
