import asyncio
import hashlib
import html as html_lib
import json
import logging
import os
import re
import shutil
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wiki")

# --- Config ---

WIKI_DIR = os.environ.get("WIKI_DIR", "/app/content")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
SESSION_FILE = os.path.join(DATA_DIR, "session.json")
WIKI_PASSWORD = os.environ.get("WIKI_PASSWORD", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")
# External ASR (speech-to-text) endpoint. Configure via .env; empty disables voice input.
ASR_UPSTREAM = os.environ.get("ASR_UPSTREAM", "")
ASR_MODEL = os.environ.get("ASR_MODEL", "gigaam-rnnt")

# Telegram bot: chat with the same Claude session from Telegram.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_IDS = {
    int(x) for x in os.environ.get("TELEGRAM_ALLOWED_IDS", "").replace(" ", "").split(",") if x.lstrip("-").isdigit()
}
TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Full access over the wiki folder, including Bash. Listing the tools in
# --allowedTools (rather than --permission-mode bypassPermissions) lets them run
# without interactive prompts in headless mode and avoids the "running as root"
# guard that bypass mode trips inside the container.
ALLOWED_TOOLS = "Read,Glob,Grep,Write,Edit,MultiEdit,Bash,WebSearch,WebFetch,TodoWrite,NotebookEdit"

SYSTEM_PROMPT = (
    "Ты — ассистент персональной вики-базы знаний. Все материалы — это markdown-файлы (.md), "
    "организованные по папкам в рабочей директории. Твои задачи:\n"
    "- Отвечать на вопросы пользователя, опираясь на содержимое вики. Перед ответом ищи по файлам "
    "(Grep, Glob) и читай нужные страницы (Read).\n"
    "- По просьбе создавать новые страницы и редактировать существующие: Edit для точечных правок, "
    "Write для новых файлов. Пиши чистый, аккуратный markdown.\n"
    "- Поддерживай структуру: осмысленные имена файлов и папок, заголовки, связи между страницами "
    "через относительные ссылки вида [текст](путь.md).\n"
    "- Помогай с организацией (переименовать, разбить, объединить страницы), делай это аккуратно.\n"
    "Отвечай кратко и по делу, на языке пользователя. Не выдумывай факты, которых нет в вики; "
    "если чего-то нет — так и скажи."
)

# --- Auth (single password) ---

# A stable bearer token derived from the password: knowing the password yields
# exactly this token. Survives restarts as long as the password is unchanged.
AUTH_TOKEN = hashlib.sha256(("wiki:" + WIKI_PASSWORD).encode()).hexdigest()

security = HTTPBearer(auto_error=False)


def check_token(token: str | None) -> bool:
    return bool(token) and bool(WIKI_PASSWORD) and token == AUTH_TOKEN


async def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(security)):
    if not creds or not check_token(creds.credentials):
        raise HTTPException(401, "Unauthorized")
    return True


# --- Single session state ---


def load_session() -> str | None:
    try:
        with open(SESSION_FILE) as f:
            return json.load(f).get("session_id")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_session(session_id: str | None):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump({"session_id": session_id}, f)


def clear_session():
    try:
        os.remove(SESSION_FILE)
    except FileNotFoundError:
        pass


# Serializes access to the single Claude session so web and Telegram turns
# never resume the same session concurrently.
claude_lock = asyncio.Lock()


# --- App ---

@asynccontextmanager
async def lifespan(_app: FastAPI):
    bot_task = None
    if TELEGRAM_BOT_TOKEN:
        bot_task = asyncio.create_task(telegram_poller())
        logger.info("Telegram bot enabled")
    else:
        logger.info("Telegram bot disabled (no TELEGRAM_BOT_TOKEN)")
    try:
        yield
    finally:
        if bot_task:
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Personal Wiki Assistant", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class LoginReq(BaseModel):
    password: str


@app.post("/auth/login")
async def login(req: LoginReq):
    if not WIKI_PASSWORD or req.password != WIKI_PASSWORD:
        raise HTTPException(401, "Неверный пароль")
    return {"token": AUTH_TOKEN}


@app.get("/auth/me")
async def me(_: bool = Depends(require_auth)):
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Claude chat (single session over WebSocket) ---


def build_cmd(message: str, session_id: str | None) -> list[str]:
    cmd = [
        "claude", "-p", message,
        "--output-format", "stream-json", "--verbose",
        "--include-partial-messages",
        "--allowedTools", ALLOWED_TOOLS,
        "--model", CLAUDE_MODEL,
        "--append-system-prompt", SYSTEM_PROMPT,
    ]
    if session_id:
        cmd += ["--resume", session_id]
    return cmd


def with_context(message: str, context: dict) -> str:
    path = (context or {}).get("path")
    selection = ((context or {}).get("selection") or "")[:4000]
    if not path and not selection:
        return message
    lines = ["[Контекст: где сейчас находится пользователь]"]
    if path:
        lines.append(f"Открытая страница: {path}")
    if selection:
        lines.append("Выделенный фрагмент страницы:\n<<<\n" + selection + "\n>>>")
    lines.append("")
    lines.append(message)
    return "\n".join(lines)


async def run_claude(ws: WebSocket, message: str):
    """Spawn claude CLI, stream events to the WebSocket, persist the session id."""
    async with claude_lock:
        await _run_claude_ws(ws, message)


async def _run_claude_ws(ws: WebSocket, message: str):
    session_id = load_session()
    proc = await asyncio.create_subprocess_exec(
        *build_cmd(message, session_id),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=10 * 1024 * 1024,
        cwd=WIKI_DIR,
    )

    streaming_text = ""
    current_msg_id = ""
    last_push = 0.0
    THROTTLE = 0.05
    final_session_id = session_id

    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode().strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "stream_event":
                inner = event.get("event", {})
                itype = inner.get("type", "")
                if itype == "message_start":
                    streaming_text = ""
                    current_msg_id = inner.get("message", {}).get("id", current_msg_id)
                elif itype == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        streaming_text += delta["text"]
                        now = time.monotonic()
                        if now - last_push >= THROTTLE:
                            await ws.send_json({"t": "text", "id": current_msg_id, "text": streaming_text})
                            last_push = now
                elif itype == "content_block_stop":
                    if streaming_text:
                        await ws.send_json({"t": "text", "id": current_msg_id, "text": streaming_text})
                        last_push = time.monotonic()
                continue

            if etype == "assistant" and event.get("message"):
                msg = event["message"]
                current_msg_id = msg.get("id", "")
                for block in msg.get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        await ws.send_json({"t": "text", "id": current_msg_id, "text": block["text"]})
                    elif block.get("type") == "tool_use":
                        inp = block.get("input", {})
                        await ws.send_json({
                            "t": "tool",
                            "name": block.get("name", ""),
                            "pattern": inp.get("pattern") or inp.get("command", "")[:80],
                            "file": inp.get("file_path", ""),
                        })
                streaming_text = ""

            elif etype == "result":
                final_session_id = event.get("session_id", session_id)

        stderr_data = await proc.stderr.read()
        await proc.wait()

        if proc.returncode != 0:
            err = stderr_data.decode()[:500]
            logger.error("claude exit %d: %s", proc.returncode, err)
            low = err.lower()
            if "context" in low or "too long" in low or "max tokens" in low:
                user_err = "Контекст сессии переполнен. Очистите его командой /clear или /compact."
            else:
                user_err = f"Ошибка Claude (код {proc.returncode})"
            await ws.send_json({"t": "error", "text": user_err})
        else:
            save_session(final_session_id)

        await ws.send_json({"t": "done", "sid": final_session_id})

    except WebSocketDisconnect:
        proc.kill()
        raise


async def run_claude_collect(message: str, on_tool=None) -> str:
    """Run one Claude turn and return the full reply text. Used by the Telegram bot."""
    async with claude_lock:
        session_id = load_session()
        proc = await asyncio.create_subprocess_exec(
            *build_cmd(message, session_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=10 * 1024 * 1024,
            cwd=WIKI_DIR,
        )

        texts: list[str] = []
        result_text = ""
        final_session_id = session_id

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            raw = line.decode().strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            if etype == "assistant" and event.get("message"):
                for block in event["message"].get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        texts.append(block["text"])
                    elif block.get("type") == "tool_use" and on_tool:
                        inp = block.get("input", {})
                        detail = inp.get("file_path") or inp.get("pattern") or (inp.get("command", "")[:80])
                        try:
                            await on_tool(block.get("name", ""), detail)
                        except Exception:
                            pass
            elif etype == "result":
                final_session_id = event.get("session_id", session_id)
                if event.get("subtype") == "success" and event.get("result"):
                    result_text = event["result"]

        stderr_data = await proc.stderr.read()
        await proc.wait()

        if proc.returncode != 0:
            err = stderr_data.decode()[:500]
            logger.error("claude(tg) exit %d: %s", proc.returncode, err)
            low = err.lower()
            if "context" in low or "too long" in low or "max tokens" in low:
                return "⚠️ Контекст сессии переполнен. Очистите его командой /clear."
            return f"⚠️ Ошибка Claude (код {proc.returncode})."

        save_session(final_session_id)
        return (result_text or "\n\n".join(texts)).strip() or "(пустой ответ)"


@app.websocket("/chat/ws")
async def chat_ws(ws: WebSocket, token: str = ""):
    if not check_token(token):
        await ws.close(code=4001, reason="Unauthorized")
        return
    await ws.accept()
    logger.info("WS connected")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "message":
                continue
            message = data.get("text", "").strip()
            if not message:
                continue

            # Session control commands, handled by the backend.
            if message == "/clear":
                clear_session()
                await ws.send_json({"t": "text", "id": "sys", "text": "Контекст очищен — начинаю новую сессию."})
                await ws.send_json({"t": "done", "sid": None})
                continue

            try:
                await run_claude(ws, with_context(message, data.get("context") or {}))
            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error("run_claude error: %s", e)
                await ws.send_json({"t": "error", "text": str(e)})
                await ws.send_json({"t": "done", "sid": load_session()})

    except WebSocketDisconnect:
        logger.info("WS disconnected")
    except Exception as e:
        logger.error("WS error: %s", e)


# --- File API ---


def safe_path(rel: str) -> str:
    """Resolve a wiki-relative path and ensure it stays inside WIKI_DIR."""
    rel = (rel or "").lstrip("/")
    abs_path = os.path.realpath(os.path.join(WIKI_DIR, rel))
    root = os.path.realpath(WIKI_DIR)
    if abs_path != root and not abs_path.startswith(root + os.sep):
        raise HTTPException(400, "Недопустимый путь")
    return abs_path


def build_tree(abs_dir: str, rel_prefix: str) -> list[dict]:
    nodes = []
    try:
        entries = sorted(os.scandir(abs_dir), key=lambda e: (e.is_file(), e.name.lower()))
    except FileNotFoundError:
        return nodes
    for entry in entries:
        if entry.name.startswith("."):
            continue
        rel = f"{rel_prefix}{entry.name}"
        if entry.is_dir():
            nodes.append({
                "name": entry.name, "path": rel, "type": "dir",
                "children": build_tree(entry.path, rel + "/"),
            })
        elif entry.name.endswith(".md"):
            nodes.append({"name": entry.name, "path": rel, "type": "file"})
    return nodes


class WriteReq(BaseModel):
    path: str
    text: str


class CreateReq(BaseModel):
    path: str
    type: str  # "file" | "dir"


class RenameReq(BaseModel):
    src: str
    dst: str


@app.get("/files/tree")
async def files_tree(_: bool = Depends(require_auth)):
    os.makedirs(WIKI_DIR, exist_ok=True)
    return {"tree": build_tree(WIKI_DIR, "")}


@app.get("/files/content")
async def files_content(path: str, _: bool = Depends(require_auth)):
    abs_path = safe_path(path)
    if not os.path.isfile(abs_path):
        raise HTTPException(404, "Файл не найден")
    with open(abs_path, encoding="utf-8") as f:
        return {"path": path, "text": f.read()}


@app.put("/files/content")
async def files_save(req: WriteReq, _: bool = Depends(require_auth)):
    abs_path = safe_path(req.path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(req.text)
    return {"ok": True}


@app.post("/files/create")
async def files_create(req: CreateReq, _: bool = Depends(require_auth)):
    abs_path = safe_path(req.path)
    if os.path.exists(abs_path):
        raise HTTPException(409, "Уже существует")
    if req.type == "dir":
        os.makedirs(abs_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write("")
    return {"ok": True}


@app.post("/files/rename")
async def files_rename(req: RenameReq, _: bool = Depends(require_auth)):
    src = safe_path(req.src)
    dst = safe_path(req.dst)
    if not os.path.exists(src):
        raise HTTPException(404, "Не найдено")
    if os.path.exists(dst):
        raise HTTPException(409, "Цель уже существует")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.rename(src, dst)
    return {"ok": True}


@app.delete("/files")
async def files_delete(path: str, _: bool = Depends(require_auth)):
    abs_path = safe_path(path)
    if os.path.realpath(abs_path) == os.path.realpath(WIKI_DIR):
        raise HTTPException(400, "Нельзя удалить корень")
    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path)
    elif os.path.isfile(abs_path):
        os.remove(abs_path)
    else:
        raise HTTPException(404, "Не найдено")
    return {"ok": True}


# --- Voice transcription (proxy to ASR service) ---


@app.post("/api/asr/transcribe")
async def asr_transcribe(audio: UploadFile, model_id: str = Form(ASR_MODEL), _: bool = Depends(require_auth)):
    if not ASR_UPSTREAM:
        raise HTTPException(503, "Распознавание речи не настроено")
    content = await audio.read()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            ASR_UPSTREAM,
            files={"audio": (audio.filename or "recording.webm", content, audio.content_type or "audio/webm")},
            data={"model_id": model_id},
        )
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "Ошибка распознавания речи")
    return resp.json()


# --- Telegram bot (long polling; shares the single Claude session) ---

TG_WELCOME = (
    "👋 Привет! Я ассистент твоей вики.\n\n"
    "Пиши текстом или надиктовывай голосовые — отвечу на вопросы по заметкам, создам и отредактирую страницы. "
    "Контекст общий с веб-версией: что обсудили здесь, помню и там.\n\n"
    "Команды:\n"
    "/clear — очистить контекст (новая сессия)\n"
    "/compact — сжать историю, сохранив суть\n"
    "/help — это сообщение"
)

TG_LIMIT = 3500  # split markdown below Telegram's 4096-char message cap (HTML adds length)


def md_to_tg_html(md: str) -> str:
    """Convert a markdown chunk to the safe HTML subset Telegram accepts."""
    md = re.sub(r"^#{1,6}[ \t]+(.+?)\s*#*$", r"**\1**", md, flags=re.M)   # headings → bold
    md = re.sub(r"^[ \t]*[-*][ \t]+", "• ", md, flags=re.M)               # bullets → •

    stash: list[str] = []

    def keep(s: str) -> str:
        stash.append(s)
        return f"\x00{len(stash) - 1}\x00"

    md = re.sub(r"```[^\n]*\n(.*?)```", lambda m: keep(f"<pre>{html_lib.escape(m.group(1))}</pre>"), md, flags=re.S)
    md = re.sub(r"```(.*?)```", lambda m: keep(f"<pre>{html_lib.escape(m.group(1))}</pre>"), md, flags=re.S)
    md = re.sub(r"`([^`\n]+)`", lambda m: keep(f"<code>{html_lib.escape(m.group(1))}</code>"), md)

    md = html_lib.escape(md)
    md = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', md)
    md = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", md)
    md = re.sub(r"__([^_\n]+)__", r"<b>\1</b>", md)

    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], md)


def split_md(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Split markdown into chunks under `limit`, preferring line boundaries."""
    if len(text) <= limit:
        return [text]
    chunks, buf = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if buf:
                chunks.append(buf); buf = ""
            chunks.append(line[:limit]); line = line[limit:]
        if len(buf) + len(line) + 1 > limit:
            chunks.append(buf); buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        chunks.append(buf)
    return chunks


async def tg_api(client: httpx.AsyncClient, method: str, **params) -> dict:
    try:
        r = await client.post(f"{TG_API}/{method}", json=params)
        data = r.json()
        if not data.get("ok"):
            logger.warning("tg %s failed: %s", method, data.get("description"))
        return data
    except Exception as e:
        logger.warning("tg %s error: %s", method, e)
        return {"ok": False}


async def tg_send(client: httpx.AsyncClient, chat_id: int, text: str):
    """Send (possibly long) markdown, rendered as HTML, falling back to plain text."""
    for chunk in split_md(text):
        res = await tg_api(
            client, "sendMessage",
            chat_id=chat_id, text=md_to_tg_html(chunk),
            parse_mode="HTML", disable_web_page_preview=True,
        )
        if not res.get("ok"):
            await tg_api(client, "sendMessage", chat_id=chat_id, text=chunk, disable_web_page_preview=True)


async def tg_typing(client: httpx.AsyncClient, chat_id: int, stop: asyncio.Event):
    """Keep the 'typing…' indicator alive while Claude works."""
    while not stop.is_set():
        await tg_api(client, "sendChatAction", chat_id=chat_id, action="typing")
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.5)
        except asyncio.TimeoutError:
            pass


async def tg_transcribe(client: httpx.AsyncClient, file_id: str, mime: str | None) -> str | None:
    """Download a Telegram voice/audio file and transcribe it via the ASR service."""
    if not ASR_UPSTREAM:
        return None
    info = await tg_api(client, "getFile", file_id=file_id)
    if not info.get("ok"):
        return None
    file_path = info["result"].get("file_path")
    if not file_path:
        return None
    try:
        dl = await client.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}")
        if dl.status_code != 200:
            return None
        audio = dl.content
    except Exception as e:
        logger.warning("tg file download error: %s", e)
        return None

    # Telegram names voice files .oga, which the ASR rejects by extension. The bytes
    # are OGG/Opus, so resend under names the ASR accepts (.ogg, then .webm as in web).
    candidates = [("voice.ogg", "audio/ogg"), ("voice.webm", "audio/webm")]
    try:
        async with httpx.AsyncClient(timeout=180) as asr:
            for fname, ctype in candidates:
                resp = await asr.post(
                    ASR_UPSTREAM,
                    files={"audio": (fname, audio, ctype)},
                    data={"model_id": ASR_MODEL},
                )
                if resp.status_code == 200:
                    text = ((resp.json() or {}).get("text") or "").strip()
                    return text or None
                logger.warning("asr(tg) %s status %s: %s", fname, resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("asr(tg) error: %s", e)
        return None
    return None


async def tg_handle(client: httpx.AsyncClient, update: dict):
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat_id = msg["chat"]["id"]
    user_id = (msg.get("from") or {}).get("id")

    # Onboarding: with no allowlist set, reply with the sender's id so the owner can configure it.
    if not TELEGRAM_ALLOWED_IDS:
        await tg_api(client, "sendMessage", chat_id=chat_id,
                     text=f"Бот ещё не настроен. Ваш Telegram ID: {user_id}\n"
                          f"Добавьте его в TELEGRAM_ALLOWED_IDS и перезапустите backend.")
        return
    if user_id not in TELEGRAM_ALLOWED_IDS:
        await tg_api(client, "sendMessage", chat_id=chat_id, text="🔒 Это приватный бот.")
        return

    text = (msg.get("text") or "").strip()

    # Voice / audio message → transcribe, then treat the text as the prompt.
    media = msg.get("voice") or msg.get("audio") or msg.get("video_note")
    if not text and media:
        await tg_api(client, "sendChatAction", chat_id=chat_id, action="typing")
        text = await tg_transcribe(client, media["file_id"], media.get("mime_type"))
        if not text:
            await tg_api(client, "sendMessage", chat_id=chat_id, text="🎤 Не удалось распознать голосовое — попробуй ещё раз.")
            return

    if not text:
        return

    if text in ("/start", "/help"):
        await tg_api(client, "sendMessage", chat_id=chat_id, text=TG_WELCOME)
        return
    if text == "/clear":
        clear_session()
        await tg_api(client, "sendMessage", chat_id=chat_id, text="🧹 Контекст очищен — начинаю новую сессию.")
        return

    stop = asyncio.Event()
    typing = asyncio.create_task(tg_typing(client, chat_id, stop))
    try:
        async def on_tool(_name, _detail):
            await tg_api(client, "sendChatAction", chat_id=chat_id, action="typing")
        reply = await run_claude_collect(text, on_tool)
    except Exception as e:
        logger.error("tg run_claude error: %s", e)
        reply = "⚠️ Что-то пошло не так при обработке запроса."
    finally:
        stop.set()
        await typing

    await tg_send(client, chat_id, reply)


async def telegram_poller():
    """Long-poll Telegram for updates and dispatch them. One worker, sequential."""
    offset = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(70.0)) as client:
        # Drop any backlog queued while the bot was offline.
        try:
            init = await tg_api(client, "getUpdates", offset=-1, timeout=0)
            if init.get("ok") and init.get("result"):
                offset = init["result"][-1]["update_id"] + 1
        except Exception:
            pass
        while True:
            try:
                resp = await tg_api(client, "getUpdates", offset=offset, timeout=50)
                for upd in resp.get("result", []):
                    offset = upd["update_id"] + 1
                    await tg_handle(client, upd)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("tg poll error: %s", e)
                await asyncio.sleep(3)
