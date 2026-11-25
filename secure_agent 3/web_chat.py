# -*- coding: utf-8 -*-
"""
Веб-версия ассистента с RAG + Web Search (локальная модель через LM Studio).

Запуск:
  python web_chat.py
  → http://127.0.0.1:5000

Рядом должен лежать chat.html из вашей рабочей версии.

ЗАЩИТА (по методичке secure_ai_agents; тяжёлая работа — в пакете security/):
  * слой 1 «Вход»  — лимит длины запроса (AGENT_MAX_INPUT_LEN), санитизация
                     невидимых символов, RateLimiter (AGENT_RATE_LIMIT за окно);
  * слой 6 «Секреты»— POST /api/session выдаёт короткоживущий серверный токен
                     (SESSION_TTL_S); прежний client-token работает как запасной
                     вариант, но новый эндпоинт — правильный путь;
  * слой 7 «Наблюдаемость» — журнал audit.jsonl без текста сообщений
                     (только длина и хеш), тревога по всплеску трафика,
                     kill-switch: создайте файл из AGENT_KILL_FLAG — и агент
                     замолчит;
  * плюс MAX_CONTENT_LENGTH против гигантских тел запросов и потолок числа
                     сессий в памяти.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import dotenv
from flask import Flask, Response, request, send_file, stream_with_context

from security.input_layer import InputRejected, RateLimiter, sanitize_input
from security.obs_layer import AlertDetector, audit_event
from security.secrets_layer import TokenVault, redact

BASE_DIR = Path(__file__).resolve().parent
dotenv.load_dotenv(BASE_DIR / ".env")

from agent_core import (  # noqa: E402
    agent_disabled,
    allowed_groups,
    load_knowledge_base,
    run_agent,
)

app = Flask(__name__)
# Слой 1/LLM06: страховка от гигантских тел запросов (64 КБ хватает с запасом)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

MAX_HISTORY = 20
MAX_INPUT_LEN = int(os.getenv("AGENT_MAX_INPUT_LEN", "2000"))
RATE_LIMIT = int(os.getenv("AGENT_RATE_LIMIT", "5"))
RATE_WINDOW_S = float(os.getenv("AGENT_RATE_WINDOW_S", "60"))
SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", "1800"))
SESSIONS_MAX = int(os.getenv("SESSIONS_MAX", "500"))
AUDIT_LOG_PATH = Path(os.getenv("AGENT_AUDIT_LOG", str(BASE_DIR / "audit_log" / "audit.jsonl")))

# ── Состояние сервера ───────────────────────────────────────────────────────
sessions: dict = {}                       # session_key -> история сообщений
_lock = threading.Lock()
limiter = RateLimiter(limit=RATE_LIMIT, window_s=RATE_WINDOW_S)   # слой 1
vault = TokenVault()                                                # слой 6


# ── Слой 7: журнал без содержимого сообщений ───────────────────────────────
def _file_audit_sink(line: str) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


audit_sink = _file_audit_sink  # тесты подменяют на свою функцию


def _log(user_id: str, event: str, text: str) -> None:
    """Запись в журнал: без текста сообщения, с маскировкой (слои 6 и 7)."""
    try:
        record = audit_event(user_id, event, text)
        audit_sink(redact(json.dumps(record, ensure_ascii=False)))
    except Exception as e:
        print(f"⚠️ audit: {e}")


def _token_fingerprint(token: str) -> str:
    """Короткий хеш токена: сам токен в журнал не пишем (слой 6)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


# ── Слой 7: тревога по всплеску числа запросов в минуту ────────────────────
class MinuteAlerts:
    def __init__(self):
        self._detector = AlertDetector()
        self._minute = None
        self._count = 0
        self._lock = threading.Lock()

    def hit(self) -> bool:
        """Засчитать запрос; True — поднята тревога."""
        with self._lock:
            minute = int(time.time() // 60)
            alert = False
            if self._minute is not None and minute != self._minute:
                alert = self._detector.observe(self._count)
                self._count = 0
            self._minute = minute
            self._count += 1
            return alert


alerts = MinuteAlerts()


# ── Вспомогательные проверки запроса ───────────────────────────────────────
def _sanitize_token(raw) -> str:
    """Токен — недоверенный ввод: оставляем только безопасный набор символов."""
    token = "".join(ch for ch in str(raw) if ch.isalnum() or ch in "_-")[:64]
    return token or "default"


def _groups_for_role(role: str) -> list:
    """Роль читателя базы знаний (слой 3). Приходит от клиента, поэтому
    пропускаем только известные группы; по умолчанию — all."""
    role = str(role or "").strip().lower()
    return [role] if role in allowed_groups() else ["all"]


def _session_key_for(token: str, now: float) -> str:
    """Серверный токен (если валиден) или запасной клиентский (слой 6)."""
    if vault.check(token, "chat", now):
        return "v:" + token
    return "c:" + token


# ── Маршруты ────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return send_file("chat.html")


@app.post("/api/session")
def api_session():
    """Слой 6: сервер выдаёт короткоживущий токен сессии."""
    if agent_disabled():
        return {"error": "Агент временно отключён"}, 503
    issued = vault.issue("chat", time.time(), SESSION_TTL_S)
    _log(_token_fingerprint(issued["token"]), "session_issued", issued["token"])
    return {"ok": True, "token": issued["token"], "expires_at": issued["expires_at"]}


@app.post("/api/reset")
def api_reset():
    if agent_disabled():
        return {"error": "Агент временно отключён"}, 503
    data = request.get_json(silent=True) or {}
    token = _sanitize_token(data.get("token"))
    with _lock:
        sessions.pop("v:" + token, None)
        sessions.pop("c:" + token, None)
    _log(_token_fingerprint(token), "session_reset", token)
    return {"ok": True}


@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    raw_message = str(data.get("message") or "")
    token = _sanitize_token(data.get("token") or "default")
    role = data.get("role") or ""

    if not raw_message.strip():
        return {"error": "Пустое сообщение"}, 400

    # Слой 7: выключатель
    if agent_disabled():
        return {"error": "Агент временно отключён"}, 503

    # Слой 1: лимит частоты
    now = time.time()
    if not limiter.allow(token, now):
        _log(_token_fingerprint(token), "rate_limited", "")
        return {"error": "Слишком много запросов, попробуйте позже"}, 429

    # Слой 1: санитизация и лимит длины
    try:
        text = sanitize_input(raw_message, MAX_INPUT_LEN)
    except InputRejected:
        _log(_token_fingerprint(token), "rejected", raw_message)
        return {"error": "Запрос отклонён: сообщение пустое, невидимое или слишком длинное"}, 413

    session_key = _session_key_for(token, now)
    with _lock:
        history = sessions.setdefault(session_key, [])

    user = {
        "id": _token_fingerprint(token),
        "groups": _groups_for_role(role),   # слой 3: права читателя базы знаний
    }
    _log(user["id"], "request", text)

    # Слой 7: счётчик трафика и тревога
    if alerts.hit():
        _log(user["id"], "alert", "аномальный всплеск запросов")
        print("⚠️ ALERT: число запросов резко выше обычного")

    def generate():
        steps = []

        def on_event(event_type, payload):
            if event_type == "tool":
                steps.append(payload)
                # Слой 7: факт вызова инструмента — в журнал (без текста запроса)
                _log(user["id"],
                     f"tool:{payload.get('name')}:{payload.get('decision', '?')}",
                     payload.get("query", ""))

        try:
            answer = run_agent(text, history, on_event=on_event, user=user)

            # сначала показываем, что искали (как «typing»-активность)
            for step in steps:
                shown_query = redact(step["query"])  # слой 6: ключи не светятся
                if step.get("decision") != "allow":
                    status = f"⛔ Действие «{step['name']}» запрещено политикой безопасности\n\n"
                elif step["name"] in ("knowledge_base", "search_knowledge_base"):
                    status = f"📚 Ищу в базе знаний: {shown_query}\n\n"
                else:
                    status = f"🌐 Ищу в интернете: {shown_query}\n\n"
                yield f"data: {json.dumps({'delta': status}, ensure_ascii=False)}\n\n"

            # затем — сам ответ (уже прошёл проверку выхода — слой 4)
            yield f"data: {json.dumps({'delta': answer}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

            with _lock:
                history.append({"role": "user", "content": text})
                history.append({"role": "assistant", "content": answer})
                if len(history) > MAX_HISTORY * 2:
                    sessions[session_key] = history[-MAX_HISTORY * 2:]
                # LLM06: потолок числа сессий в памяти — старые вытесняем
                while len(sessions) > SESSIONS_MAX:
                    sessions.pop(next(iter(sessions)))

            _log(user["id"], "reply", answer)

        except Exception as e:
            print(f"Error: {redact(str(e))}")
            yield f"data: {json.dumps({'error': redact(str(e))}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    try:
        load_knowledge_base(verbose=True)
    except Exception as e:
        print(f"⚠️ RAG не загрузился: {e}")

    # Какая модель будет отвечать (конфигурация, без обращения к серверу)
    try:
        import llm_provider
        print(f"\n🧠 Модель: {llm_provider.describe()}")
    except Exception as e:
        print(f"\n⚠️ Модель не настроена: {e}")

    print("\n🛠️  Чат запущен: http://127.0.0.1:5000\n")
    print(f"🛡️  Защита: вход ≤{MAX_INPUT_LEN} симв., {RATE_LIMIT} запр./"
          f"{RATE_WINDOW_S:.0f} c, kill-switch-файл: {os.getenv('AGENT_KILL_FLAG', 'STOP_AGENT')}\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
