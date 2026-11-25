# -*- coding: utf-8 -*-
"""Подключение к модели: один интерфейс для любого провайдера.

Все провайдеры говорят на одном «диалекте» — OpenAI-совместимом API
(/v1/chat/completions, /v1/models, /v1/embeddings). Клиент один, меняются
только адрес, ключ и модель. Выбор — переменной LLM_PROVIDER в .env:

    LLM_PROVIDER=lmstudio   LM Studio на вашем компьютере (по умолчанию),
                            http://localhost:1234/v1 — без ключей и без интернета
    LLM_PROVIDER=ollama     локальный Ollama, http://localhost:11434/v1
    LLM_PROVIDER=custom     любой OpenAI-совместимый сервис:
                            LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    LLM_PROVIDER=mock       учебная «доверчивая» имитация (mock_llm.py)

Реализация — на чистой стандартной библиотеке (urllib): новых зависимостей
нет. Модель определяется автоматически (первая из загруженных в LM Studio,
см. /v1/models); можно задать явно через LLM_MODEL.

Эмбеддинги для базы знаний — там же: embed_texts() зовёт /v1/embeddings.
Модель эмбеддингов выбирается переменной EMBEDDING_MODEL или автоматически
(первая модель в LM Studio, которая отвечает на /v1/embeddings).

Qwen3.5 и другие «думающие» модели оборачивают рассуждения в <think>…</think>:
перед выдачей агенту эти блоки вырезаются (strip_think), чтобы размышления
не попадали в ответы пользователю и не ломали ReAct-парсер.
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional

DEFAULTS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434/v1",
}

_THINK = re.compile(r"<think>.*?</think>", re.S)

PROVIDERS = ("lmstudio", "ollama", "custom", "mock")


class LLMConfigError(RuntimeError):
    """Провайдер настроен неверно; сообщение говорит, что исправить."""


def strip_think(text: str) -> str:
    """Убрать блоки размышлений <think>…</think> (Qwen3.5 и другие reasoning-модели)."""
    if not text:
        return text
    return _THINK.sub("", text).strip()


# ── HTTP на стандартной библиотеке ──────────────────────────────────────────
def _request(url: str, payload: Optional[dict], headers: dict, timeout: float) -> dict:
    """GET (payload=None) или POST JSON. Возвращает разобранный JSON."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise LLMConfigError(f"{url} ответил {e.code}: {body}") from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        raise LLMConfigError(
            f"Не удалось подключиться к {url}: {reason}. "
            "Сервер запущен? (LM Studio: вкладка Developer → Start Server, порт 1234)"
        ) from e
    except json.JSONDecodeError as e:
        raise LLMConfigError(f"{url} вернул не-JSON ответ: {e}") from e


class ChatMessage:
    def __init__(self, content):
        self.content = content


class ChatResponse:
    """Ответ в ollama-подобной форме: resp.message.content (совместимо с агентом)."""

    def __init__(self, content):
        self.message = ChatMessage(content)


class OpenAICompatClient:
    """Клиент OpenAI-совместимого API (LM Studio, Ollama, любой облако-сервис)."""

    def __init__(self, provider: str, base_url: str, model: str,
                 api_key: str = "", timeout: float = 300.0):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = f"{base_url} · {model}"

    # -- служебное -----------------------------------------------------------
    @property
    def _headers(self) -> dict:
        # LM Studio ключ не требует; для custom-облаков Authorization нужен
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _payload(self, messages: list) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        try:
            payload["temperature"] = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        except ValueError:
            pass
        # Qwen3.5 — «думающая» модель: по желанию отключаем размышления
        # (LLM_ENABLE_THINKING=0), чтобы ускорить ответы на слабом железе.
        if os.getenv("LLM_ENABLE_THINKING", "").strip().lower() in ("0", "false", "no"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    # -- основной интерфейс ---------------------------------------------------
    def chat(self, messages: list, stream: bool = False) -> ChatResponse:
        """POST /v1/chat/completions → ChatResponse (думание вырезано)."""
        data = _request(f"{self.base_url}/chat/completions",
                        self._payload(messages), self._headers, self.timeout)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMConfigError(f"Неожиданный формат ответа модели: "
                                 f"{json.dumps(data, ensure_ascii=False)[:300]}") from e
        # Некоторые серверы рассуждения кладут в reasoning_content
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return ChatResponse(strip_think(content))

    def list_models(self) -> List[str]:
        """GET /v1/models — загруженные модели (для автоопределения)."""
        data = _request(f"{self.base_url}/models", None, self._headers, 15)
        return [m.get("id") for m in data.get("data", []) if m.get("id")]

    def embed_texts(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """POST /v1/embeddings. Векторы нормируются (косинусная близость)."""
        import math
        url = f"{self.base_url}/embeddings"
        out: List[List[float]] = []
        for i in range(0, len(texts), 32):  # батчами, чтобы тело запроса не росло
            batch = texts[i:i + 32]
            data = _request(url, {"model": model or self.model, "input": batch},
                            self._headers, self.timeout)
            try:
                rows = sorted(data["data"], key=lambda d: d.get("index", 0))
                for row in rows:
                    vec = [float(x) for x in row["embedding"]]
                    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                    out.append([x / norm for x in vec])
            except (KeyError, TypeError) as e:
                raise LLMConfigError(f"Неожиданный формат ответа /v1/embeddings: "
                                     f"{json.dumps(data, ensure_ascii=False)[:300]}") from e
        if len(out) != len(texts):
            raise LLMConfigError(f"/v1/embeddings вернул {len(out)} векторов "
                                 f"на {len(texts)} текстов")
        return out


# ── Выбор провайдера и модели ───────────────────────────────────────────────
def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _config(provider: Optional[str]) -> dict:
    provider = (provider or _env("LLM_PROVIDER", "lmstudio")).lower()
    if provider == "mock":
        return {"provider": "mock"}
    if provider in ("lmstudio", "ollama"):
        base = _env(f"{provider.upper()}_BASE_URL", DEFAULTS[provider])
        return {"provider": provider, "base_url": base,
                "api_key": "", "model": _env("LLM_MODEL")}
    if provider == "custom":
        base, model = _env("LLM_BASE_URL"), _env("LLM_MODEL")
        if not base:
            raise LLMConfigError("Для LLM_PROVIDER=custom нужен LLM_BASE_URL "
                                 "(и LLM_MODEL; LLM_API_KEY — если сервис требует ключ).")
        return {"provider": "custom", "base_url": base,
                "api_key": _env("LLM_API_KEY"), "model": model}
    raise LLMConfigError(f"Неизвестный LLM_PROVIDER={provider!r}. "
                         f"Допустимо: {', '.join(PROVIDERS)}.")


def _resolve_model(cfg: dict) -> str:
    """Явное имя модели из LLM_MODEL или первая загруженная в /v1/models."""
    if cfg["model"]:
        return cfg["model"]
    client = OpenAICompatClient(cfg["provider"], cfg["base_url"],
                                model="__probe__", api_key=cfg["api_key"])
    models = client.list_models()
    if not models:
        raise LLMConfigError(
            f"На {cfg['base_url']} нет загруженных моделей. "
            "Откройте LM Studio, загрузите модель (например, Qwen3.5-9B) "
            "и запустите сервер: Developer → Start Server.")
    return models[0]


_cached_client: Optional[OpenAICompatClient] = None
_cached_cfg: Optional[dict] = None


def make_client(provider: Optional[str] = None):
    """Клиент по LLM_PROVIDER (или аргументу). Кэшируется; смена .env подхватывается."""
    global _cached_client, _cached_cfg
    cfg = _config(provider)
    if cfg["provider"] == "mock":
        from mock_llm import CredulousLLM
        return CredulousLLM()
    if _cached_client is None or _cached_cfg != cfg:
        client = OpenAICompatClient(
            cfg["provider"], cfg["base_url"], _resolve_model(cfg),
            api_key=cfg["api_key"],
            timeout=float(_env("LLM_TIMEOUT_S", "300") or 300),
        )
        _cached_client, _cached_cfg = client, cfg
    return _cached_client


def chat(messages: list, stream: bool = False):
    """Единая точка входа для агента: llm_provider.chat(messages)."""
    return make_client().chat(messages, stream=stream)


# ── Эмбеддинги (база знаний) ────────────────────────────────────────────────
_embed_model: Optional[str] = None


def embed_texts(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Эмбеддинги через /v1/embeddings того же сервера, что и модель.

    Модель эмбеддингов: EMBEDDING_MODEL из .env либо автоопределение —
    первая модель сервера, которая отвечает на запрос /v1/embeddings
    (чатовые модели на нём честно падают ошибкой).
    """
    global _embed_model
    cfg = _config(None)
    if cfg["provider"] == "mock":
        raise LLMConfigError(
            "Провайдер mock не умеет эмбеддинги, поэтому RAG недоступен. "
            "Для базы знаний запустите LM Studio с embedding-моделью "
            "(LLM_PROVIDER=lmstudio), например text-embedding-nomic-embed-text-v1.5.")
    client = make_client()
    explicit = model or _env("EMBEDDING_MODEL")
    if explicit:
        return client.embed_texts(texts, model=explicit)
    if _embed_model is None:
        for candidate in client.list_models():
            try:
                client.embed_texts(["проверка"], model=candidate)
                _embed_model = candidate
                break
            except LLMConfigError:
                continue
        if _embed_model is None:
            raise LLMConfigError(
                "Не нашлась модель эмбеддингов: в LM Studio не загружена "
                "embedding-модель. Загрузите, например, "
                "text-embedding-nomic-embed-text-v1.5 (вкладка Developer), "
                "или укажите имя явно в EMBEDDING_MODEL.")
    return client.embed_texts(texts, model=_embed_model)


def describe(provider: Optional[str] = None) -> str:
    """Короткая строка о текущей конфигурации (для запуска сервера; без сети)."""
    cfg = _config(provider)
    if cfg["provider"] == "mock":
        return "mock — учебная «доверчивая» имитация (офлайн)"
    model = cfg["model"] or (_cached_client.model if _cached_client else "автоопределение")
    key = ", ключ задан" if cfg["api_key"] else ", без ключа (локально)"
    return f"{cfg['provider']} · {model} · {cfg['base_url']}{key}"


def probe(provider: Optional[str] = None) -> str:
    """Живая предпроверка: один короткий запрос к модели. Возвращает её ответ."""
    client = make_client(provider)
    resp = client.chat([{"role": "user",
                         "content": "Проверка связи. Ответь ровно одним словом: работает"}])
    return (resp.message.content or "").strip()
