# -*- coding: utf-8 -*-
"""Слой 6. Секреты: короткая жизнь и никаких ключей в журнале (глава VI).

Закрывает:
  - долгоживущие «токены доступа», которые подходят ко всем задачам:
    TokenVault выдаёт короткоживущие токены, привязанные к области действия
    (scope), с возможностью отзыва;
  - LLM02 Sensitive Information Disclosure через журнал: redact() прячет всё,
    что похоже на секрет, прежде чем текст попадёт в лог или консоль.

  TokenVault:
    issue(scope, now, ttl_s=None) -> {"token", "scope", "expires_at"}
      срок по умолчанию 300 секунд; запросить можно не больше 3600,
      больший срок срезается; токены генерируются secrets.token_urlsafe
      (random предсказуем и не подходит);
    check(token, scope, now) -> bool — истина, только если токен выдан этим
      хранилищем, не истёк, не отозван и выдан именно для этой области;
    revoke(token) — после отзыва check всегда False.

  redact(text) — заменяет на «[СКРЫТО]»:
      sk-... и tvly-... (после префикса не меньше 8 символов из букв, цифр,
      «-» и «_»), «Bearer <токен>», пары вида api_key=..., token: ...,
      password=... (регистр не важен).
"""
import re
import secrets
import threading
from typing import Optional

_BEARER = re.compile(r"\b[Bb]earer\s+\S+")
_KEY_VALUE = re.compile(
    r"\b(api[_-]?key|token|password|passwd|secret)\s*[=:]\s*\S+", re.IGNORECASE
)
_SK_TOKEN = re.compile(r"\b(?:sk|tvly)-[A-Za-z0-9_-]{8,}")


def redact(text: str) -> str:
    """Спрятать всё, что похоже на секрет. Порядок важен: сначала пары и
    Bearer (забирают значение целиком), затем одиночные sk-/tvly-ключи."""
    text = _BEARER.sub("Bearer [СКРЫТО]", text)
    text = _KEY_VALUE.sub(lambda m: m.group(1) + "=[СКРЫТО]", text)
    text = _SK_TOKEN.sub("[СКРЫТО]", text)
    return text


class TokenVault:
    """Выдача и проверка короткоживущих токенов с областями действия."""

    DEFAULT_TTL_S = 300
    MAX_TTL_S = 3600

    def __init__(self):
        self._tokens: dict = {}
        self._revoked: set = set()
        self._lock = threading.Lock()

    def issue(self, scope: str, now: float, ttl_s: Optional[float] = None) -> dict:
        ttl = self.DEFAULT_TTL_S if ttl_s is None else min(float(ttl_s), self.MAX_TTL_S)
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._tokens[token] = {"scope": scope, "expires_at": now + ttl}
            self._revoked.discard(token)
        return {"token": token, "scope": scope, "expires_at": now + ttl}

    def check(self, token: str, scope: str, now: float) -> bool:
        with self._lock:
            if token in self._revoked:
                return False
            record = self._tokens.get(token)
            if record is None:
                return False
            return record["scope"] == scope and now < record["expires_at"]

    def revoke(self, token: str) -> None:
        with self._lock:
            self._revoked.add(token)
            self._tokens.pop(token, None)
