# -*- coding: utf-8 -*-
"""Слой 1. Вход: что агент принимает от пользователя (методичка, глава III).

Закрывает:
  - LLM01 Prompt Injection через невидимые символы (нулевой ширины, «символы-теги»,
    полноширинные буквы): «И​г​н​о​р​и​р​у​й» перестаёт отличаться от «Игнорируй»
    и ловится правилами уже после нормализации;
  - LLM06 Unbounded Consumption: гигантские запросы и флуд запросами.

Что делает:
  sanitize_input(text, max_len)
    - вырезает невидимые символы: нулевой ширины (U+200B…U+200F, U+2060, U+FEFF),
      «символы-теги» U+E0000…U+E007F и управляющие символы (кроме \\n и \\t);
    - приводит текст к нормальной форме Юникода (NFKC): полноширинное «ｉｇｎｏｒｅ»
      превращается в обычное «ignore»;
    - если после очистки текст пустой или длиннее max_len — выбрасывает
      InputRejected. Молча обрезать нельзя: обрезанный запрос меняет смысл.

  RateLimiter(limit, window_s)
    - скользящее окно: не больше `limit` запросов от одного пользователя
      за `window_s` секунд; запросы старше окна не считаются;
    - пользователи независимы; отказ не записывается как попытка
      (иначе атакующий сам себя заблокирует навсегда).
"""
import threading
import unicodedata


class InputRejected(Exception):
    """Запрос не принят: слишком длинный, пустой или состоит из мусора."""


# Явные диапазоны из методички + категории Юникода Cc (управляющие) и Cf (формата).
_TAG_RANGE = range(0xE0000, 0xE0080)
_ALLOWED_CONTROL = {"\n", "\t"}


def _is_invisible(ch: str) -> bool:
    cp = ord(ch)
    if ch in _ALLOWED_CONTROL:
        return False
    if cp in _TAG_RANGE:                      # «символы-теги» U+E0000…U+E007F
        return True
    return unicodedata.category(ch) in ("Cc", "Cf")


def _strip_invisible(text: str) -> str:
    return "".join(ch for ch in text if not _is_invisible(ch))


def sanitize_input(text: str, max_len: int = 2000) -> str:
    """Очистка и проверка пользовательского ввода.

    Порядок: вырезать невидимые → NFKC → ещё раз вырезать невидимые
    (нормализация может породить новые символы формата) → проверить.
    """
    cleaned = _strip_invisible(text)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = _strip_invisible(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        raise InputRejected("Пустой или полностью невидимый ввод")
    if len(cleaned) > max_len:
        raise InputRejected(f"Запрос длиннее {max_len} символов")
    return cleaned


class RateLimiter:
    """Не больше `limit` запросов на пользователя за скользящее окно `window_s`."""

    def __init__(self, limit: int = 5, window_s: float = 60):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict = {}
        self._lock = threading.Lock()

    def allow(self, user_id: str, now: float) -> bool:
        with self._lock:
            fresh = [t for t in self._hits.get(user_id, []) if now - t < self.window_s]
            if len(fresh) >= self.limit:
                self._hits[user_id] = fresh
                return False
            fresh.append(now)
            self._hits[user_id] = fresh
            return True
