# -*- coding: utf-8 -*-
"""Слой 7. Наблюдаемость: журнал без содержимого, тревога, выключатель (глава VII).

  audit_event(user_id, event, text) -> dict — запись журнала БЕЗ текста
    сообщения: только user_id, event, chars (длина) и sha256 (первые 12
    символов хеша). По хешу можно найти повторы одного запроса, не храня
    сам запрос; журнал не превращается в склад чужих писем и паролей.

  AlertDetector(window, factor) — тревога по числу запросов в минуту.
    observe(count) вызывается раз в минуту и возвращает True, если число
    запросов резко выше обычного:
      - обычное = среднее по последним `window` минутам БЕЗ тревоги
        (минуты с тревогой в базу сравнения не попадают, иначе всплеск
        «приучит» детектор к себе);
      - пока накоплено меньше 5 минут истории, тревоги нет;
      - тревога, если count > factor * обычное.
    Значения по умолчанию (window=10, factor=2.5) подобраны так, что на
    шумном графике (фон ~20±6 запросов/мин) ложных тревог нет, а всплеск
    ×4 замечается на первой же минуте.

  kill_switch_active(path) -> bool — выключатель снаружи модели: агент
    отключён, пока на диске существует файл `path`. Агент не может
    «передумать» и обойти проверку — она в коде, а не в промпте.
"""
import hashlib
import os
from collections import deque


def audit_event(user_id: str, event: str, text: str) -> dict:
    """Запись журнала без текста сообщения: длина + короткий хеш."""
    return {
        "user_id": user_id,
        "event": event,
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }


class AlertDetector:
    MIN_HISTORY = 5  # минут истории до первой тревоги

    def __init__(self, window: int = 10, factor: float = 2.5):
        self.window = window
        self.factor = factor
        self.history = deque(maxlen=window)

    def observe(self, count: float) -> bool:
        if len(self.history) < self.MIN_HISTORY:
            self.history.append(count)
            return False
        baseline = sum(self.history) / len(self.history)
        if count > self.factor * baseline:
            return True  # минута с тревогой в историю не попадает
        self.history.append(count)
        return False


def kill_switch_active(path: str) -> bool:
    """Агент отключён, пока существует файл-флаг."""
    try:
        return os.path.exists(path)
    except OSError:
        return False
