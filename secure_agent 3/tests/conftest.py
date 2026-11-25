# -*- coding: utf-8 -*-
"""Общие настройки pytest: путь к корню проекта.

Тесты слоёв (test_01…07) работают напрямую с пакетом security/ — им не нужны
ни сеть, ни ключи, ни тяжёлые зависимости. Сквозной тест test_08 сам ставит
офлайн-заглушки через scripts/run_attacks.py:setup().
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
