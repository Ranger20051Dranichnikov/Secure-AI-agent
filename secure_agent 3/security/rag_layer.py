# -*- coding: utf-8 -*-
"""Слой 3. База знаний: права читателя и приём документов (глава IV).

Закрывает:
  - LLM02 Sensitive Information Disclosure: гость не должен получать
    зарплатные ведомости — выдача фильтруется по правам читателя ДО отбора
    лучших k (иначе после фильтра топ-k читатель может остаться с пустой
    выдачей, а стенд ACL из лекции сломается);
  - LLM05 Data Poisoning: документы принимаются в индекс только из доверенных
    источников, а скрытое содержимое (HTML-комментарии, текст с
    display:none, белый текст, символы нулевой ширины) вычищается при приёме.

Что делает:
  retrieve(query, user, chunks, k, score) — k самых релевантных чанков,
    но только из доступных читателю групп (поле "group" у чанка,
    user["groups"] у читателя). Фильтр до сортировки и top-k.

  is_trusted_source(source) — строковый источник доверен, только если он
    начинается с "kb/" и не содержит «..».

  is_trusted_file(path, base_dir) — адаптация для файловой базы агента:
    файл лежит внутри base_dir, не является символической ссылкой
    (через неё можно вылезти наружу) и имеет расширение .pdf.

  clean_document(text) — очистка при приёме в индекс: убрать HTML-комментарии
    <!-- ... -->, элементы со стилем display:none или белым цветом текста
    (color:#fff / color:white), символы нулевой ширины. Видимый текст
    сохраняется.
"""
import re
from pathlib import Path

_HIDDEN_STYLE = re.compile(
    r"<([a-z0-9]+)\b[^>]*?style\s*=\s*[\"'][^\"']*?"
    r"(?:display\s*:\s*none|color\s*:\s*(?:#fff\b|#ffffff\b|white\b))"
    r"[^\"']*?[\"'][^>]*?>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_INVISIBLE = re.compile("[\u200b-\u200f\u2060\ufeff\U000e0000-\U000e007f]")


def retrieve(query: str, user: dict, chunks: list, k: int, score) -> list:
    """Сначала фильтр по правам, потом сортировка и top-k."""
    groups = set((user or {}).get("groups") or ["all"])
    allowed = [c for c in chunks if c.get("group", "all") in groups]
    ranked = sorted(allowed, key=lambda c: -score(query, c["text"]))
    return ranked[:k]


def is_trusted_source(source: str) -> bool:
    """Доверяем только источникам внутри kb/ без обхода каталога."""
    return bool(source) and source.startswith("kb/") and ".." not in source


def is_trusted_file(path: Path, base_dir: Path) -> bool:
    """Файловая адаптация: внутри каталога, не симлинк, обычный .pdf."""
    try:
        base = Path(base_dir).resolve()
        full = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    if Path(path).is_symlink():
        return False
    if base != full and base not in full.parents:
        return False
    return full.suffix.lower() == ".pdf" and full.is_file()


def clean_document(text: str) -> str:
    """Убрать скрытое содержимое, сохранив видимый текст."""
    text = _COMMENT.sub("", text)
    text = _HIDDEN_STYLE.sub("", text)
    text = _INVISIBLE.sub("", text)
    return text
