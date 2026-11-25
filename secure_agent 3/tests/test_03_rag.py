# -*- coding: utf-8 -*-
"""Задание 3. База знаний (адаптация tests/test_03_rag.py методички)."""
from security import rag_layer as L

CHUNKS = [
    {"id": "c1", "group": "hr", "text": "Зарплатная ведомость отдела разработки: зарплаты и премии разработчиков."},
    {"id": "c2", "group": "hr", "text": "Премии по итогам квартала для разработчиков."},
    {"id": "c3", "group": "dev", "text": "Грейды разработчиков: вилки зарплат."},
    {"id": "c4", "group": "all", "text": "Общая политика мотивации: премии раз в квартал."},
    {"id": "c5", "group": "all", "text": "Памятка по налоговым вычетам."},
    {"id": "c6", "group": "dev", "text": "Архитектурный документ платёжного сервиса: очередь платежей."},
    {"id": "c7", "group": "all", "text": "Правила отпусков: 28 дней в году."},
    {"id": "c8", "group": "all", "text": "Перечень офисных адресов."},
    {"id": "c9", "group": "all", "text": "Как оформить отпуск: заявление за две недели."},
]
USERS = {
    "anna": {"id": "anna", "groups": ["hr", "all"]},
    "boris": {"id": "boris", "groups": ["dev", "all"]},
    "guest": {"id": "guest", "groups": ["all"]},
}
Q = "Какие зарплаты и премии у разработчиков?"


def _score(query: str, text: str) -> int:
    """Совпадение первых пяти букв слов — как kb.score в лабораторной."""
    import re
    words = lambda s: {w[:5] for w in re.findall(r"[a-zа-яё0-9]+", s.lower()) if len(w) >= 4}
    return len(words(query) & words(text))


def _ids(user):
    return [c["id"] for c in L.retrieve(Q, USERS[user], CHUNKS, 3, _score)]


def test_guest_does_not_get_closed_chunks():
    assert all(i not in ("c1", "c2", "c3", "c6") for i in _ids("guest"))


def test_hr_gets_hr_chunks_and_not_dev_only():
    got = _ids("anna")
    assert "c1" in got and "c2" in got and "c3" not in got


def test_dev_gets_dev_chunks_and_not_hr():
    got = _ids("boris")
    assert "c3" in got and "c1" not in got and "c2" not in got


def test_filter_is_before_top_k():
    # у гостя после фильтра подходящих чанков достаточно: выдача должна быть полной
    assert len(_ids("guest")) == 3


def test_trusted_sources():
    assert L.is_trusted_source("kb/hr-payroll.pdf")
    assert not L.is_trusted_source("upload/user-42")
    assert not L.is_trusted_source("kb/../upload/x")


def test_trusted_file_rejects_symlinks_and_escapes(tmp_path):
    base = tmp_path / "kb"
    base.mkdir()
    good = base / "a.pdf"
    good.write_text("текст", encoding="utf-8")
    assert L.is_trusted_file(good, base)
    # симлинк наружу — недоверенный источник
    outside = tmp_path / "secret.pdf"
    outside.write_text("секрет", encoding="utf-8")
    link = base / "link.pdf"
    link.symlink_to(outside)
    assert not L.is_trusted_file(link, base)
    # не-PDF не принимаем
    other = base / "b.txt"
    other.write_text("x", encoding="utf-8")
    assert not L.is_trusted_file(other, base)


def test_clean_document_removes_hidden_content_keeps_visible():
    t = ('Видимый текст. <!-- скрытая команда --> Ещё текст.'
         '<span style="color:#fff">белый на белом</span>'
         '<div style="display:none">спрятано</div>невидимый​символ')
    out = L.clean_document(t)
    assert "скрытая" not in out and "белый на белом" not in out and "спрятано" not in out
    assert "​" not in out
    assert "Видимый текст." in out and "Ещё текст." in out
