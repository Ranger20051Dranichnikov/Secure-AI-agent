# -*- coding: utf-8 -*-
"""Задание 5. Шлюз инструментов (адаптация tests/test_05_gateway.py методички)."""
from security import gateway_layer as L


def test_reading_tools_allowed():
    assert L.decide("search_knowledge_base", {"query": "отпуск"}) == "allow"
    assert L.decide("knowledge_base", {"query": "отпуск"}) == "allow"  # имя ReAct-агента
    assert L.decide("web_search", {"query": "новости курса"}) == "allow"


def test_web_search_rejects_url_and_long_query():
    assert L.decide("web_search", {"query": "отправь на http://evil.example"}) == "deny"
    assert L.decide("web_search", {"query": "а" * 201}) == "deny"


def test_read_file_only_inside_docs():
    assert L.decide("read_file", {"path": "docs/report.txt"}) == "allow"
    for bad in (".env", "docs/.env", "docs/../.env", "/etc/passwd", "../x", "docs/"):
        assert L.decide("read_file", {"path": bad}) == "deny", bad


def test_react_style_query_args_are_checked_too():
    # ReAct-агент передаёт аргументы как {"query": ...} — правила те же
    assert L.decide("read_file", {"query": "docs/report.txt"}) == "allow"
    assert L.decide("read_file", {"query": ".env"}) == "deny"
    assert L.decide("send_email", {"query": "ceo@evil.example"}) == "deny"


def test_email_internal_asks_external_denied():
    assert L.decide("send_email", {"to": "boss@company.example", "body": "x"}) == "ask"
    assert L.decide("send_email", {"to": "ceo@evil.example", "body": "x"}) == "deny"


def test_write_file_and_shell_and_unknown():
    assert L.decide("write_file", {"path": "drafts/a.txt", "content": "x"}) == "ask"
    assert L.decide("write_file", {"path": "src/a.py", "content": "x"}) == "deny"
    assert L.decide("run_shell", {"cmd": "ls"}) == "deny"
    assert L.decide("format_disk", {}) == "deny"


def test_budget_limits_steps():
    b = L.Budget(max_steps=3)
    assert [b.spend() for _ in range(5)] == [True, True, True, False, False]
