# -*- coding: utf-8 -*-
"""Задание 6. Секреты (адаптация tests/test_06_secrets.py методички)."""
from security import secrets_layer as L

FAKE_KEY = "sk-lab-FAKE-7f3a9c21d4e8b650"


def test_default_ttl_is_short_and_capped():
    v = L.TokenVault()
    assert v.issue("kb", now=1000)["expires_at"] - 1000 <= 3600
    assert v.issue("kb", now=1000, ttl_s=10**9)["expires_at"] - 1000 <= 3600


def test_token_valid_until_expiry_only():
    v = L.TokenVault()
    t = v.issue("kb", now=0, ttl_s=60)["token"]
    assert v.check(t, "kb", now=59)
    assert not v.check(t, "kb", now=61)


def test_token_is_bound_to_scope():
    v = L.TokenVault()
    t = v.issue("kb", now=0)["token"]
    assert not v.check(t, "email", now=1)


def test_unknown_and_revoked_tokens_fail():
    v = L.TokenVault()
    assert not v.check("nope", "kb", now=0)
    t = v.issue("kb", now=0)["token"]
    v.revoke(t)
    assert not v.check(t, "kb", now=1)


def test_tokens_are_unique_and_unpredictable():
    v = L.TokenVault()
    assert len({v.issue("kb", now=0)["token"] for _ in range(50)}) == 50


def test_redact_hides_secrets_keeps_text():
    out = L.redact(f"ключ {FAKE_KEY}, Bearer abc.def-123, api_key=hunter2 и просто текст")
    assert FAKE_KEY not in out and "abc.def-123" not in out and "hunter2" not in out
    assert "просто текст" in out


def test_redact_covers_more_key_styles():
    out = L.redact("tvly-abcdefgh1234, token: abc123XYZ, password=qwerty123, secret=zzz9")
    for leaked in ("tvly-abcdefgh1234", "abc123XYZ", "qwerty123", "zzz9"):
        assert leaked not in out, leaked
