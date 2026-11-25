# -*- coding: utf-8 -*-
"""Слой 5. Шлюз инструментов: код решает, что агенту можно (глава V).

Закрывает LLM03 Excessive Agency и LLM06 Unbounded Consumption: модель можно
уговорить на что угодно, поэтому решение «можно ли» принимает код вне модели,
по явным правилам. Правило по умолчанию — «нельзя» (fail-safe default).

  decide(tool, args) -> "allow" | "ask" | "deny":
      search_knowledge_base / knowledge_base — allow
      web_search            — allow, но запрос длиннее 200 символов или
                              содержащий «http» — deny (через веб-поиск нельзя
                              протаскивать адреса и простыни текста)
      read_file             — allow только для путей внутри docs/
                              (posixpath.normpath; «..», абсолютные пути и
                              любые .env — deny)
      send_email            — адрес на @company.example: ask (нужно
                              подтверждение человека); любой другой: deny
      write_file            — путь внутри drafts/: ask, иначе deny
      run_shell             — deny
      неизвестный инструмент — deny

  Аргументы распознаются и в формате OpenAI-tools ({"path": ...}), и в формате
  ReAct-агента ({"query": ...}) — правила одни и те же.

  Budget(max_steps) — не больше max_steps обращений к модели за один запрос
  пользователя. spend() возвращает True, пока лимит не исчерпан.
"""
import posixpath


def decide(tool: str, args: dict) -> str:
    args = args or {}
    query = str(args.get("query") or "")

    if tool in ("search_knowledge_base", "knowledge_base"):
        return "allow"

    if tool == "web_search":
        if len(query) > 200 or "http" in query.lower():
            return "deny"
        return "allow"

    if tool == "read_file":
        path = str(args.get("path") or query or "")
        norm = posixpath.normpath(path)
        if norm.startswith("/") or norm.startswith(".."):
            return "deny"
        norm = norm.rstrip("/")
        if not norm.startswith("docs/") or norm == "docs":
            return "deny"
        if posixpath.basename(norm) == ".env":
            return "deny"
        return "allow"

    if tool == "send_email":
        to = str(args.get("to") or query or "").strip().lower()
        return "ask" if to.endswith("@company.example") else "deny"

    if tool == "write_file":
        path = str(args.get("path") or query or "")
        norm = posixpath.normpath(path)
        if norm.startswith("/") or norm.startswith(".."):
            return "deny"
        norm = norm.rstrip("/")
        if norm.startswith("drafts/") and norm != "drafts":
            return "ask"
        return "deny"

    # run_shell и всё, что не перечислено выше, — нельзя.
    return "deny"


class Budget:
    """Лимит обращений к модели за один запрос пользователя."""

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self._spent = 0

    def spend(self) -> bool:
        if self._spent >= self.max_steps:
            return False
        self._spent += 1
        return True
