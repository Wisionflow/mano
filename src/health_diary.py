"""Health diary — chronological log of how the patient feels."""

import json
import re
from datetime import datetime
from pathlib import Path

DIARY_PATH = Path(__file__).resolve().parent.parent / "data" / "health_diary.json"

# Keywords that indicate the patient is describing how she feels
FEELING_KEYWORDS = [
    "чувствую", "чувствовала", "самочувствие",
    "болит", "болело", "боль", "боли",
    "тошнит", "тошнота", "рвота",
    "слабость", "устала", "усталость", "утомляемость",
    "кружится", "головокружение",
    "отёк", "отек", "отекл", "опухл",
    "давление", "померила давление", "давление было",
    "температура", "темпер",
    "плохо", "стало плохо", "нехорошо",
    "не спала", "бессонница", "плохо спала",
    "одышка", "задыхаюсь", "не хватает воздуха",
    "зуд", "чешется", "чесотка",
    "понос", "запор", "стул",
    "паника", "паническая", "тревога", "тревожно",
    "отекает", "отекают", "припухл",
    "моча", "кровь в моче", "мутная моча",
    "сегодня", "вчера", "утром", "вечером", "ночью",
    "лучше", "хуже", "получше", "похуже",
    "приняла", "выпила", "не пила", "пропустила",
]

# Patterns that are clearly questions, not status reports
QUESTION_PATTERNS = [
    r"\?$",
    r"^можно\s",
    r"^а\s(можно|что|как|почему|зачем)",
    r"^что\s(такое|значит|делать)",
    r"^как\s(принимать|пить|часто)",
    r"^почему\s",
    r"^зачем\s",
    r"^сколько\s",
]


def is_health_status(text: str) -> bool:
    """Detect if the message is about how the patient feels (not a question)."""
    lower = text.lower().strip()

    # Short messages are usually commands or quick questions
    if len(lower) < 10:
        return False

    # If it's clearly a question — don't log as diary
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, lower):
            # But even questions can contain health status: "болит голова, что делать?"
            # Check if there are enough feeling keywords
            keyword_count = sum(1 for kw in FEELING_KEYWORDS if kw in lower)
            if keyword_count < 2:
                return False

    # Check for feeling keywords
    keyword_count = sum(1 for kw in FEELING_KEYWORDS if kw in lower)
    return keyword_count >= 1


def save_entry(text: str, reported_by: str = None) -> dict:
    """Save a diary entry with timestamp. reported_by = family member name if not patient."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text": text.strip(),
    }
    if reported_by:
        entry["reported_by"] = reported_by

    entries = load_entries()
    entries.append(entry)

    DIARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIARY_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return entry


def load_entries(last_n: int = 0) -> list:
    """Load diary entries. If last_n > 0, return only last N entries."""
    if not DIARY_PATH.exists():
        return []

    try:
        entries = json.loads(DIARY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []

    if last_n > 0:
        return entries[-last_n:]
    return entries


def format_entries(entries: list) -> str:
    """Format diary entries for display."""
    if not entries:
        return "Дневник пока пуст."

    lines = []
    for e in entries:
        prefix = f"[{e.get('reported_by', '')}] " if e.get("reported_by") else ""
        lines.append(f"{e['timestamp']}  —  {prefix}{e['text']}")

    return "\n\n".join(lines)


def get_diary_context(last_n: int = 10) -> str:
    """Get recent diary entries as context for the agent."""
    entries = load_entries(last_n=last_n)
    if not entries:
        return ""

    lines = [f"ДНЕВНИК САМОЧУВСТВИЯ (последние {len(entries)} записей):"]
    for e in entries:
        by = f" (со слов {e['reported_by']})" if e.get("reported_by") else ""
        lines.append(f"[{e['timestamp']}]{by} {e['text']}")

    return "\n".join(lines)
