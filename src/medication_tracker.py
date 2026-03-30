"""Medication tracker — tracks current medications, courses, and discontinued drugs.

Stores per-patient medication data:
- current: ongoing medications (with or without end date)
- courses: time-limited medication courses with start/end/result
- discontinued: medications stopped due to side effects, allergy, etc.

Telegram commands:
    /meds — show current medications
    /meds add Канефрон 2т×3р 4 недели — add a course
    /meds stop Канефрон результат: без эффекта — complete a course
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _meds_path(patient_id: str) -> Path:
    from src.patient_manager import get_patient_dir
    return get_patient_dir(patient_id) / "medications.json"


def _load(patient_id: str) -> dict:
    path = _meds_path(patient_id)
    if not path.exists():
        return {"current": [], "courses": [], "discontinued": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {"current": [], "courses": [], "discontinued": []}


def _save(patient_id: str, data: dict):
    path = _meds_path(patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_medication(
    patient_id: str,
    name: str,
    dose: str = "",
    frequency: str = "",
    prescribed_by: str = "",
    duration_weeks: int = 0,
    purpose: str = "",
    critical: bool = False,
) -> dict:
    """Add a medication — either ongoing or as a course with duration."""
    data = _load(patient_id)
    today = datetime.now().strftime("%Y-%m-%d")

    entry = {
        "name": name,
        "dose": dose,
        "frequency": frequency,
        "prescribed_by": prescribed_by,
        "started": today,
        "purpose": purpose,
        "added": datetime.now().isoformat(),
    }

    if duration_weeks > 0:
        end_date = (datetime.now() + timedelta(weeks=duration_weeks)).strftime("%Y-%m-%d")
        entry["end_date"] = end_date
        entry["status"] = "active"
        entry["result"] = ""
        data["courses"].append(entry)
    else:
        entry["end_date"] = None
        entry["critical"] = critical
        data["current"].append(entry)

    _save(patient_id, data)
    return entry


def stop_medication(patient_id: str, name: str, reason: str = "", result: str = "") -> bool:
    """Stop a medication or complete a course. Returns True if found."""
    data = _load(patient_id)
    name_lower = name.lower().strip()
    found = False

    # Check current medications
    for i, med in enumerate(data["current"]):
        if med["name"].lower().strip() == name_lower:
            removed = data["current"].pop(i)
            removed["stopped"] = datetime.now().strftime("%Y-%m-%d")
            removed["reason"] = reason
            data["discontinued"].append(removed)
            found = True
            break

    # Check active courses
    if not found:
        for course in data["courses"]:
            if course["name"].lower().strip() == name_lower and course.get("status") == "active":
                course["status"] = "completed"
                course["stopped"] = datetime.now().strftime("%Y-%m-%d")
                course["result"] = result or reason
                found = True
                break

    if found:
        _save(patient_id, data)
    return found


def get_current_medications(patient_id: str) -> list:
    """Get all current (ongoing) medications."""
    data = _load(patient_id)
    return data.get("current", [])


def get_active_courses(patient_id: str) -> list:
    """Get all active (not completed) courses."""
    data = _load(patient_id)
    return [c for c in data.get("courses", []) if c.get("status") == "active"]


def get_expiring_courses(patient_id: str, days_ahead: int = 7) -> list:
    """Get courses ending within N days."""
    data = _load(patient_id)
    cutoff = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    expiring = []
    for c in data.get("courses", []):
        if c.get("status") == "active" and c.get("end_date"):
            if today <= c["end_date"] <= cutoff:
                expiring.append(c)
    return expiring


def format_medications(patient_id: str, lang: str = "ru") -> str:
    """Format all medications for display."""
    data = _load(patient_id)
    current = data.get("current", [])
    courses = [c for c in data.get("courses", []) if c.get("status") == "active"]
    discontinued = data.get("discontinued", [])[-5:]  # last 5

    if not current and not courses:
        if lang == "lt":
            return "Vaistų sąrašas tuščias. Naudokite /meds add arba siųskite recepto nuotrauką."
        return "Список лекарств пуст. Используйте /meds add или отправьте фото рецепта."

    lines = []

    if current:
        lines.append("💊 Постоянные:" if lang == "ru" else "💊 Nuolatiniai:")
        for m in current:
            line = f"  • {m['name']}"
            if m.get("dose"):
                line += f" {m['dose']}"
            if m.get("frequency"):
                line += f", {m['frequency']}"
            if m.get("critical"):
                line += " ⚠️"
            lines.append(line)

    if courses:
        lines.append(f"\n{'📅 Курсы:' if lang == 'ru' else '📅 Kursai:'}")
        for c in courses:
            line = f"  • {c['name']}"
            if c.get("dose"):
                line += f" {c['dose']}"
            if c.get("end_date"):
                days_left = (datetime.strptime(c["end_date"], "%Y-%m-%d") - datetime.now()).days
                if days_left > 0:
                    line += f" (осталось {days_left} дн.)" if lang == "ru" else f" (liko {days_left} d.)"
                elif days_left == 0:
                    line += " (последний день!)" if lang == "ru" else " (paskutinė diena!)"
                else:
                    line += " (просрочен)" if lang == "ru" else " (pasibaigęs)"
            lines.append(line)

    # Expiring soon
    expiring = get_expiring_courses(patient_id, days_ahead=3)
    if expiring:
        lines.append(f"\n⏰ {'Скоро заканчиваются:' if lang == 'ru' else 'Greitai baigiasi:'}")
        for c in expiring:
            days_left = (datetime.strptime(c["end_date"], "%Y-%m-%d") - datetime.now()).days
            lines.append(f"  • {c['name']} — {days_left} дн." if lang == "ru" else f"  • {c['name']} — {days_left} d.")

    if discontinued:
        lines.append(f"\n{'🚫 Отменённые (последние):' if lang == 'ru' else '🚫 Atšaukti (paskutiniai):'}")
        for d in discontinued:
            line = f"  • {d['name']}"
            if d.get("reason"):
                line += f" — {d['reason']}"
            lines.append(line)

    return "\n".join(lines)


def parse_meds_command(text: str) -> dict:
    """Parse /meds subcommand text.

    Examples:
        "add Канефрон 2т×3р 4 недели" → {"action": "add", "name": "Канефрон", "dose": "2т×3р", "weeks": 4}
        "stop Канефрон результат: без эффекта" → {"action": "stop", "name": "Канефрон", "result": "без эффекта"}
        "" → {"action": "list"}
    """
    text = text.strip()
    if not text:
        return {"action": "list"}

    parts = text.split(None, 1)
    action = parts[0].lower()

    if action == "add" and len(parts) > 1:
        rest = parts[1].strip()
        # Try to extract name, dose, duration
        tokens = rest.split()
        name = tokens[0] if tokens else rest
        dose = ""
        weeks = 0

        # Look for duration pattern (N недел/нед)
        for i, t in enumerate(tokens):
            t_lower = t.lower()
            if t_lower.isdigit() and i + 1 < len(tokens) and "недел" in tokens[i + 1].lower():
                weeks = int(t_lower)
                # dose is everything between name and duration
                dose = " ".join(tokens[1:i])
                break
        else:
            dose = " ".join(tokens[1:]) if len(tokens) > 1 else ""

        return {"action": "add", "name": name, "dose": dose, "weeks": weeks}

    elif action == "stop" and len(parts) > 1:
        rest = parts[1].strip()
        # Split on "результат:" or "причина:"
        result = ""
        name = rest
        for sep in ["результат:", "причина:", "result:", "reason:"]:
            if sep in rest.lower():
                idx = rest.lower().index(sep)
                name = rest[:idx].strip()
                result = rest[idx + len(sep):].strip()
                break
        return {"action": "stop", "name": name, "result": result}

    return {"action": "list"}
