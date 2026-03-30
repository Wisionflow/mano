"""Patient manager — multi-patient data isolation and profile management.

Each patient has their own directory under data/patients/{patient_id}/ with:
- profile.json — personal info, diagnoses, medications, allergies, healthcare system
- health_diary.json — chronological symptom/feeling log
- lab_values.json — structured lab results with trends
- medications.json — current meds, courses, discontinued
- conversations.db — full conversation history (SQLite)
- chromadb/ — vector store for RAG
- documents/ — uploaded document copies
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base directory for all patient data
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "patients"

# Patient registry — maps telegram_user_id to patient_id
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "patient_registry.json"


def _load_registry() -> dict:
    """Load patient registry mapping telegram IDs to patient IDs."""
    if not REGISTRY_PATH.exists():
        return {"patients": {}, "access": {}}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {"patients": {}, "access": {}}


def _save_registry(registry: dict):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def register_patient(
    patient_id: str,
    patient_telegram_id: int,
    name: str,
    language: str = "ru",
    healthcare_system: str = "russia_moscow",
    family_members: dict = None,
) -> dict:
    """Register a new patient and create their data directory.

    Args:
        patient_id: Unique slug (e.g. "olga", "mama", "colleague-wife")
        patient_telegram_id: Telegram user ID of the patient
        name: Display name
        language: Interface language ("ru", "lt")
        healthcare_system: Healthcare system config to use
        family_members: Dict of {telegram_id: {"name": str, "role": str}}
    """
    registry = _load_registry()

    # Create patient entry
    registry["patients"][patient_id] = {
        "telegram_id": patient_telegram_id,
        "name": name,
        "language": language,
        "healthcare_system": healthcare_system,
        "created": datetime.now().isoformat(),
    }

    # Map telegram ID → patient_id + role
    registry["access"][str(patient_telegram_id)] = {
        "patient_id": patient_id,
        "role": "patient",
    }

    # Map family members
    if family_members:
        for tg_id, info in family_members.items():
            registry["access"][str(tg_id)] = {
                "patient_id": patient_id,
                "role": "family",
                "name": info.get("name", ""),
            }

    _save_registry(registry)

    # Create patient directory structure
    patient_dir = get_patient_dir(patient_id)
    (patient_dir / "documents").mkdir(parents=True, exist_ok=True)
    (patient_dir / "chromadb").mkdir(parents=True, exist_ok=True)

    # Create initial profile
    profile = _create_initial_profile(patient_id, name, language, healthcare_system)
    save_profile(patient_id, profile)

    # Initialize conversation DB
    _init_conversation_db(patient_id)

    logger.info("Registered patient '%s' (telegram_id=%s)", patient_id, patient_telegram_id)
    return profile


def get_patient_dir(patient_id: str) -> Path:
    """Get the data directory for a patient."""
    return DATA_ROOT / patient_id


def get_patient_id_by_telegram(telegram_user_id: int) -> Optional[str]:
    """Look up patient_id for a telegram user. Returns None if not registered."""
    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    if access:
        return access["patient_id"]
    return None


def get_user_role(telegram_user_id: int) -> Optional[str]:
    """Get role for telegram user: 'patient' or 'family'. None if not registered."""
    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    if access:
        return access["role"]
    return None


def get_user_name(telegram_user_id: int) -> Optional[str]:
    """Get display name for a family member. Returns None for patients."""
    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    if access and access["role"] == "family":
        return access.get("name")
    return None


def get_all_patients() -> dict:
    """Return all registered patients."""
    registry = _load_registry()
    return registry.get("patients", {})


# --- Profile ---

def _create_initial_profile(
    patient_id: str, name: str, language: str, healthcare_system: str
) -> dict:
    """Create a blank patient profile."""
    return {
        "patient_id": patient_id,
        "name": name,
        "language": language,
        "healthcare_system": healthcare_system,
        "date_of_birth": None,
        "gender": None,
        "height_cm": None,
        "weight_kg": None,
        "insurance": None,
        "address": None,
        "diagnoses": [],
        "allergies": [],
        "surgeries": [],
        "current_medications": [],
        "contraindicated": [],
        "doctors": [],
        "emergency_contacts": [],
        "notes": "",
        "profile_version": 1,
        "last_updated": datetime.now().isoformat(),
        "auto_extracted": False,
    }


def load_profile(patient_id: str) -> Optional[dict]:
    """Load patient profile."""
    profile_path = get_patient_dir(patient_id) / "profile.json"
    if not profile_path.exists():
        return None
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def save_profile(patient_id: str, profile: dict):
    """Save patient profile."""
    profile["last_updated"] = datetime.now().isoformat()
    profile_path = get_patient_dir(patient_id) / "profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_medical_summary(patient_id: str) -> str:
    """Build a MEDICAL_SUMMARY text from the patient profile (dynamic, not static file).

    This replaces the hardcoded MEDICAL_SUMMARY.md approach.
    """
    profile = load_profile(patient_id)
    if not profile:
        return ""

    parts = []

    # Basic info
    name = profile.get("name", "Пациент")
    parts.append(f"ПАЦИЕНТ: {name}")
    if profile.get("date_of_birth"):
        parts.append(f"Дата рождения: {profile['date_of_birth']}")
    if profile.get("gender"):
        parts.append(f"Пол: {profile['gender']}")
    if profile.get("height_cm"):
        parts.append(f"Рост: {profile['height_cm']} см")
    if profile.get("weight_kg"):
        parts.append(f"Вес: {profile['weight_kg']} кг")

    # Diagnoses
    if profile.get("diagnoses"):
        parts.append("\nДИАГНОЗЫ:")
        for d in profile["diagnoses"]:
            line = f"• {d['name']}"
            if d.get("icd_code"):
                line += f" ({d['icd_code']})"
            if d.get("year"):
                line += f", {d['year']}"
            if d.get("status"):
                line += f" — {d['status']}"
            parts.append(line)

    # Allergies
    if profile.get("allergies"):
        parts.append("\nАЛЛЕРГИИ / НЕПЕРЕНОСИМОСТЬ:")
        for a in profile["allergies"]:
            line = f"• {a['substance']}"
            if a.get("reaction"):
                line += f" — {a['reaction']}"
            parts.append(line)

    # Contraindicated
    if profile.get("contraindicated"):
        parts.append("\nПРОТИВОПОКАЗАНО:")
        for c in profile["contraindicated"]:
            line = f"• {c['substance']}"
            if c.get("reason"):
                line += f" — {c['reason']}"
            parts.append(line)

    # Current medications
    if profile.get("current_medications"):
        parts.append("\nТЕКУЩИЕ ЛЕКАРСТВА:")
        for m in profile["current_medications"]:
            line = f"• {m['name']}"
            if m.get("dose"):
                line += f" {m['dose']}"
            if m.get("frequency"):
                line += f", {m['frequency']}"
            if m.get("critical"):
                line += " ⚠️ НЕ ОТМЕНЯТЬ"
            parts.append(line)

    # Surgeries
    if profile.get("surgeries"):
        parts.append("\nОПЕРАЦИИ:")
        for s in profile["surgeries"]:
            line = f"• {s['name']}"
            if s.get("date"):
                line += f", {s['date']}"
            parts.append(line)

    # Doctors
    if profile.get("doctors"):
        parts.append("\nЛЕЧАЩИЕ ВРАЧИ:")
        for doc in profile["doctors"]:
            line = f"• {doc.get('specialty', '')}: {doc.get('name', '')}"
            if doc.get("clinic"):
                line += f", {doc['clinic']}"
            parts.append(line)

    # Notes
    if profile.get("notes"):
        parts.append(f"\nОСОБЫЕ ЗАМЕЧАНИЯ:\n{profile['notes']}")

    return "\n".join(parts)


def build_emergency_card(patient_id: str) -> str:
    """Generate emergency card dynamically from patient profile."""
    profile = load_profile(patient_id)
    if not profile:
        return "Профиль пациента не заполнен. Загрузите медицинские документы."

    lang = profile.get("language", "ru")
    name = profile.get("name", "—")

    lines = []
    if lang == "lt":
        lines.append("🚨 SKUBI KORTELĖ")
    else:
        lines.append("🚨 ЭКСТРЕННАЯ КАРТОЧКА")

    lines.append(f"\n{name}")
    if profile.get("date_of_birth"):
        lines.append(f"{'Gimimo data' if lang == 'lt' else 'Дата рождения'}: {profile['date_of_birth']}")
    if profile.get("height_cm"):
        lines.append(f"{'Ūgis' if lang == 'lt' else 'Рост'}: {profile['height_cm']} {'cm' if lang == 'lt' else 'см'}")
    if profile.get("weight_kg"):
        lines.append(f"{'Svoris' if lang == 'lt' else 'Вес'}: {profile['weight_kg']} {'kg' if lang == 'lt' else 'кг'}")
    if profile.get("insurance"):
        lines.append(f"{'Draudimas' if lang == 'lt' else 'Полис'}: {profile['insurance']}")

    # Diagnoses
    if profile.get("diagnoses"):
        lines.append(f"\n{'DIAGNOZĖS' if lang == 'lt' else 'ДИАГНОЗЫ'}:")
        for d in profile["diagnoses"]:
            line = f"• {d['name']}"
            if d.get("year"):
                line += f" ({d['year']})"
            if d.get("status"):
                line += f" — {d['status']}"
            lines.append(line)

    # Medications
    if profile.get("current_medications"):
        lines.append(f"\n{'VAISTAI DABAR' if lang == 'lt' else 'ЛЕКАРСТВА СЕЙЧАС'}:")
        for m in profile["current_medications"]:
            line = f"• {m['name']}"
            if m.get("dose"):
                line += f" {m['dose']}"
            lines.append(line)

    # Allergies + contraindicated
    forbidden = []
    for a in profile.get("allergies", []):
        entry = a["substance"]
        if a.get("reaction"):
            entry += f" — {a['reaction']}"
        forbidden.append(entry)
    for c in profile.get("contraindicated", []):
        entry = c["substance"]
        if c.get("reason"):
            entry += f" — {c['reason']}"
        forbidden.append(entry)
    if forbidden:
        lines.append(f"\n🚫 {'NEGALIMA' if lang == 'lt' else 'НЕЛЬЗЯ'}:")
        for f in forbidden:
            lines.append(f"• {f}")

    # Surgeries
    if profile.get("surgeries"):
        lines.append(f"\n{'OPERACIJOS' if lang == 'lt' else 'ОПЕРАЦИИ'}:")
        for s in profile["surgeries"]:
            line = f"• {s['name']}"
            if s.get("date"):
                line += f", {s['date']}"
            lines.append(line)

    # Emergency contacts
    if profile.get("emergency_contacts"):
        lines.append(f"\n{'KONTAKTAI' if lang == 'lt' else 'КОНТАКТЫ'}:")
        for c in profile["emergency_contacts"]:
            lines.append(f"• {c.get('relation', '')}: {c.get('name', '')} — {c.get('phone', '')}")

    return "\n".join(lines)


# --- Conversation Memory (SQLite) ---

def _get_conv_db_path(patient_id: str) -> Path:
    return get_patient_dir(patient_id) / "conversations.db"


def _init_conversation_db(patient_id: str):
    """Create conversation history table."""
    db_path = _get_conv_db_path(patient_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,
            sender_telegram_id INTEGER,
            sender_name TEXT,
            content TEXT NOT NULL,
            message_type TEXT DEFAULT 'text'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            summary TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_message(
    patient_id: str,
    role: str,
    content: str,
    sender_telegram_id: int = None,
    sender_name: str = None,
    message_type: str = "text",
):
    """Save a message to conversation history."""
    db_path = _get_conv_db_path(patient_id)
    if not db_path.exists():
        _init_conversation_db(patient_id)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO messages (timestamp, role, sender_telegram_id, sender_name, content, message_type) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            role,
            sender_telegram_id,
            sender_name,
            content,
            message_type,
        ),
    )
    conn.commit()
    conn.close()


def load_recent_messages(patient_id: str, limit: int = 20) -> list:
    """Load recent conversation messages for context."""
    db_path = _get_conv_db_path(patient_id)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT timestamp, role, sender_name, content, message_type "
        "FROM messages ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    messages = []
    for ts, role, sender, content, msg_type in reversed(rows):
        messages.append({
            "timestamp": ts,
            "role": role,
            "sender_name": sender,
            "content": content,
            "message_type": msg_type,
        })
    return messages


def get_conversation_context(patient_id: str, limit: int = 20) -> str:
    """Build conversation context string for the agent."""
    messages = load_recent_messages(patient_id, limit=limit)
    if not messages:
        return ""

    lines = ["ИСТОРИЯ ОБЩЕНИЯ (последние сообщения):"]
    for m in messages:
        ts = m["timestamp"][:16].replace("T", " ")
        sender = m["sender_name"] or ("Пациент" if m["role"] == "user" else "Ассистент")
        lines.append(f"[{ts}] {sender}: {m['content'][:500]}")

    return "\n".join(lines)


def search_conversations(patient_id: str, query: str, limit: int = 20) -> list:
    """Search conversation history by keyword."""
    db_path = _get_conv_db_path(patient_id)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT timestamp, role, sender_name, content FROM messages "
        "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
        (f"%{query}%", limit),
    ).fetchall()
    conn.close()

    return [
        {"timestamp": ts, "role": role, "sender_name": sender, "content": content}
        for ts, role, sender, content in rows
    ]
