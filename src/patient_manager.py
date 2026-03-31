"""Patient manager — multi-patient data isolation, access control, and profile management.

Registry v3: role-based access (owner/family), invite system, self-registration.

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
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base directory for all patient data
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "patients"

# Patient registry — maps telegram_user_id to patient_id
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "patient_registry.json"

CURRENT_VERSION = 3


def _load_registry() -> dict:
    """Load patient registry (v3: owner/family roles, invites)."""
    if not REGISTRY_PATH.exists():
        return {"version": CURRENT_VERSION, "patients": {}, "access": {}, "invites": {}}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        version = data.get("version", 1)
        if version < 2:
            data = _migrate_registry_v1_to_v2(data)
        if version < 3:
            data = _migrate_registry_v2_to_v3(data)
            _save_registry(data)
        return data
    except (json.JSONDecodeError, ValueError):
        return {"version": CURRENT_VERSION, "patients": {}, "access": {}, "invites": {}}


def _migrate_registry_v1_to_v2(old: dict) -> dict:
    """Migrate v1 registry (single patient per user) to v2 (multi-patient)."""
    new = {"version": 2, "patients": old.get("patients", {}), "access": {}}
    for tg_id, entry in old.get("access", {}).items():
        pid = entry.get("patient_id", "")
        role = entry.get("role", "family")
        name = entry.get("name", "")
        new["access"][tg_id] = {
            "name": name,
            "patients": {pid: role},
            "default_patient": pid,
        }
    return new


def _migrate_registry_v2_to_v3(old: dict) -> dict:
    """Migrate v2 → v3: add owner field to patients, normalize roles, add invites."""
    new = {
        "version": CURRENT_VERSION,
        "patients": old.get("patients", {}),
        "access": old.get("access", {}),
        "invites": {},
    }
    # For each patient, find who has role "patient" and make them "owner"
    for tg_id, access_entry in new["access"].items():
        patients = access_entry.get("patients", {})
        for pid, role in list(patients.items()):
            if role == "patient":
                patients[pid] = "owner"
                # Set owner on the patient record
                if pid in new["patients"]:
                    new["patients"][pid].setdefault("owner", tg_id)
    return new


def _save_registry(registry: dict):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --- Self-registration ---

def create_self_patient(
    telegram_user_id: int,
    name: str,
    language: str = "ru",
    healthcare_system: str = "russia_moscow",
) -> str:
    """Create a personal patient profile for a new user. Returns patient_id."""
    registry = _load_registry()
    tg_key = str(telegram_user_id)

    # Generate patient_id from name (slug)
    patient_id = _generate_patient_id(name, registry)

    # Create patient record
    registry["patients"][patient_id] = {
        "name": name,
        "language": language,
        "healthcare_system": healthcare_system,
        "owner": tg_key,
        "created_by": tg_key,
        "created": datetime.now().isoformat(),
    }

    # Create access entry
    if tg_key not in registry["access"]:
        registry["access"][tg_key] = {
            "name": name,
            "patients": {},
            "default_patient": patient_id,
        }
    registry["access"][tg_key]["patients"][patient_id] = "owner"
    if not registry["access"][tg_key].get("default_patient"):
        registry["access"][tg_key]["default_patient"] = patient_id

    _save_registry(registry)

    # Create patient directory structure
    patient_dir = get_patient_dir(patient_id)
    (patient_dir / "documents").mkdir(parents=True, exist_ok=True)
    (patient_dir / "chromadb").mkdir(parents=True, exist_ok=True)

    # Create initial profile
    profile = _create_initial_profile(patient_id, name, language, healthcare_system)
    save_profile(patient_id, profile)
    _init_conversation_db(patient_id)

    logger.info("Self-registered patient '%s' (telegram_id=%s)", patient_id, telegram_user_id)
    return patient_id


def create_patient_for_other(
    creator_telegram_id: int,
    patient_name: str,
    language: str = "ru",
    healthcare_system: str = "russia_moscow",
) -> str:
    """Create a patient profile for someone else. Creator gets 'family' role. Returns patient_id."""
    registry = _load_registry()
    tg_key = str(creator_telegram_id)

    patient_id = _generate_patient_id(patient_name, registry)

    # Create patient record — no owner yet (will be set when patient joins via invite)
    registry["patients"][patient_id] = {
        "name": patient_name,
        "language": language,
        "healthcare_system": healthcare_system,
        "owner": None,
        "created_by": tg_key,
        "created": datetime.now().isoformat(),
    }

    # Creator gets family role
    if tg_key not in registry["access"]:
        creator_name = ""  # will be set from telegram later
        registry["access"][tg_key] = {
            "name": creator_name,
            "patients": {},
            "default_patient": patient_id,
        }
    registry["access"][tg_key]["patients"][patient_id] = "family"

    _save_registry(registry)

    # Create patient directory
    patient_dir = get_patient_dir(patient_id)
    (patient_dir / "documents").mkdir(parents=True, exist_ok=True)
    (patient_dir / "chromadb").mkdir(parents=True, exist_ok=True)

    profile = _create_initial_profile(patient_id, patient_name, language, healthcare_system)
    save_profile(patient_id, profile)
    _init_conversation_db(patient_id)

    logger.info("Created patient '%s' for other (creator=%s)", patient_id, creator_telegram_id)
    return patient_id


def _generate_patient_id(name: str, registry: dict) -> str:
    """Generate a unique patient_id slug from name."""
    # Transliterate Russian to ASCII slug
    _translit = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    }
    slug = ""
    for ch in name.lower().strip():
        if ch in _translit:
            slug += _translit[ch]
        elif ch.isascii() and ch.isalnum():
            slug += ch
        elif ch in (' ', '-', '_'):
            slug += '-'
    slug = slug.strip('-') or "patient"

    # Take first name only for brevity
    slug = slug.split('-')[0]

    # Ensure uniqueness
    base = slug
    counter = 2
    while slug in registry.get("patients", {}):
        slug = f"{base}{counter}"
        counter += 1
    return slug


# --- Invite system ---

def create_invite(
    creator_telegram_id: int,
    patient_id: str,
    role: str = "family",
) -> Optional[str]:
    """Create an invite token for a patient. Returns token or None if not authorized.

    Only owner or family (if no owner yet) can create invites.
    role='owner' — invite the actual patient to become owner.
    role='family' — invite another family member.
    """
    registry = _load_registry()
    tg_key = str(creator_telegram_id)

    # Check creator has access
    access = registry.get("access", {}).get(tg_key)
    if not access or patient_id not in access.get("patients", {}):
        return None

    patient = registry.get("patients", {}).get(patient_id)
    if not patient:
        return None

    creator_role = access["patients"][patient_id]

    # Only owner can invite family. Family can invite owner (if no owner yet).
    if role == "family" and creator_role not in ("owner", "family"):
        return None
    if role == "owner" and patient.get("owner"):
        return None  # owner already exists

    token = secrets.token_urlsafe(12)
    registry.setdefault("invites", {})[token] = {
        "patient_id": patient_id,
        "role": role,
        "created_by": tg_key,
        "created": datetime.now().isoformat(),
    }
    _save_registry(registry)

    logger.info("Invite created for patient '%s' role=%s by %s", patient_id, role, tg_key)
    return token


def accept_invite(
    telegram_user_id: int,
    token: str,
    user_name: str = "",
) -> Optional[dict]:
    """Accept an invite. Returns {"patient_id", "role", "patient_name"} or None."""
    registry = _load_registry()
    tg_key = str(telegram_user_id)

    invite = registry.get("invites", {}).get(token)
    if not invite:
        return None

    patient_id = invite["patient_id"]
    role = invite["role"]

    patient = registry.get("patients", {}).get(patient_id)
    if not patient:
        return None

    # If role is owner and patient already has an owner — reject
    if role == "owner" and patient.get("owner"):
        return None

    # Add access
    if tg_key not in registry["access"]:
        registry["access"][tg_key] = {
            "name": user_name,
            "patients": {},
            "default_patient": patient_id,
        }
    registry["access"][tg_key]["patients"][patient_id] = role

    # Set owner on patient record
    if role == "owner":
        patient["owner"] = tg_key

    # Remove used invite
    del registry["invites"][token]

    _save_registry(registry)

    logger.info("Invite accepted: user %s → patient '%s' as %s", tg_key, patient_id, role)
    return {
        "patient_id": patient_id,
        "role": role,
        "patient_name": patient.get("name", patient_id),
    }


# --- Access management ---

def revoke_access(
    owner_telegram_id: int,
    patient_id: str,
    target_telegram_id: int,
) -> bool:
    """Revoke access to a patient. Only owner can revoke. Returns True on success."""
    registry = _load_registry()

    patient = registry.get("patients", {}).get(patient_id)
    if not patient:
        return False

    # Only owner can revoke
    if patient.get("owner") != str(owner_telegram_id):
        return False

    # Can't revoke yourself
    if owner_telegram_id == target_telegram_id:
        return False

    target_key = str(target_telegram_id)
    access = registry.get("access", {}).get(target_key)
    if not access or patient_id not in access.get("patients", {}):
        return False

    del access["patients"][patient_id]

    # If this was their default, reset
    if access.get("default_patient") == patient_id:
        remaining = list(access["patients"].keys())
        access["default_patient"] = remaining[0] if remaining else None

    _save_registry(registry)
    logger.info("Access revoked: user %s removed from patient '%s' by owner %s",
                target_key, patient_id, owner_telegram_id)
    return True


def get_patient_access_list(patient_id: str) -> list[dict]:
    """Get all users who have access to a patient. Returns [{telegram_id, name, role}]."""
    registry = _load_registry()
    result = []
    for tg_id, access in registry.get("access", {}).items():
        patients = access.get("patients", {})
        if patient_id in patients:
            result.append({
                "telegram_id": tg_id,
                "name": access.get("name", ""),
                "role": patients[patient_id],
            })
    return result


def is_owner(telegram_user_id: int, patient_id: str) -> bool:
    """Check if user is the owner of a patient."""
    registry = _load_registry()
    patient = registry.get("patients", {}).get(patient_id)
    if not patient:
        return False
    return patient.get("owner") == str(telegram_user_id)


def patient_has_owner(patient_id: str) -> bool:
    """Check if a patient has an owner assigned."""
    registry = _load_registry()
    patient = registry.get("patients", {}).get(patient_id)
    if not patient:
        return False
    return bool(patient.get("owner"))


# --- Original API (kept for compatibility) ---

def register_patient(
    patient_id: str,
    patient_telegram_id: int,
    name: str,
    language: str = "ru",
    healthcare_system: str = "russia_moscow",
    family_members: dict = None,
) -> dict:
    """Register a new patient and create their data directory (legacy API)."""
    registry = _load_registry()
    tg_key = str(patient_telegram_id)

    registry["patients"][patient_id] = {
        "name": name,
        "language": language,
        "healthcare_system": healthcare_system,
        "owner": tg_key,
        "created_by": tg_key,
        "created": datetime.now().isoformat(),
    }

    if tg_key not in registry["access"]:
        registry["access"][tg_key] = {
            "name": name,
            "patients": {},
            "default_patient": patient_id,
        }
    registry["access"][tg_key]["patients"][patient_id] = "owner"
    if not registry["access"][tg_key].get("default_patient"):
        registry["access"][tg_key]["default_patient"] = patient_id

    if family_members:
        for tg_id, info in family_members.items():
            fam_key = str(tg_id)
            if fam_key not in registry["access"]:
                registry["access"][fam_key] = {
                    "name": info.get("name", ""),
                    "patients": {},
                    "default_patient": patient_id,
                }
            registry["access"][fam_key]["patients"][patient_id] = "family"

    _save_registry(registry)

    patient_dir = get_patient_dir(patient_id)
    (patient_dir / "documents").mkdir(parents=True, exist_ok=True)
    (patient_dir / "chromadb").mkdir(parents=True, exist_ok=True)

    profile = _create_initial_profile(patient_id, name, language, healthcare_system)
    save_profile(patient_id, profile)
    _init_conversation_db(patient_id)

    logger.info("Registered patient '%s' (telegram_id=%s)", patient_id, patient_telegram_id)
    return profile


def get_patient_dir(patient_id: str) -> Path:
    """Get the data directory for a patient."""
    return DATA_ROOT / patient_id


# --- Active patient tracking (in-memory, per telegram user) ---
_active_patient: dict[int, str] = {}  # telegram_user_id → patient_id


def get_patient_id_by_telegram(telegram_user_id: int) -> Optional[str]:
    """Look up active patient_id for a telegram user.

    Priority: 1) explicitly switched, 2) default_patient from registry.
    """
    if telegram_user_id in _active_patient:
        return _active_patient[telegram_user_id]

    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    if not access:
        return None
    return access.get("default_patient")


def set_active_patient(telegram_user_id: int, patient_id: str) -> bool:
    """Switch active patient for a user. Returns False if user has no access."""
    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    if not access:
        return False
    if patient_id not in access.get("patients", {}):
        return False
    _active_patient[telegram_user_id] = patient_id
    return True


def get_accessible_patients(telegram_user_id: int) -> dict:
    """Get all patients this user has access to. Returns {patient_id: role}."""
    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    if not access:
        return {}
    return access.get("patients", {})


def get_user_role(telegram_user_id: int, patient_id: str = None) -> Optional[str]:
    """Get role for telegram user for a specific patient (or active patient)."""
    if patient_id is None:
        patient_id = get_patient_id_by_telegram(telegram_user_id)
    if not patient_id:
        return None
    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    if not access:
        return None
    return access.get("patients", {}).get(patient_id)


def get_user_name(telegram_user_id: int) -> Optional[str]:
    """Get display name for a user."""
    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    if access:
        return access.get("name")
    return None


def update_user_name(telegram_user_id: int, name: str):
    """Update display name for a user in registry."""
    registry = _load_registry()
    tg_key = str(telegram_user_id)
    access = registry.get("access", {}).get(tg_key)
    if access:
        access["name"] = name
        _save_registry(registry)


def get_patient_name(patient_id: str) -> Optional[str]:
    """Get display name for a patient."""
    registry = _load_registry()
    patient = registry.get("patients", {}).get(patient_id)
    if patient:
        return patient.get("name")
    return None


def add_patient_access(telegram_user_id: int, patient_id: str, role: str, user_name: str = None):
    """Grant a user access to a patient."""
    registry = _load_registry()
    tg_key = str(telegram_user_id)
    if tg_key not in registry["access"]:
        registry["access"][tg_key] = {
            "name": user_name or "",
            "patients": {},
            "default_patient": patient_id,
        }
    registry["access"][tg_key]["patients"][patient_id] = role
    if not registry["access"][tg_key].get("default_patient"):
        registry["access"][tg_key]["default_patient"] = patient_id
    _save_registry(registry)


def get_all_patients() -> dict:
    """Return all registered patients."""
    registry = _load_registry()
    return registry.get("patients", {})


def is_registered(telegram_user_id: int) -> bool:
    """Check if a telegram user is registered (has any access)."""
    registry = _load_registry()
    access = registry.get("access", {}).get(str(telegram_user_id))
    return bool(access and access.get("patients"))


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
    """Build a MEDICAL_SUMMARY text from the patient profile."""
    profile = load_profile(patient_id)
    if not profile:
        return ""

    parts = []

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

    if profile.get("allergies"):
        parts.append("\nАЛЛЕРГИИ / НЕПЕРЕНОСИМОСТЬ:")
        for a in profile["allergies"]:
            line = f"• {a['substance']}"
            if a.get("reaction"):
                line += f" — {a['reaction']}"
            parts.append(line)

    if profile.get("contraindicated"):
        parts.append("\nПРОТИВОПОКАЗАНО:")
        for c in profile["contraindicated"]:
            line = f"• {c['substance']}"
            if c.get("reason"):
                line += f" — {c['reason']}"
            parts.append(line)

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

    if profile.get("surgeries"):
        parts.append("\nОПЕРАЦИИ:")
        for s in profile["surgeries"]:
            line = f"• {s['name']}"
            if s.get("date"):
                line += f", {s['date']}"
            parts.append(line)

    if profile.get("doctors"):
        parts.append("\nЛЕЧАЩИЕ ВРАЧИ:")
        for doc in profile["doctors"]:
            line = f"• {doc.get('specialty', '')}: {doc.get('name', '')}"
            if doc.get("clinic"):
                line += f", {doc['clinic']}"
            parts.append(line)

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

    if profile.get("diagnoses"):
        lines.append(f"\n{'DIAGNOZĖS' if lang == 'lt' else 'ДИАГНОЗЫ'}:")
        for d in profile["diagnoses"]:
            line = f"• {d['name']}"
            if d.get("year"):
                line += f" ({d['year']})"
            if d.get("status"):
                line += f" — {d['status']}"
            lines.append(line)

    if profile.get("current_medications"):
        lines.append(f"\n{'VAISTAI DABAR' if lang == 'lt' else 'ЛЕКАРСТВА СЕЙЧАС'}:")
        for m in profile["current_medications"]:
            line = f"• {m['name']}"
            if m.get("dose"):
                line += f" {m['dose']}"
            lines.append(line)

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

    if profile.get("surgeries"):
        lines.append(f"\n{'OPERACIJOS' if lang == 'lt' else 'ОПЕРАЦИИ'}:")
        for s in profile["surgeries"]:
            line = f"• {s['name']}"
            if s.get("date"):
                line += f", {s['date']}"
            lines.append(line)

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
