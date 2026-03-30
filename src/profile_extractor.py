"""Auto-extract patient profile data from medical documents using Claude.

When a patient uploads documents, this module analyzes the text and
extracts structured medical data (diagnoses, medications, allergies, etc.)
to build and incrementally update the patient profile.
"""

import json
import logging
import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from src.patient_manager import load_profile, save_profile

load_dotenv()
logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Ты — медицинский аналитик. Из текста медицинского документа извлеки
структурированные данные о пациенте. Извлекай ТОЛЬКО то, что ЯВНО указано в тексте.
Не додумывай, не предполагай.

ФОРМАТ ОТВЕТА — строго JSON (без markdown, без комментариев):
{
  "name": "ФИО если найдено" или null,
  "date_of_birth": "ДД.ММ.ГГГГ" или null,
  "gender": "ж" или "м" или null,
  "height_cm": число или null,
  "weight_kg": число или null,
  "insurance": "номер полиса" или null,
  "address": "адрес" или null,
  "diagnoses": [
    {"name": "название", "icd_code": "код МКБ или null", "year": "год или null", "status": "ремиссия/активный/хронический или null"}
  ],
  "allergies": [
    {"substance": "вещество", "reaction": "описание реакции или null"}
  ],
  "contraindicated": [
    {"substance": "вещество", "reason": "причина"}
  ],
  "surgeries": [
    {"name": "операция", "date": "дата или год"}
  ],
  "current_medications": [
    {"name": "препарат", "dose": "доза", "frequency": "частота приёма", "critical": true/false}
  ],
  "doctors": [
    {"specialty": "специальность", "name": "ФИО врача", "clinic": "учреждение"}
  ],
  "emergency_contacts": [
    {"name": "имя", "relation": "кем приходится", "phone": "телефон"}
  ],
  "notes": "важные замечания не вошедшие в категории выше"
}

Если поле не найдено в документе — ставь null или пустой массив [].
Поле "critical" = true только для лекарств, отмена которых опасна.

ТЕКСТ ДОКУМЕНТА:
{text}"""


def extract_profile_data(text: str) -> Optional[dict]:
    """Extract structured medical data from document text using Claude."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return None

    client = Anthropic(api_key=api_key, timeout=120.0, max_retries=3)
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    # Limit text to avoid excessive token usage
    truncated = text[:15000]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(text=truncated),
            }],
        )
        raw = response.content[0].text.strip()

        # Parse JSON — handle cases where model wraps in ```json blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]

        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error("Failed to parse extraction result as JSON: %s", e)
        return None
    except Exception as e:
        logger.error("Profile extraction failed: %s", e, exc_info=True)
        return None


def merge_into_profile(patient_id: str, extracted: dict) -> dict:
    """Merge extracted data into existing patient profile.

    Rules:
    - Scalar fields (name, dob, etc.): fill if currently empty, don't overwrite
    - List fields (diagnoses, meds, etc.): append new items, avoid duplicates
    - Returns updated profile
    """
    profile = load_profile(patient_id)
    if not profile:
        logger.error("Profile not found for %s", patient_id)
        return {}

    # Scalar fields — fill blanks
    for field in ["name", "date_of_birth", "gender", "height_cm", "weight_kg", "insurance", "address"]:
        if not profile.get(field) and extracted.get(field):
            profile[field] = extracted[field]

    # Notes — append
    if extracted.get("notes"):
        existing_notes = profile.get("notes", "")
        new_note = extracted["notes"]
        if new_note not in existing_notes:
            profile["notes"] = (existing_notes + "\n" + new_note).strip()

    # List fields — append unique
    _merge_list(profile, extracted, "diagnoses", key_field="name")
    _merge_list(profile, extracted, "allergies", key_field="substance")
    _merge_list(profile, extracted, "contraindicated", key_field="substance")
    _merge_list(profile, extracted, "surgeries", key_field="name")
    _merge_list(profile, extracted, "current_medications", key_field="name")
    _merge_list(profile, extracted, "doctors", key_field="name")
    _merge_list(profile, extracted, "emergency_contacts", key_field="name")

    profile["auto_extracted"] = True
    save_profile(patient_id, profile)

    logger.info("Profile updated for %s", patient_id)
    return profile


def _merge_list(profile: dict, extracted: dict, field: str, key_field: str):
    """Merge a list field, avoiding duplicates by key_field."""
    existing = profile.get(field, [])
    new_items = extracted.get(field, [])

    if not new_items:
        return

    existing_keys = {
        item.get(key_field, "").lower().strip()
        for item in existing
        if item.get(key_field)
    }

    for item in new_items:
        key = item.get(key_field, "").lower().strip()
        if key and key not in existing_keys:
            existing.append(item)
            existing_keys.add(key)

    profile[field] = existing


def process_document_for_profile(patient_id: str, document_text: str) -> dict | None:
    """Full pipeline: extract from document text → merge into profile.

    Returns updated profile or None on failure.
    """
    extracted = extract_profile_data(document_text)
    if not extracted:
        return None

    return merge_into_profile(patient_id, extracted)
