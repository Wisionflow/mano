"""Lab value tracker — structured storage and trend analysis."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "lab_values.json"


def _data_path(patient_id: str = None) -> Path:
    """Get lab values path — patient-specific if patient_id given."""
    if patient_id:
        from src.patient_manager import get_patient_dir
        return get_patient_dir(patient_id) / "lab_values.json"
    return DATA_PATH

# Key lab parameters to track with units and normal ranges
LAB_PARAMS = {
    "креатинин": {"unit": "мкмоль/л", "norm_min": 44, "norm_max": 96, "critical_high": 200},
    "мочевина": {"unit": "мМоль/л", "norm_min": 2.8, "norm_max": 7.2},
    "гемоглобин": {"unit": "г/л", "norm_min": 120, "norm_max": 140},
    "глюкоза": {"unit": "мМоль/л", "norm_min": 4.1, "norm_max": 5.9},
    "алт": {"unit": "Ед/л", "norm_min": 0, "norm_max": 41},
    "аст": {"unit": "Ед/л", "norm_min": 0, "norm_max": 40},
    "общий белок": {"unit": "г/л", "norm_min": 66, "norm_max": 83},
    "калий": {"unit": "мМоль/л", "norm_min": 3.5, "norm_max": 5.1, "critical_high": 6.0},
    "натрий": {"unit": "мМоль/л", "norm_min": 136, "norm_max": 145},
    "кальций": {"unit": "мМоль/л", "norm_min": 2.15, "norm_max": 2.55},
    "фосфор": {"unit": "мМоль/л", "norm_min": 0.81, "norm_max": 1.45},
    "мочевая кислота": {"unit": "мкмоль/л", "norm_min": 155, "norm_max": 357},
    "железо": {"unit": "мкмоль/л", "norm_min": 9, "norm_max": 30.4},
    "ферритин": {"unit": "мкг/л", "norm_min": 10, "norm_max": 120},
    "соэ": {"unit": "мм/ч", "norm_min": 0, "norm_max": 20},
    "лейкоциты": {"unit": "×10⁹/л", "norm_min": 4.0, "norm_max": 9.0},
    "тромбоциты": {"unit": "×10⁹/л", "norm_min": 180, "norm_max": 320},
    "билирубин общий": {"unit": "мкмоль/л", "norm_min": 3.4, "norm_max": 20.5},
    "щф": {"unit": "Ед/л", "norm_min": 35, "norm_max": 105},
    "ттг": {"unit": "мМЕ/л", "norm_min": 0.27, "norm_max": 4.2},
    "паратгормон": {"unit": "пг/мл", "norm_min": 15, "norm_max": 65},
    "рскф": {"unit": "мл/мин", "norm_min": 60, "norm_max": 120, "critical_low": 15},
    "оксалаты": {"unit": "количество/мкл", "norm_min": 0, "norm_max": 40},
}

# Aliases for matching
ALIASES = {
    "крeat": "креатинин", "creatinine": "креатинин", "креат": "креатинин",
    "urea": "мочевина",
    "hb": "гемоглобин", "hemoglobin": "гемоглобин", "гемогл": "гемоглобин",
    "glucose": "глюкоза", "глюк": "глюкоза",
    "alt": "алт", "алат": "алт", "аланинаминотрансфераза": "алт",
    "ast": "аст", "асат": "аст", "аспартатаминотрансфераза": "аст",
    "белок общий": "общий белок",
    "k+": "калий", "калий общий": "калий",
    "na+": "натрий", "натрий общий": "натрий",
    "ca": "кальций", "кальций общий": "кальций", "ca2+": "кальций",
    "p": "фосфор", "фосфор неорганический": "фосфор",
    "fe": "железо",
    "esr": "соэ", "скорость оседания": "соэ",
    "wbc": "лейкоциты",
    "plt": "тромбоциты",
    "tsh": "ттг", "тиреотропин": "ттг", "тиреотропный": "ттг",
    "пт": "паратгормон", "паратирин": "паратгормон", "птг": "паратгормон",
    "скф": "рскф", "gfr": "рскф", "клубочковая фильтрация": "рскф",
    "щелочная фосфатаза": "щф",
    "оксалат кальция": "оксалаты", "кристаллы оксалата": "оксалаты",
    # Lithuanian aliases
    "kreatininas": "креатинин",
    "hemoglobinas": "гемоглобин", "hgb": "гемоглобин",
    "gliukozė": "глюкоза", "gliukoze": "глюкоза", "gli": "глюкоза",
    "šlapimas": "мочевина", "slapimas": "мочевина",
    "kalcis": "кальций",
    "fosforas": "фосфор",
    "kalio": "калий",
    "natrio": "натрий",
    "geležis": "железо", "gelezis": "железо",
    "feritinas": "ферритин",
    "eng": "соэ",
    "trombocitai": "тромбоциты",
    "leukocitai": "лейкоциты",
    "gfg": "рскф",
    "crb": "общий белок",
}


def _load(patient_id: str = None) -> list:
    path = _data_path(patient_id)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []


def _save(entries: list, patient_id: str = None):
    path = _data_path(patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _normalize_param(name: str) -> Optional[str]:
    """Match a parameter name to our tracked list."""
    name = name.lower().strip()
    # Direct match
    if name in LAB_PARAMS:
        return name
    # Alias match
    if name in ALIASES:
        return ALIASES[name]
    # Partial match
    for key in LAB_PARAMS:
        if key in name or name in key:
            return key
    for alias, key in ALIASES.items():
        if alias in name or name in alias:
            return key
    return None


def extract_lab_values(text: str) -> list:
    """Extract lab values from OCR text. Returns list of {param, value, date, unit}."""
    results = []

    # Try to find date in the text
    date_match = re.search(
        r'(\d{1,2})[./](\d{1,2})[./](20\d{2})', text
    )
    date_str = None
    if date_match:
        d, m, y = date_match.groups()
        date_str = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # Pattern: parameter_name ... number (with optional units)
    # Handles: "Креатинин 155.8 мкмоль/л", "АЛТ: 48.00", "рСКФ - 32"
    lines = text.split("\n")
    for line in lines:
        line_lower = line.lower().strip()
        if not line_lower:
            continue

        # Try to match known parameters in this line
        for param_name in list(LAB_PARAMS.keys()) + list(ALIASES.keys()):
            if param_name in line_lower:
                # Find a number after the parameter name
                idx = line_lower.index(param_name)
                after = line[idx + len(param_name):]
                num_match = re.search(r'[\s:=\-–—]*(\d+[.,]?\d*)', after)
                if num_match:
                    value_str = num_match.group(1).replace(",", ".")
                    try:
                        value = float(value_str)
                    except ValueError:
                        continue

                    normalized = _normalize_param(param_name)
                    if normalized and value > 0:
                        # Avoid duplicates in same extraction
                        if not any(r["param"] == normalized and r["value"] == value for r in results):
                            results.append({
                                "param": normalized,
                                "value": value,
                                "date": date_str or datetime.now().strftime("%Y-%m-%d"),
                                "unit": LAB_PARAMS[normalized]["unit"],
                            })
                break  # One match per line is enough

    return results


def save_lab_values(values: list, patient_id: str = None) -> int:
    """Save extracted lab values to storage. Returns count of new values saved."""
    if not values:
        return 0

    entries = _load(patient_id)
    saved = 0

    for v in values:
        # Check for duplicates (same param + date + value)
        is_dup = any(
            e["param"] == v["param"] and e["date"] == v["date"] and e["value"] == v["value"]
            for e in entries
        )
        if not is_dup:
            entries.append({
                "param": v["param"],
                "value": v["value"],
                "date": v["date"],
                "unit": v["unit"],
                "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            saved += 1

    if saved:
        _save(entries, patient_id)
    return saved


def get_param_history(param: str, last_n: int = 10, patient_id: str = None) -> list:
    """Get history of a specific parameter, sorted by date."""
    normalized = _normalize_param(param)
    if not normalized:
        return []

    entries = _load(patient_id)
    history = [e for e in entries if e["param"] == normalized]
    history.sort(key=lambda x: x["date"])

    if last_n:
        history = history[-last_n:]
    return history


def get_all_latest(patient_id: str = None) -> dict:
    """Get the most recent value for each tracked parameter."""
    entries = _load(patient_id)
    latest = {}
    for e in sorted(entries, key=lambda x: x["date"]):
        latest[e["param"]] = e

    return latest


def format_trends(patient_id: str = None) -> str:
    """Format all tracked parameters with trends for display."""
    latest = get_all_latest(patient_id)
    if not latest:
        return "Трекер показателей пуст. Отправь фото анализа — я извлеку цифры."

    lines = ["📊 Показатели:\n"]

    # Group by priority: kidney first, then blood, then other
    kidney = ["креатинин", "рскф", "мочевина", "калий", "оксалаты"]
    blood = ["гемоглобин", "лейкоциты", "тромбоциты", "соэ"]
    liver = ["алт", "аст", "билирубин общий", "щф"]

    def _format_param(param_name):
        if param_name not in latest:
            return None
        e = latest[param_name]
        info = LAB_PARAMS.get(param_name, {})
        value = e["value"]
        date = e["date"]

        # Trend arrow
        history = get_param_history(param_name, last_n=3, patient_id=patient_id)
        trend = ""
        if len(history) >= 2:
            prev = history[-2]["value"]
            if value > prev * 1.05:
                trend = " ↑"
            elif value < prev * 0.95:
                trend = " ↓"
            else:
                trend = " →"

        # Normal range check
        flag = ""
        if info.get("norm_max") and value > info["norm_max"]:
            flag = " ⚠️"
            if info.get("critical_high") and value > info["critical_high"]:
                flag = " 🚨"
        elif info.get("norm_min") and value < info["norm_min"]:
            flag = " ⚠️"
            if info.get("critical_low") and value < info["critical_low"]:
                flag = " 🚨"

        display_name = param_name.capitalize()
        norm = f"(норма {info['norm_min']}-{info['norm_max']})" if info.get("norm_max") else ""

        return f"{display_name}: {value} {info.get('unit', '')}{trend}{flag}  [{date}] {norm}"

    # Kidney
    kidney_lines = [_format_param(p) for p in kidney if p in latest]
    if kidney_lines:
        lines.append("Почки:")
        lines.extend(f"  {l}" for l in kidney_lines if l)

    # Blood
    blood_lines = [_format_param(p) for p in blood if p in latest]
    if blood_lines:
        lines.append("\nКровь:")
        lines.extend(f"  {l}" for l in blood_lines if l)

    # Liver
    liver_lines = [_format_param(p) for p in liver if p in latest]
    if liver_lines:
        lines.append("\nПечень:")
        lines.extend(f"  {l}" for l in liver_lines if l)

    # Everything else
    shown = set(kidney + blood + liver)
    other = [_format_param(p) for p in latest if p not in shown]
    if other:
        lines.append("\nПрочее:")
        lines.extend(f"  {l}" for l in other if l)

    return "\n".join(lines)


def format_param_detail(param: str, patient_id: str = None) -> str:
    """Format detailed history of one parameter."""
    normalized = _normalize_param(param)
    if not normalized:
        return f"Параметр '{param}' не найден. Доступные: {', '.join(sorted(LAB_PARAMS.keys()))}"

    history = get_param_history(normalized, patient_id=patient_id)
    if not history:
        return f"Нет данных по '{normalized}'."

    info = LAB_PARAMS[normalized]
    lines = [f"📈 {normalized.capitalize()} ({info['unit']}, норма {info['norm_min']}-{info['norm_max']}):\n"]

    for e in history:
        flag = ""
        if info.get("norm_max") and e["value"] > info["norm_max"]:
            flag = " ⚠️"
        elif info.get("norm_min") and e["value"] < info["norm_min"]:
            flag = " ⚠️"
        lines.append(f"  {e['date']}  —  {e['value']}{flag}")

    # Trend summary
    if len(history) >= 2:
        first = history[0]["value"]
        last = history[-1]["value"]
        diff = last - first
        pct = (diff / first) * 100 if first else 0
        direction = "рост" if diff > 0 else "снижение"
        lines.append(f"\nДинамика: {direction} на {abs(diff):.1f} ({abs(pct):.0f}%)")

    return "\n".join(lines)
