"""Migrate Olga's existing data to the new multi-patient structure.

Copies:
- med_docs_olga/*.md → data/patients/olga/documents/
- data/health_diary.json → data/patients/olga/health_diary.json
- data/lab_values.json → data/patients/olga/lab_values.json
- data/db/ (ChromaDB) → data/patients/olga/chromadb/
- Populates profile.json from MEDICAL_SUMMARY.md data

Usage:
    python -X utf8 migrate_olga.py
"""

import json
import shutil
from pathlib import Path

from src.patient_manager import load_profile, save_profile, get_patient_dir


def main():
    patient_id = "olga"
    patient_dir = get_patient_dir(patient_id)

    # 1. Copy documents
    src_docs = Path("med_docs_olga")
    dst_docs = patient_dir / "documents"
    dst_docs.mkdir(parents=True, exist_ok=True)

    if src_docs.exists():
        copied = 0
        for f in src_docs.iterdir():
            if f.is_file():
                dst = dst_docs / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    copied += 1
        print(f"Documents: copied {copied} files to {dst_docs}")
    else:
        print("No med_docs_olga/ directory found")

    # 2. Copy health diary
    src_diary = Path("data/health_diary.json")
    dst_diary = patient_dir / "health_diary.json"
    if src_diary.exists() and not dst_diary.exists():
        shutil.copy2(src_diary, dst_diary)
        print(f"Health diary: copied to {dst_diary}")
    else:
        print("Health diary: already exists or no source")

    # 3. Copy lab values
    src_labs = Path("data/lab_values.json")
    dst_labs = patient_dir / "lab_values.json"
    if src_labs.exists() and not dst_labs.exists():
        shutil.copy2(src_labs, dst_labs)
        print(f"Lab values: copied to {dst_labs}")
    else:
        print("Lab values: already exists or no source")

    # 4. Copy ChromaDB
    src_db = Path("data/db")
    dst_db = patient_dir / "chromadb"
    if src_db.exists() and not (dst_db / "chroma.sqlite3").exists():
        # ChromaDB stores data in files, copy the whole directory
        if dst_db.exists():
            shutil.rmtree(dst_db)
        shutil.copytree(src_db, dst_db)
        print(f"ChromaDB: copied to {dst_db}")
    else:
        print("ChromaDB: already exists or no source")

    # 5. Populate profile with known data from MEDICAL_SUMMARY
    profile = load_profile(patient_id)
    if profile and not profile.get("auto_extracted"):
        profile.update({
            "name": "Суджювене Ольга Елмаровна",
            "date_of_birth": "18.08.1971",
            "gender": "ж",
            "height_cm": 150,
            "weight_kg": 65,
            "insurance": "ОМС 6951820881000209",
            "address": "Москва, Винницкая ул., д. 5, кв. 43",
            "diagnoses": [
                {"name": "Рак шейки матки T3bN0M0", "icd_code": "C53.9", "year": "1993", "status": "ремиссия 30+ лет"},
                {"name": "Лейомиосаркома прямой кишки pT3N0M0", "icd_code": "C20", "year": "2023", "status": "ремиссия"},
                {"name": "ХБП С3б-4, тубулоинтерстициальный нефрит", "icd_code": "N18.3", "year": None, "status": "хронический"},
                {"name": "Полинейропатия нижних конечностей", "icd_code": None, "year": None, "status": "хронический"},
                {"name": "Тревожное расстройство, панические атаки", "icd_code": "G90.8", "year": None, "status": "хронический"},
                {"name": "ГЭРБ, гастрит с атрофией, H. pylori", "icd_code": "K21.0", "year": None, "status": "хронический"},
                {"name": "Остеопороз поясничного отдела", "icd_code": "M81", "year": None, "status": "хронический"},
            ],
            "allergies": [
                {"substance": "Кетопрофен", "reaction": "аллергия"},
                {"substance": "Карбамазепин", "reaction": "зуд, одышка, анафилактоидная реакция, скорая"},
                {"substance": "Венлафаксин", "reaction": "непереносимость"},
            ],
            "contraindicated": [
                {"substance": "Все НПВС (ибупрофен, диклофенак, нимесулид)", "reason": "единственная почка, рСКФ 32"},
                {"substance": "Аминогликозиды", "reason": "нефротоксичны"},
                {"substance": "Тетрациклины", "reason": "нефротоксичны"},
                {"substance": "Рентгенконтраст парентерально", "reason": "нефротоксичен, рСКФ 32"},
                {"substance": "Прегабалин", "reason": "непереносимость"},
                {"substance": "Трамадол", "reason": "непереносимость"},
            ],
            "surgeries": [
                {"name": "Лучевая + химиотерапия (84 Гр)", "date": "1993"},
                {"name": "Лапаротомия", "date": "2010"},
                {"name": "Пластика мочеточника по Боари", "date": "2016"},
                {"name": "Резекция прямой кишки + колостома (операция Гартмана)", "date": "16.03.2023"},
                {"name": "Нефрэктомия справа", "date": "14.01.2025"},
            ],
            "current_medications": [
                {"name": "Дулоксетин", "dose": "90 мг/сут", "frequency": "60 мг утро + 30 мг вечер", "critical": True},
                {"name": "Габапентин", "dose": "600 мг/сут", "frequency": "300 мг × 2 раза", "critical": True},
                {"name": "Холекальциферол", "dose": "2500 МЕ/сут", "frequency": "1 раз в день", "critical": False},
            ],
            "doctors": [
                {"specialty": "Терапевт", "name": "Родина А.А.", "clinic": "ГБУЗ ГП №209"},
                {"specialty": "Уролог", "name": "Карасев А.А.", "clinic": "ГБУЗ ГП №209"},
                {"specialty": "Невролог", "name": "Ситдикова В.А.", "clinic": "ГБУЗ ГП №209"},
                {"specialty": "Нефролог", "name": "Будаева З.Р.", "clinic": "ГБУЗ ММНКЦ им. С.П. Боткина ДЗМ"},
            ],
            "emergency_contacts": [
                {"name": "Мантас", "relation": "муж", "phone": ""},
                {"name": "Мария", "relation": "дочь", "phone": ""},
            ],
            "notes": "ЕДИНСТВЕННАЯ ЛЕВАЯ ПОЧКА. Креатинин растёт: 113→139→155 (янв-мар 2026). "
                     "Порог: 170+ срочно нефролог, 200+ стационар с гемодиализом. "
                     "Колостома (операция Гартмана). "
                     "Дулоксетин и габапентин ЖИЗНЕННО необходимы — без них панические атаки и скорая.",
            "auto_extracted": True,
        })
        save_profile(patient_id, profile)
        print("Profile: populated with full medical data")
    else:
        print("Profile: already populated")

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
