"""Register patients for the multi-patient medical assistant.

Run once to set up patient registry, then start the bot.

Usage:
    python register_patients.py
"""

from src.patient_manager import register_patient, get_all_patients


def main():
    # --- Patient 1: Olga (existing) ---
    register_patient(
        patient_id="olga",
        patient_telegram_id=1547855978,
        name="Суджювене Ольга Елмаровна",
        language="ru",
        healthcare_system="russia_moscow",
        family_members={
            5159906520: {"name": "Мантас", "role": "family"},
            2013402134: {"name": "Мария", "role": "family"},
        },
    )
    print("✓ Olga registered (ru, EMIAS Moscow)")

    # --- Patient 2: Mama (Lithuanian) ---
    # TODO: Fill in real telegram IDs
    # register_patient(
    #     patient_id="mama",
    #     patient_telegram_id=MAMA_TELEGRAM_ID,
    #     name="[Имя мамы]",
    #     language="lt",
    #     healthcare_system="lithuania",
    #     family_members={
    #         5159906520: {"name": "Mantas", "role": "family"},
    #     },
    # )
    # print("✓ Mama registered (lt, Lithuania)")

    # --- Patient 3: Colleague's wife ---
    # TODO: Fill in real telegram IDs
    # register_patient(
    #     patient_id="colleague-wife",
    #     patient_telegram_id=COLLEAGUE_WIFE_TELEGRAM_ID,
    #     name="[Имя]",
    #     language="ru",
    #     healthcare_system="russia_moscow",
    #     family_members={
    #         COLLEAGUE_TELEGRAM_ID: {"name": "[Имя коллеги]", "role": "family"},
    #     },
    # )
    # print("✓ Colleague's wife registered (ru, EMIAS Moscow)")

    # Show all registered
    patients = get_all_patients()
    print(f"\nRegistered patients: {len(patients)}")
    for pid, info in patients.items():
        print(f"  {pid}: {info['name']} ({info['language']}, {info['healthcare_system']})")


if __name__ == "__main__":
    main()
