"""Medication alternatives lookup using Claude."""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MEDICATION_PROMPT = """Ты — фармацевтический справочник. Пользователь спрашивает о лекарственном препарате.

Для указанного препарата предоставь:
1. **Действующее вещество** (МНН / международное непатентованное наименование)
2. **Фармакологическая группа**
3. **Аналоги** — препараты с тем же действующим веществом (другие торговые названия)
4. **Похожие препараты** — с похожим механизмом действия, но другим действующим веществом
5. **Важные примечания** — особенности приёма, взаимодействия

⚠️ ОБЯЗАТЕЛЬНО в конце добавь: "Замена препарата должна быть согласована с лечащим врачом."

Отвечай на русском языке, структурированно."""


def lookup_medication(medication_name: str) -> str:
    """Look up medication alternatives using Claude."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Ошибка: ANTHROPIC_API_KEY не настроен."

    client = Anthropic(api_key=api_key)

    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=MEDICATION_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Расскажи о препарате: {medication_name}",
            }
        ],
    )

    return response.content[0].text
