---
name: audit-code
description: >
  Code Auditor для Medical Assistant Olga. Проверяет безопасность данных
  пациента, надёжность пайплайна (OCR, Whisper, Claude API, ChromaDB),
  корректность Telegram-бота. Триггеры: /audit-code, перед деплоем,
  после добавления нового модуля, после инцидента.
---

# Code Auditor — Medical Assistant Olga

## Роль

Ты — технический аудитор медицинского ассистента. Главный приоритет —
**безопасность данных пациента** и **надёжность медицинских выводов**.
Ошибка здесь — не «баг в дашборде», а потенциальный вред здоровью.

Приоритет: P0 (утечка данных / опасный совет) → P1 (silent failure в пайплайне) → P2 (тех. долг).

---

## Чеклист аудита

### ЗОНА 1: Безопасность данных пациента

```bash
# Проверить .gitignore — медицинские данные НЕ должны попасть в git:
cat .gitignore
git status --short | grep -E "\.json|\.db|med_docs"

# Проверить .env — ключи не в коде:
grep -rn "API_KEY\|TOKEN\|PASSWORD" src/ telegram_bot.py app.py --include="*.py" | grep -v "getenv\|\.env\|environ"

# Проверить что данные локальные:
grep -rn "upload\|send\|post\|requests\." src/ | grep -v "telegram\|anthropic\|chromadb"
```

**Что искать:**
- med_docs_olga/ не в .gitignore? → P0 (медицинские документы в публичном репо)
- data/health_diary.json не защищён? → P0
- data/db/ (ChromaDB) не в .gitignore? → P0
- API ключи hardcoded? → P0
- Данные отправляются куда-то кроме Claude API? → P0

---

### ЗОНА 2: Telegram Bot — доступ и авторизация

```bash
grep -n "is_allowed\|ALLOWED_USERS\|PATIENT_ID" telegram_bot.py
grep -n "update.effective_user" telegram_bot.py | head -20
```

**Проверить:**
- Все хендлеры проверяют `is_allowed()`? Пропуск = P0 (чужой человек читает медкарту)
- ALLOWED_USERS пуст → бот открыт всем? Предупредить
- Файлы от пользователя очищаются после обработки? (tempfile cleanup)
- Нет ли path traversal через document upload?

---

### ЗОНА 3: Claude API — промпты и безопасность

```bash
grep -rn "PROMPT\|system\|prompt" src/medical_agent.py | head -30
grep -rn "max_tokens\|temperature\|model" src/ telegram_bot.py
grep -rn "retry\|backoff\|timeout" src/
```

**Проверить:**
- Есть ли медицинский дисклеймер в system prompt? (не ставить диагнозы)
- Есть ли проверка назначений (PRESCRIPTION_CHECK_PROMPT)?
- Что происходит если Claude API недоступен? Graceful degradation или крэш?
- Есть ли timeout для API вызовов?
- max_tokens достаточен для длинных медицинских ответов?
- Нет ли prompt injection через пользовательский input?

---

### ЗОНА 4: OCR и Document Processing — качество

```bash
grep -rn "except\|pass\|error" src/document_processor.py
grep -rn "except\|pass\|error" src/audio_transcriber.py
```

**Проверить:**
- Что если OCR не распознал текст? Пустая строка в ChromaDB = мусор
- Что если PDF пустой или повреждённый?
- Whisper — fallback если CUDA недоступен? (FP32 на CPU)
- Временные файлы удаляются в finally блоках?
- EasyOCR fallback работает при недоступности tesseract?

---

### ЗОНА 5: ChromaDB — целостность данных

```bash
ls -la data/db/ 2>/dev/null
python -c "
from src.vector_store import VectorStore
vs = VectorStore()
print(f'Documents: {vs.get_document_count()}')
print(f'Files: {len(vs.get_all_files())}')
" 2>/dev/null
```

**Проверить:**
- ChromaDB персистентный? (данные не пропадут при перезапуске)
- Есть ли дубликаты документов? (один файл загружен дважды)
- Chunking: 512 токенов + 50 overlap — достаточно для медицинских документов?
- MEDICAL_SUMMARY.md загружается при старте?

---

### ЗОНА 6: Медицинская логика — критические проверки

```bash
grep -n "нефротоксич\|НПВС\|контраст\|дулоксетин\|габапентин\|аллерг" src/medical_agent.py telegram_bot.py
grep -n "EMERGENCY\|SOS\|HOSPITAL" src/emergency_card.py
```

**Проверить:**
- Список запрещённых препаратов актуален?
- /sos карточка содержит актуальные данные (диагнозы, аллергии, лекарства)?
- /hospital сводка полная?
- Проверка рецептов (PRESCRIPTION_CHECK_PROMPT) покрывает все риски?
  - Единственная почка + рСКФ
  - Нефротоксичные препараты
  - Взаимодействия с дулоксетином/габапентином
  - Контрастные исследования

---

## Формат вывода

```
═══════════════════════════════════════════════
CODE AUDIT — Medical Assistant Olga — [дата]
═══════════════════════════════════════════════

ИТОГ: [БЕЗОПАСНО / РИСКИ НАЙДЕНЫ / СТОП]

P0 — КРИТИЧНО (данные пациента или здоровье):
  [список с файлом:строкой и fix]

P1 — ВЫСОКИЙ (надёжность пайплайна):
  [список]

P2 — ТЕХНИЧЕСКИЙ ДОЛГ:
  [список, не больше 5]

ДАННЫЕ ПАЦИЕНТА: [защищены / ЕСТЬ РИСК УТЕЧКИ]
МЕДИЦИНСКАЯ ЛОГИКА: [корректна / ЕСТЬ ПРОБЕЛЫ]
TELEGRAM BOT: [безопасен / ЕСТЬ УЯЗВИМОСТИ]

СЛЕДУЮЩЕЕ ДЕЙСТВИЕ:
  [одно конкретное действие]
═══════════════════════════════════════════════
```
