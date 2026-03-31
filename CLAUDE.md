# Mano — Claude Code Guide

## Project Purpose
**Mano** — multi-patient medical AI assistant. Processes medical documents (PDFs,
photos, scans, Excel), enables Q&A over full health history, tracks medications
and lab values, verifies drug safety across all diagnoses, and prepares for
doctor visits. Telegram bot: @mano_med_bot

## Architecture: Multi-Patient
Each patient has isolated data under `data/patients/{patient_id}/`:
- `profile.json` — auto-built from documents (diagnoses, meds, allergies, etc.)
- `medications.json` — current meds, courses, discontinued
- `health_diary.json` — chronological symptom log
- `lab_values.json` — structured lab results with trends
- `conversations.db` — SQLite, full conversation history (persists across sessions)
- `chromadb/` — vector store for RAG
- `documents/` — uploaded document copies

Patient registry: `data/patient_registry.json` — maps Telegram user IDs to patients.

## Stack
- **Python 3.10+**
- **ChromaDB** — local vector database per patient
- **Anthropic Claude API** — claude-haiku-4-5-20251001 (default, configurable)
- **python-telegram-bot** — Telegram bot interface (primary)
- **Gradio** — web UI (legacy, single-patient)
- **Whisper** — voice message + audio transcription (local, CUDA)
- **pdfplumber** — PDF extraction
- **pytesseract + easyocr** — OCR (Russian, Lithuanian, English)
- **openpyxl / pandas** — Excel processing

## Project Structure
```
mano/
├── CLAUDE.md               ← you are here
├── .env                    ← API keys (never commit)
├── requirements.txt
├── register_patients.py    ← register new patients
├── migrate_olga.py         ← one-time migration of Olga's data
├── telegram_bot.py         ← Telegram bot (multi-patient)
├── app.py                  ← Gradio web app (legacy, single-patient)
├── ingest.py               ← CLI: add documents to knowledge base
├── src/
│   ├── patient_manager.py      ← multi-patient registry, profiles, conversation memory
│   ├── profile_extractor.py    ← auto-extract profile data from documents via Claude
│   ├── medication_tracker.py   ← medication courses, tracking, reminders
│   ├── medical_agent.py        ← Claude-powered Q&A agent with RAG
│   ├── document_processor.py   ← PDF, image OCR, Excel extraction
│   ├── vector_store.py         ← ChromaDB RAG wrapper
│   ├── health_diary.py         ← auto-detect health status, chronological log
│   ├── lab_tracker.py          ← lab value extraction, trends, critical alerts
│   ├── medication_lookup.py    ← drug alternatives lookup
│   ├── audio_transcriber.py    ← Whisper transcription
│   ├── emergency_card.py       ← legacy static cards (replaced by dynamic generation)
│   └── analytics.py            ← health dynamics charting
├── config/
│   └── healthcare_systems/
│       ├── russia_moscow.md    ← EMIAS context for doctor visit prep
│       └── lithuania.md        ← e.sveikata context
├── data/
│   ├── patient_registry.json   ← telegram_id → patient_id mapping
│   └── patients/
│       └── {patient_id}/       ← isolated patient data
│           ├── profile.json
│           ├── medications.json
│           ├── health_diary.json
│           ├── lab_values.json
│           ├── conversations.db
│           ├── chromadb/
│           └── documents/
└── med_docs_olga/          ← legacy (315+ documents, kept as backup)
```

## Key Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Register patients
python -X utf8 register_patients.py

# Migrate Olga's data (one-time)
python -X utf8 migrate_olga.py

# Run the Telegram bot
python -X utf8 telegram_bot.py

# Run the web app (legacy, single-patient)
python -X utf8 app.py
```

## Environment Variables
```
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
TELEGRAM_BOT_TOKEN=your_bot_token
```

## Telegram Bot Commands
- **/start** — registration (new users) or welcome (existing). Deep link: `/start inv_TOKEN`
- **/me** — create own medical profile (for users registered only as family)
- **/newpatient** — create profile for a family member
- **/invite [владелец]** — generate invite link (family or owner role)
- **/access** — show who has access to current patient
- **/revoke [id/name]** — owner removes someone's access
- **/switch [patient_id]** — switch active patient
- **/sos** — emergency card for paramedics (dynamic from profile)
- **/hospital** — full summary for ER admission
- **/doctor [specialty]** — speech for doctor visit (healthcare-system-aware)
- **/labs [param]** — lab value trends
- **/meds** — current medications
- **/meds add [name] [dose] [N weeks]** — add medication/course
- **/meds stop [name] result: [text]** — complete/stop medication
- **/diary [N]** — health diary entries
- **/profile** — what the bot knows about the patient
- **/files** — documents in knowledge base
- **/clear** — reset conversation history

## Auto Features
- **Document upload** → OCR → knowledge base + auto-profile extraction
- **Photo of prescription** → auto-check drug safety against all diagnoses
- **Health status detection** → auto-record in diary
- **Doctor visit detection** → auto-prepare speech
- **Voice messages** → Whisper transcription (ru/lt) → Q&A
- **Audio files** → full visit transcription + analysis
- **Lab values from photos** → auto-extract to tracker

## Access Control (Registry v3)
- **Registry v3** (`data/patient_registry.json`): role-based access + invite system
- Roles: `owner` (patient — admin of their data) | `family` (caregiver)
- **Self-registration**: new user sends /start → creates own profile as owner
- **Invite system**: /invite generates deep-link `t.me/bot?start=inv_TOKEN`
  - Owner invites family members
  - Family (if no owner yet) can invite the actual patient as owner
- **Owner becomes admin**: can /revoke anyone's access
- No whitelist (TELEGRAM_ALLOWED_USERS removed) — access via registry only
- `default_patient` sets which patient is active on bot start
- **/switch** lets users change active patient at runtime (in-memory)
- Each patient gets isolated VectorStore, MedicalAgent, diary, labs, meds
- Profile builds incrementally from uploaded documents
- System prompt includes: patient profile + healthcare system + medications + diary + conversation history

## Docker Deploy
```bash
docker compose build
docker compose up -d
docker logs -f mano_bot
```
- Dockerfile: Python 3.11-slim + ffmpeg + tesseract-ocr-rus
- Data persisted via volume: `./data:/app/data`
- Whisper runs on CPU (no GPU) — voice ETA shown to user

## Supported Languages
- **ru** — Russian (full support)
- **lt** — Lithuanian (interface, OCR, Whisper, prompts)

## Important Notes
- All data is LOCAL per patient — nothing shared between patients
- Claude API: text-only queries (documents processed locally)
- Medical disclaimer shown in UI
- Agent verifies EVERY new drug prescription against ALL diagnoses and current meds
- Windows: always use `python -X utf8` for correct encoding
