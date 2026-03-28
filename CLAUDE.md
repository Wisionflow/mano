# Medical Assistant — Claude Code Guide

## Project Purpose
Personal medical AI assistant for tracking a patient's long-term health history.
Processes 315+ documents (PDFs, photos, scans, Excel), enables Q&A over the full
history, tracks health dynamics over time, verifies drug safety, and prepares
for doctor visits.

## Stack
- **Python 3.10+**
- **ChromaDB** — local vector database (all data stays on-device)
- **Anthropic Claude API** — claude-haiku-4-5-20251001 (default, configurable)
- **Gradio** — web UI
- **python-telegram-bot** — Telegram bot interface
- **SpeechRecognition + ffmpeg** — voice message transcription
- **pdfplumber** — PDF text extraction
- **pytesseract + easyocr** — OCR for photos and scans (Russian + English)
- **openpyxl / pandas** — Excel/table processing

## Project Structure
```
medical-assistant-olga/
├── CLAUDE.md               ← you are here
├── .env                    ← API keys (never commit)
├── .env.example            ← template
├── requirements.txt
├── ingest.py               ← CLI: add documents to the knowledge base
├── app.py                  ← Main Gradio web app
├── telegram_bot.py         ← Telegram bot (text, voice, docs, diary, /doctor)
├── src/
│   ├── __init__.py
│   ├── document_processor.py   ← handles PDF, images, Excel, text
│   ├── vector_store.py         ← ChromaDB RAG wrapper
│   ├── medical_agent.py        ← Claude-powered Q&A and analysis agent
│   ├── health_diary.py         ← auto-detection of health status, chronological log
│   ├── medication_lookup.py    ← drug alternatives
│   └── analytics.py            ← health dynamics tracking and charting
├── med_docs_olga/          ← 315+ medical documents (1993-2026)
│   └── MEDICAL_SUMMARY.md  ← comprehensive patient summary (auto-loaded by agent)
└── data/
    ├── documents/          ← drop files here to ingest
    ├── db/                 ← ChromaDB persistent storage (auto-created)
    └── health_diary.json   ← diary entries (auto-created at runtime)
```

## Key Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# System dependencies
# ffmpeg: required for voice messages in Telegram bot
#   Ubuntu: sudo apt-get install ffmpeg
#   macOS: brew install ffmpeg
#   Windows: winget install ffmpeg
# tesseract: required for image/scan OCR
#   Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-rus
#   macOS: brew install tesseract tesseract-lang
#   Windows: download from https://github.com/UB-Mannheim/tesseract/wiki

# Ingest documents
python ingest.py --dir data/documents

# Run the web app (Gradio)
python app.py

# Run the Telegram bot
python telegram_bot.py
```

## Environment Variables
```
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_PATIENT_ID=patient_telegram_id
TELEGRAM_FAMILY=id1:Name1,id2:Name2
TELEGRAM_ALLOWED_USERS=id1,id2,id3
```

## Telegram Bot Features
- **Text Q&A** — ask anything about medical history, meds, lab values
- **Voice messages** — speak instead of type (Google Speech API, Russian)
- **Document upload** — PDF, Excel, photos → OCR → knowledge base
- **Health diary** — auto-detects when patient describes symptoms, logs with timestamp
- **/doctor [specialty]** — generates first-person speech for doctor visit (EMIAS-aware)
- **/diary [N]** — show last N diary entries
- **/files** — list documents in knowledge base
- **/clear** — reset conversation history
- **Roles:** Patient (auto-diary), Family (diary with attribution tag)

## Development Priorities (in order)
1. ✅ Project structure
2. ✅ Document processor (PDF, image OCR, Excel)
3. ✅ Vector store with ChromaDB
4. ✅ Medical agent with Claude
5. ✅ Basic Gradio UI
6. ✅ Telegram bot interface
7. ✅ Voice messages (speech-to-text)
8. ✅ Health diary (auto-detection + chronological log)
9. ✅ Doctor visit preparation (/doctor command)
10. ✅ Drug safety verification (cross-specialty checks)
11. ✅ Family roles (patient vs family members)
12. ✅ Medication alternatives lookup
13. ✅ Health dynamics charts
14. ⬜ Timeline view of medical history
15. ⬜ Automatic document metadata extraction (date, doctor, type)

## Important Notes
- All data is LOCAL — nothing sent to cloud except Claude API queries (text only)
- Language: interface in Russian, OCR supports Russian
- Medical disclaimer shown in UI
- Agent verifies EVERY new drug prescription against all diagnoses and current meds
- Documents chunked at 512 tokens with 50-token overlap for best retrieval
- MEDICAL_SUMMARY.md loaded at startup — agent always has full patient context

## Agent Behavior
The medical agent:
- Answers questions about specific test results
- Compares values across dates (dynamics)
- Explains medical terms in simple language
- Verifies drug safety across ALL diagnoses (not just one specialty)
- Checks interactions with current medications
- Warns about kidney load (single kidney, eGFR 32)
- Prepares EMIAS-aware speech for doctor visits
- Logs health status from diary entries
- Never diagnoses — only informs, structures, and warns about risks
