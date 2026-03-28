# Medical Assistant — Claude Code Guide

## Project Purpose
Personal medical AI assistant for tracking a patient's long-term health history.
Processes 200+ documents (PDFs, photos, scans, Excel), enables Q&A over the full
history, tracks health dynamics over time, and suggests medication alternatives.

## Stack
- **Python 3.10+**
- **ChromaDB** — local vector database (all data stays on-device)
- **LlamaIndex** — RAG pipeline for document ingestion & retrieval
- **Anthropic Claude API** — claude-sonnet-4-20250514 as the agent brain
- **Gradio** — web UI
- **pdfplumber** — PDF text extraction
- **pytesseract + easyocr** — OCR for photos and scans (Russian + English)
- **openpyxl / pandas** — Excel/table processing

## Project Structure
```
medical-assistant/
├── CLAUDE.md               ← you are here
├── .env                    ← API keys (never commit)
├── .env.example            ← template
├── requirements.txt
├── ingest.py               ← CLI: add documents to the knowledge base
├── app.py                  ← Main Gradio web app
├── src/
│   ├── __init__.py
│   ├── document_processor.py   ← handles PDF, images, Excel, text
│   ├── vector_store.py         ← ChromaDB RAG wrapper
│   ├── medical_agent.py        ← Claude-powered Q&A and analysis agent
│   ├── medication_lookup.py    ← drug alternatives + web search
│   └── analytics.py            ← health dynamics tracking and charting
└── data/
    ├── documents/          ← drop files here to ingest
    └── db/                 ← ChromaDB persistent storage (auto-created)
```

## Key Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install tesseract OCR (required for image/scan processing)
# Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-rus
# macOS: brew install tesseract tesseract-lang
# Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki

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
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_USERS=123456789   # comma-separated Telegram user IDs
```

## Development Priorities (in order)
1. ✅ Project structure
2. ✅ Document processor (PDF, image OCR, Excel)
3. ✅ Vector store with ChromaDB
4. ✅ Medical agent with Claude
5. ✅ Basic Gradio UI
6. ✅ Telegram bot interface
7. ⬜ Medication alternatives lookup (web search via Claude tool)
7. ⬜ Health dynamics charts (plot lab values over time)
8. ⬜ Timeline view of medical history
9. ⬜ Automatic document metadata extraction (date, doctor, type)

## Important Notes
- All data is LOCAL — nothing sent to cloud except Claude API queries (text only, no images sent externally unless user explicitly uses image analysis)
- Language: interface in Russian, OCR supports Russian
- Medical disclaimer must be shown in UI
- When suggesting medication alternatives, always note "consult your doctor"
- Documents are chunked at 512 tokens with 50-token overlap for best retrieval

## Agent Behavior
The medical agent should:
- Answer questions about specific test results
- Compare values across dates (dynamics)
- Explain medical terms in simple language
- Suggest questions to ask the doctor
- Find medication alternatives (same active substance, different brand/manufacturer)
- Never diagnose — only inform and structure information
