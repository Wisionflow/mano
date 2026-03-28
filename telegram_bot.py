"""Telegram bot interface for the medical assistant."""

import os
import tempfile
import logging
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from src.document_processor import process_file, SUPPORTED_EXTENSIONS
from src.vector_store import VectorStore
from src.medical_agent import MedicalAgent
from src.health_diary import is_health_status, save_entry, load_entries, format_entries

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Config ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Patient — diary auto-records from this user
PATIENT_ID = int(os.getenv("TELEGRAM_PATIENT_ID", "0") or "0")

# Family members — full access, diary only with explicit tag
FAMILY_MEMBERS = {}
for pair in os.getenv("TELEGRAM_FAMILY", "").split(","):
    pair = pair.strip()
    if ":" in pair:
        uid, name = pair.split(":", 1)
        FAMILY_MEMBERS[int(uid.strip())] = name.strip()

# All allowed user IDs
ALLOWED_USERS = {
    int(uid.strip())
    for uid in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
    if uid.strip()
}

# --- Shared state ---
vector_store = VectorStore()
agent = MedicalAgent(vector_store=vector_store)

WELCOME_TEXT = (
    "Привет! Я твой медицинский ассистент.\n\n"
    "Я могу:\n"
    "- Отвечать на вопросы по твоим анализам и документам\n"
    "- Объяснять медицинские термины простым языком\n"
    "- Сравнивать показатели в динамике\n"
    "- Искать аналоги лекарств\n"
    "- Проверять назначения врачей на совместимость\n\n"
    "Можно писать текстом или отправить голосовое сообщение.\n"
    "Документы (PDF, фото, Excel) — добавлю в базу знаний.\n\n"
    "Команды:\n"
    "/doctor — подготовить речь для врача\n"
    "/diary — дневник самочувствия\n"
    "/files — список документов в базе\n"
    "/clear — очистить историю диалога\n\n"
    "⚠️ Я не ставлю диагнозы и не заменяю врача."
)


# Doctor visit: speech for the doctor, first person
DOCTOR_VISIT_PROMPT = """Пациентка идёт на приём к врачу: {specialty}.

Составь РЕЧЬ ОТ ПЕРВОГО ЛИЦА — то что пациентка скажет врачу когда он спросит
"На что жалуетесь?" или подобный вопрос. Это единственное окно для передачи информации.

КОНТЕКСТ:
- Врач в московской поликлинике работает в ЕМИАС. Он уже ВИДИТ в системе:
  основные диагнозы, историю обращений в госучреждения, лабораторные результаты
  из государственных лабораторий, рецепты выписанные через ЕМИАС
- Приём длится 12-15 минут, из них 5-8 на общение
- НЕ надо повторять то что врач видит в ЕМИАС (диагнозы, операции, базовый анамнез)

ЧТО ВКЛЮЧИТЬ В РЕЧЬ (только то что врач НЕ увидит в системе или может пропустить):
- Свежая динамика ключевых показателей для ЭТОЙ специальности (цифры + даты)
- Лекарства назначенные ДРУГИМИ специалистами — включать ТОЛЬКО те, которые ВЛИЯЮТ
  на решения этого конкретного врача (например нефрологу — недавние антибиотики,
  потому что нагрузка на почку; печёночные показатели для выбора лекарств)
- Недавние события (инфекции, обострения, смена лекарств, побочки, вызовы скорой)
- Аллергии и непереносимости, особенно НОВЫЕ
- Конкретные жалобы если есть

ОБЯЗАТЕЛЬНО ВКЛЮЧИТЬ (для ЛЮБОГО врача, любой специальности):
Пациентка критически зависит от дулоксетина и габапентина (5+ лет). Без них —
тяжёлые панические атаки и боли. Уже пробовали заменить — скорая. В речи это
должно звучать как ПРОСЬБА ПАЦИЕНТКИ: "мне очень важно их сохранить, без них мне
очень плохо" + факт про аллергию на замену. Любой врач может назначить несовместимое
или предложить отмену — пациентка должна это предупредить.

ФОРМАТ:
- Речь от первого лица, начинается с "Доктор, ..."
- Разговорный стиль, как человек говорит устно
- Без эмодзи, без заголовков, без списков, без форматирования
- Сплошной текст, 5-8 предложений
- Укладывается в 1-2 минуты устной речи (не больше 1000 символов)
- НИКАКИХ советов, поддержки, мотивации, "что спросить"
- НИКАКОЙ самодиагностики и интерпретации результатов — только факты и одна просьба"""


# --- Access control ---

def is_allowed(user_id: int) -> bool:
    """Check if user is in the allowed list."""
    if not ALLOWED_USERS:
        return True
    return user_id in ALLOWED_USERS


# --- Voice transcription ---

def transcribe_voice(ogg_path: str) -> str:
    """Convert voice message to text using speech_recognition + Google API."""
    import speech_recognition as sr

    # Convert OGG/OGA to WAV using ffmpeg
    wav_path = ogg_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", ogg_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg не найден. Установите ffmpeg для работы с голосовыми сообщениями."
        )

    if not Path(wav_path).exists():
        raise RuntimeError("Не удалось конвертировать аудио.")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    os.unlink(wav_path)

    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text
    except sr.UnknownValueError:
        raise RuntimeError("Не удалось распознать речь. Попробуй сказать ещё раз.")
    except sr.RequestError as e:
        raise RuntimeError(f"Ошибка сервиса распознавания: {e}")


async def _ask_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str):
    """Common logic: send question to agent, reply with answer."""
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        answer = agent.ask(question)
        if len(answer) <= 4096:
            await update.message.reply_text(answer)
        else:
            for i in range(0, len(answer), 4096):
                await update.message.reply_text(answer[i : i + 4096])
    except Exception as e:
        logger.error("Error answering question: %s", e, exc_info=True)
        await update.message.reply_text(
            "Произошла ошибка при обработке вопроса. Попробуй позже."
        )


# --- Handlers ---

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("Доступ ограничен.")
        return
    await update.message.reply_text(WELCOME_TEXT)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear — reset conversation history."""
    if not is_allowed(update.effective_user.id):
        return
    agent.clear_history()
    await update.message.reply_text("История диалога очищена.")


async def cmd_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /files — list documents in knowledge base."""
    if not is_allowed(update.effective_user.id):
        return
    summary = agent.get_summary()
    await update.message.reply_text(summary)


async def cmd_diary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /diary — show health diary entries."""
    if not is_allowed(update.effective_user.id):
        return

    args = context.args
    n = 10  # default
    if args and args[0].isdigit():
        n = int(args[0])

    entries = load_entries(last_n=n)
    text = format_entries(entries)

    if len(text) <= 4096:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), 4096):
            await update.message.reply_text(text[i : i + 4096])


async def cmd_doctor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /doctor — prepare speech for a doctor visit."""
    if not is_allowed(update.effective_user.id):
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Укажи специальность после команды.\n\n"
            "Примеры:\n"
            "/doctor нефролог\n"
            "/doctor уролог\n"
            "/doctor терапевт\n"
            "/doctor невролог\n"
            "/doctor онколог"
        )
        return

    specialty = " ".join(args)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        prompt = DOCTOR_VISIT_PROMPT.format(specialty=specialty)
        answer = agent.ask(prompt)
        if len(answer) <= 4096:
            await update.message.reply_text(answer)
        else:
            for i in range(0, len(answer), 4096):
                await update.message.reply_text(answer[i : i + 4096])
    except Exception as e:
        logger.error("Error preparing doctor visit: %s", e, exc_info=True)
        await update.message.reply_text("Ошибка при подготовке. Попробуй позже.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages — Q&A."""
    if not is_allowed(update.effective_user.id):
        return

    question = update.message.text.strip()
    if not question:
        return

    user_id = update.effective_user.id

    # Auto-detect health status — diary only from patient
    if is_health_status(question):
        if user_id == PATIENT_ID:
            entry = save_entry(question)
            await update.message.reply_text(f"📝 Записано в дневник ({entry['timestamp']})")
        elif user_id in FAMILY_MEMBERS:
            # Family member reporting about patient — save with tag
            name = FAMILY_MEMBERS[user_id]
            entry = save_entry(question, reported_by=name)
            await update.message.reply_text(
                f"📝 Записано в дневник от {name} ({entry['timestamp']})"
            )

    await _ask_and_reply(update, context, question)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages — transcribe and answer."""
    if not is_allowed(update.effective_user.id):
        return

    await update.message.reply_text("Слушаю...")

    try:
        voice = update.message.voice
        tg_file = await voice.get_file()

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await tg_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        text = transcribe_voice(tmp_path)
        os.unlink(tmp_path)

        # Show what was recognized
        await update.message.reply_text(f"Распознано: {text}")

        # Auto-detect health status and save to diary
        user_id = update.effective_user.id
        if is_health_status(text):
            if user_id == PATIENT_ID:
                entry = save_entry(text)
                await update.message.reply_text(f"📝 Записано в дневник ({entry['timestamp']})")
            elif user_id in FAMILY_MEMBERS:
                name = FAMILY_MEMBERS[user_id]
                entry = save_entry(text, reported_by=name)
                await update.message.reply_text(
                    f"📝 Записано в дневник от {name} ({entry['timestamp']})"
                )

        # Answer the question
        await _ask_and_reply(update, context, text)

    except RuntimeError as e:
        await update.message.reply_text(str(e))
    except Exception as e:
        logger.error("Error processing voice: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка при обработке голосового: {e}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads — ingest into knowledge base."""
    if not is_allowed(update.effective_user.id):
        return

    doc = update.message.document
    file_name = doc.file_name or "document"
    ext = Path(file_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        await update.message.reply_text(
            f"Формат {ext} не поддерживается.\n"
            f"Поддерживаемые: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return

    await update.message.reply_text(f"Обрабатываю {file_name}...")

    try:
        tg_file = await doc.get_file()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            await tg_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        result = process_file(tmp_path)
        chunks = vector_store.add_document(
            result["text"],
            {"file_name": file_name, "file_type": result["file_type"]},
        )
        os.unlink(tmp_path)

        await update.message.reply_text(
            f"Готово! {file_name}: {chunks} фрагментов ({result['char_count']} символов) "
            f"добавлено в базу знаний."
        )
    except Exception as e:
        logger.error("Error processing document: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка при обработке {file_name}: {e}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads — OCR and ingest."""
    if not is_allowed(update.effective_user.id):
        return

    await update.message.reply_text("Обрабатываю фото (OCR)...")

    try:
        photo = update.message.photo[-1]  # largest resolution
        tg_file = await photo.get_file()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            await tg_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        result = process_file(tmp_path)
        file_name = f"photo_{photo.file_unique_id}.jpg"
        chunks = vector_store.add_document(
            result["text"],
            {"file_name": file_name, "file_type": "image"},
        )
        os.unlink(tmp_path)

        # Show a preview of extracted text
        preview = result["text"][:300]
        if len(result["text"]) > 300:
            preview += "..."

        await update.message.reply_text(
            f"Фото обработано: {chunks} фрагментов ({result['char_count']} символов).\n\n"
            f"Распознанный текст:\n{preview}"
        )
    except Exception as e:
        logger.error("Error processing photo: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка при обработке фото: {e}")


def main():
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        print("1. Create a bot via @BotFather in Telegram")
        print("2. Add TELEGRAM_BOT_TOKEN=your_token to .env")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("files", cmd_files))
    app.add_handler(CommandHandler("doctor", cmd_doctor))
    app.add_handler(CommandHandler("diary", cmd_diary))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot started. Allowed users: %s", ALLOWED_USERS or "ALL (dev mode)")
    app.run_polling()


if __name__ == "__main__":
    main()
