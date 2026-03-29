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
from src.emergency_card import EMERGENCY_CARD, HOSPITAL_CARD
from src.lab_tracker import extract_lab_values, save_lab_values, format_trends, format_param_detail
from src.audio_transcriber import (
    transcribe_audio, save_transcript, format_transcript_md,
    SUPPORTED_AUDIO,
)

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
    "Документы (PDF, фото, Excel) — добавлю в базу знаний.\n"
    "Аудиозапись визита к врачу — транскрибирую и проанализирую.\n\n"
    "Команды:\n"
    "/sos — экстренная карточка для скорой\n"
    "/hospital — сводка для приёмного отделения\n"
    "/doctor — подготовить речь для врача\n"
    "/labs — показатели (тренды)\n"
    "/diary — дневник самочувствия\n"
    "/files — список документов в базе\n"
    "/clear — очистить историю диалога\n\n"
    "Фото рецепта/назначения — автоматически проверю безопасность.\n\n"
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

ВАЖНО — ОШИБКИ В ЕМИАС:
Документы в ЕМИАС могут содержать неточности. Известные расхождения:
- Визит к нефрологу 29.03.2026: в ЕМИАС записано "мочеиспускание свободно,
  безболезненное" — на самом деле есть ложные позывы (ощущение цистита).
  Нефролог подтвердила проблему устно, но не записала.
- Инцидент с заменой дулоксетина/габапентина неврологом — не зафиксирован в ЕМИАС.
- Пороги эскалации креатинина (170+ срочно, 200+ стационар) — не записаны.
Если речь готовится для специалиста, которому важны эти факты — ВКЛЮЧИ
корректировку в речь пациентки ("доктор, в карте написано что мочеиспускание
в норме, но на самом деле...").

ФОРМАТ:
- Речь от первого лица, начинается с "Доктор, ..."
- Разговорный стиль, как человек говорит устно
- Без эмодзи, без заголовков, без списков, без форматирования
- Сплошной текст, 5-8 предложений
- Укладывается в 1-2 минуты устной речи (не больше 1000 символов)
- НИКАКИХ советов, поддержки, мотивации, "что спросить"
- НИКАКОЙ самодиагностики и интерпретации результатов — только факты и одна просьба"""


# Prescription/procedure verification prompt
VISIT_ANALYSIS_PROMPT = """Это транскрипт аудиозаписи медицинского визита пациентки.

ЗАДАЧА 1 — ИДЕНТИФИКАЦИЯ СПИКЕРОВ:
Определи по контексту речи кто говорит в каждом сегменте:
- ВРАЧ — использует медицинские термины, задаёт вопросы о симптомах, назначает, осматривает
- ПАЦИЕНТКА (Ольга) — описывает жалобы, отвечает на вопросы врача о самочувствии
- СОПРОВОЖДАЮЩИЙ — дополняет, уточняет, задаёт вопросы врачу от третьего лица

Перепиши ключевые моменты с указанием спикера.

ЗАДАЧА 2 — СОПОСТАВЛЕНИЕ С ДОКУМЕНТОМ ВИЗИТА:
Сравни что было СКАЗАНО на приёме с тем что НАПИСАНО в документе.
Найди:
- Расхождения (врач сказал одно, в документе другое)
- Что обсуждалось устно, но НЕ попало в документ
- Что в документе есть, но на приёме НЕ обсуждалось
- Нюансы и оговорки врача которые важны но не зафиксированы

ЗАДАЧА 3 — РИСКИ И ВАЖНОЕ:
- Устные рекомендации врача которые легко забыть
- Предупреждения и оговорки
- Что нужно сделать (анализы, записи к другим врачам, дневник)
- Противоречия между устными словами и письменным документом

ФОРМАТ: структурированный отчёт на русском, с цитатами из транскрипта.

ТРАНСКРИПТ:
{transcript}"""


PRESCRIPTION_CHECK_PROMPT = """Пациентка прислала фото рецепта или медицинского назначения.
Распознанный текст ниже.

ЗАДАЧА: Проверь КАЖДЫЙ препарат и процедуру из этого назначения на безопасность
для ЭТОЙ КОНКРЕТНОЙ пациентки. Используй МЕДИЦИНСКУЮ СВОДКУ.

Проверяй:
1. Есть ли препарат в списке запрещённых/аллергий?
2. Совместимость с текущими лекарствами (дулоксетин, габапентин, холекальциферол, канефрон)
3. Нагрузка на ЕДИНСТВЕННУЮ почку при рСКФ 32
4. Нагрузка на печень (АЛТ 48)
5. Взаимодействия с диагнозами (ХБП, ГЭРБ, остеопороз, тревожное расстройство)

ОСОБОЕ ВНИМАНИЕ:
- РЕНТГЕНКОНТРАСТ — ЗАПРЕЩЁН (единственная почка, рСКФ 32). Недавно делали КТ с контрастом → неделю не вставала с кровати.
- НПВС — ЗАПРЕЩЕНЫ все
- Любые нефротоксичные препараты

ФОРМАТ ОТВЕТА:
Для каждого найденного препарата/процедуры:
- Название: что это
- Вердикт: ✅ безопасно / ⚠️ осторожно / 🚫 ОПАСНО
- Почему (1 предложение)

Если нашёл что-то опасное — скажи ПРЯМО и ГРОМКО. Лучше перебдеть.

РАСПОЗНАННЫЙ ТЕКСТ:
{text}"""


# --- Auto-detect doctor visit ---

_DOCTOR_KEYWORDS = {
    "нефролог": "нефролог",
    "уролог": "уролог",
    "терапевт": "терапевт",
    "участков": "терапевт (участковый)",
    "невролог": "невролог",
    "онколог": "онколог",
    "гинеколог": "гинеколог",
    "эндокринолог": "эндокринолог",
    "кардиолог": "кардиолог",
    "гастроэнтеролог": "гастроэнтеролог",
    "хирург": "хирург",
    "окулист": "окулист",
    "офтальмолог": "офтальмолог",
    "дерматолог": "дерматолог",
    "лор": "ЛОР",
    "отоларинголог": "ЛОР",
    "ревматолог": "ревматолог",
    "пульмонолог": "пульмонолог",
    "психиатр": "психиатр",
    "психотерапевт": "психотерапевт",
}

# Stems for fuzzy match (handles typos like "учестковобу", "нефрологу", "тирапевту")
_DOCTOR_STEMS = {
    "нефр": "нефролог",
    "урол": "уролог",
    "терап": "терапевт",
    "участк": "терапевт (участковый)",
    "учест": "терапевт (участковый)",
    "невр": "невролог",
    "онко": "онколог",
    "гинек": "гинеколог",
    "эндокр": "эндокринолог",
    "кардио": "кардиолог",
    "гастр": "гастроэнтеролог",
    "хирур": "хирург",
    "окули": "окулист",
    "офтальм": "офтальмолог",
    "дерма": "дерматолог",
    "ревмат": "ревматолог",
    "пульм": "пульмонолог",
    "психиат": "психиатр",
    "психот": "психотерапевт",
}

_VISIT_MARKERS = [
    "иду к", "иду на", "еду к", "еду на", "записана к", "записалась к",
    "завтра к", "сегодня к", "пойду к", "поеду к", "была у", "хожу к",
    "приём у", "прием у", "визит к", "на приём", "на прием", "к врачу",
    "запись к", "направили к", "отправили к", "пойдём к", "пойдем к",
    "уду к", "буду у",
]


def _find_specialty(text: str) -> str | None:
    """Find doctor specialty in text, with fuzzy stem matching."""
    lower = text.lower()
    # Exact keyword match first
    for keyword, specialty in _DOCTOR_KEYWORDS.items():
        if keyword in lower:
            return specialty
    # Fuzzy stem match (handles typos and word forms)
    for stem, specialty in _DOCTOR_STEMS.items():
        if stem in lower:
            return specialty
    return None


def _detect_doctor_visit(text: str) -> str | None:
    """Detect if user mentions going to a doctor. Return specialty or None."""
    lower = text.lower().strip()

    # Check for visit markers
    has_marker = any(marker in lower for marker in _VISIT_MARKERS)

    # Check for time words
    time_words = ["завтра", "сегодня", "утром", "вечером", "послезавтра"]
    has_time = any(tw in lower for tw in time_words)

    # Check for generic visit words
    visit_words = ["врач", "доктор", "приём", "прием", "поликлиник", "больниц"]
    has_visit_word = any(vw in lower for vw in visit_words)

    if has_marker or has_time or has_visit_word:
        specialty = _find_specialty(lower)
        if specialty:
            return specialty
        if has_marker or has_visit_word:
            return "терапевт"

    return None


# --- Access control ---

def is_allowed(user_id: int) -> bool:
    """Check if user is in the allowed list."""
    if not ALLOWED_USERS:
        return True
    return user_id in ALLOWED_USERS


# --- Voice transcription ---

def transcribe_voice(ogg_path: str) -> str:
    """Convert voice message to text using local Whisper (no data sent to cloud)."""
    from src.audio_transcriber import transcribe_audio

    result = transcribe_audio(ogg_path)
    text = result["text"].strip()
    if not text:
        raise RuntimeError("Не удалось распознать речь. Попробуй сказать ещё раз.")
    return text


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


async def cmd_labs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /labs — show lab value trends. /labs креатинин for detail."""
    if not is_allowed(update.effective_user.id):
        return

    args = context.args
    if args:
        param = " ".join(args)
        text = format_param_detail(param)
    else:
        text = format_trends()

    if len(text) <= 4096:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), 4096):
            await update.message.reply_text(text[i : i + 4096])


async def cmd_sos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /sos — instant emergency card for paramedics."""
    if not is_allowed(update.effective_user.id):
        return
    await update.message.reply_text(EMERGENCY_CARD)


async def cmd_hospital(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /hospital — full summary for ER admission."""
    if not is_allowed(update.effective_user.id):
        return
    # Hospital card is longer, may need splitting
    text = HOSPITAL_CARD
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

    # Auto-detect doctor visit — prepare speech automatically
    doctor_specialty = _detect_doctor_visit(question)
    if doctor_specialty:
        await update.message.reply_text(f"Готовлю речь для {doctor_specialty}...")
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
        prompt = DOCTOR_VISIT_PROMPT.format(specialty=doctor_specialty)
        answer = agent.ask(prompt)
        if len(answer) <= 4096:
            await update.message.reply_text(answer)
        else:
            for i in range(0, len(answer), 4096):
                await update.message.reply_text(answer[i : i + 4096])
        return

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

    tmp_path = None
    try:
        voice = update.message.voice
        tg_file = await voice.get_file()

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await tg_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        text = transcribe_voice(tmp_path)

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
    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads — ingest into knowledge base."""
    if not is_allowed(update.effective_user.id):
        return

    doc = update.message.document
    file_name = doc.file_name or "document"
    ext = Path(file_name).suffix.lower()

    # Redirect audio files to the audio handler
    if ext in SUPPORTED_AUDIO:
        await handle_audio(update, context)
        return

    if ext not in SUPPORTED_EXTENSIONS:
        await update.message.reply_text(
            f"Формат {ext} не поддерживается.\n"
            f"Поддерживаемые: {', '.join(sorted(SUPPORTED_EXTENSIONS | SUPPORTED_AUDIO))}"
        )
        return

    await update.message.reply_text(f"Обрабатываю {file_name}...")

    tmp_path = None
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

        await update.message.reply_text(
            f"Готово! {file_name}: {chunks} фрагментов ({result['char_count']} символов) "
            f"добавлено в базу знаний."
        )
    except Exception as e:
        logger.error("Error processing document: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка при обработке {file_name}: {e}")
    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle audio file uploads — transcribe with Whisper and analyze visit."""
    if not is_allowed(update.effective_user.id):
        return

    # Audio can come as audio message or as document
    if update.message.audio:
        tg_file = await update.message.audio.get_file()
        file_name = update.message.audio.file_name or "audio.mp3"
        duration = update.message.audio.duration or 0
    elif update.message.document:
        ext = Path(update.message.document.file_name or "").suffix.lower()
        if ext not in SUPPORTED_AUDIO:
            return  # not an audio file, let handle_document deal with it
        tg_file = await update.message.document.get_file()
        file_name = update.message.document.file_name or "audio"
        duration = 0
    else:
        return

    duration_str = f" ({duration // 60}:{duration % 60:02d})" if duration else ""
    await update.message.reply_text(
        f"Транскрибирую {file_name}{duration_str}...\n"
        "Это может занять несколько минут для длинных записей."
    )

    tmp_path = None
    try:
        ext = Path(file_name).suffix.lower() or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            await tg_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        # Transcribe with Whisper
        result = transcribe_audio(tmp_path)
        duration_min = result["duration"] / 60

        # Save transcript to med_docs_olga/
        transcript_path = save_transcript(result, file_name)

        await update.message.reply_text(
            f"Транскрибация завершена: {duration_min:.1f} мин, "
            f"{len(result['segments'])} сегментов.\n"
            f"Сохранено: {transcript_path.name}\n\n"
            "Анализирую визит..."
        )

        # Add transcript to vector store
        md_text = format_transcript_md(result, file_name)
        vector_store.add_document(
            md_text,
            {"file_name": transcript_path.name, "file_type": "transcript"},
        )

        # Analyze visit with Claude — speaker ID + cross-reference with docs
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
        analysis_prompt = VISIT_ANALYSIS_PROMPT.format(transcript=md_text)
        answer = agent.ask(analysis_prompt)

        # Send analysis — may be long
        if len(answer) <= 4096:
            await update.message.reply_text(answer)
        else:
            for i in range(0, len(answer), 4096):
                await update.message.reply_text(answer[i : i + 4096])

    except Exception as e:
        logger.error("Error transcribing audio: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка при транскрибации: {e}")
    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads — OCR, ingest, and auto-check prescriptions."""
    if not is_allowed(update.effective_user.id):
        return

    await update.message.reply_text("Обрабатываю фото (OCR)...")

    tmp_path = None
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

        # Show a preview of extracted text
        preview = result["text"][:300]
        if len(result["text"]) > 300:
            preview += "..."

        await update.message.reply_text(
            f"Фото обработано: {chunks} фрагментов ({result['char_count']} символов).\n\n"
            f"Распознанный текст:\n{preview}"
        )

        # Auto-extract lab values and save to tracker
        lab_values = extract_lab_values(result["text"])
        if lab_values:
            saved = save_lab_values(lab_values)
            if saved:
                param_list = ", ".join(f"{v['param']} {v['value']}" for v in lab_values[:5])
                await update.message.reply_text(
                    f"📊 Найдено {len(lab_values)} показателей, сохранено {saved} новых: {param_list}"
                )

        # Auto-check for prescriptions/procedures in the recognized text
        text_lower = result["text"].lower()
        med_markers = [
            "рецепт", "назначен", "принимать", "таблет", "капсул", "мг ", "мл ",
            "раз в день", "р/д", "курс", "направлен", "исследован", "контраст",
            "рентген", "кт ", "мрт ", "узи ", "анализ",
        ]
        if any(marker in text_lower for marker in med_markers):
            await update.message.reply_text("🔍 Обнаружено назначение — проверяю безопасность...")
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action="typing"
            )
            check_prompt = PRESCRIPTION_CHECK_PROMPT.format(text=result["text"])
            answer = agent.ask(check_prompt)
            if len(answer) <= 4096:
                await update.message.reply_text(answer)
            else:
                for i in range(0, len(answer), 4096):
                    await update.message.reply_text(answer[i : i + 4096])

    except Exception as e:
        logger.error("Error processing photo: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка при обработке фото: {e}")
    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)


def main():
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        print("1. Create a bot via @BotFather in Telegram")
        print("2. Add TELEGRAM_BOT_TOKEN=your_token to .env")
        return

    if not ALLOWED_USERS:
        logger.warning("TELEGRAM_ALLOWED_USERS is empty — bot accepts ALL users (DEV MODE)")
    if not PATIENT_ID:
        logger.warning("TELEGRAM_PATIENT_ID not set — diary auto-recording disabled")

    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("files", cmd_files))
    app.add_handler(CommandHandler("labs", cmd_labs))
    app.add_handler(CommandHandler("sos", cmd_sos))
    app.add_handler(CommandHandler("hospital", cmd_hospital))
    app.add_handler(CommandHandler("doctor", cmd_doctor))
    app.add_handler(CommandHandler("diary", cmd_diary))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot started. Allowed users: %s", ALLOWED_USERS or "ALL (dev mode)")
    app.run_polling()


if __name__ == "__main__":
    main()
