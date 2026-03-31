"""Telegram bot interface for Mano — personal medical assistant."""

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
from src.lab_tracker import extract_lab_values, save_lab_values, format_trends, format_param_detail
from src.audio_transcriber import (
    transcribe_audio, save_transcript, format_transcript_md,
    SUPPORTED_AUDIO,
)
from src.patient_manager import (
    get_patient_id_by_telegram,
    set_active_patient,
    get_accessible_patients,
    get_user_role,
    get_user_name,
    get_patient_name,
    get_patient_dir,
    load_profile,
    build_emergency_card,
    add_patient_access,
    is_registered,
    create_self_patient,
    create_patient_for_other,
    create_invite,
    accept_invite,
    revoke_access,
    get_patient_access_list,
    is_owner,
    patient_has_owner,
    update_user_name,
)
from src.profile_extractor import process_document_for_profile
from src.medication_tracker import (
    format_medications, parse_meds_command,
    add_medication, stop_medication,
)

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Config ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BOT_NAME = os.getenv("TELEGRAM_BOT_NAME", "mano_med_bot")

# --- Registration state (in-memory, pending registrations) ---
# telegram_user_id → {"state": "awaiting_name"|"awaiting_newpatient_name", ...}
_pending_action: dict[int, dict] = {}

# --- Multi-patient state ---
# Each patient gets their own VectorStore + MedicalAgent, cached by patient_id
_agents: dict[str, MedicalAgent] = {}
_vector_stores: dict[str, VectorStore] = {}


def _get_vector_store(patient_id: str) -> VectorStore:
    """Get or create a patient-specific VectorStore."""
    if patient_id not in _vector_stores:
        db_path = str(get_patient_dir(patient_id) / "chromadb")
        _vector_stores[patient_id] = VectorStore(db_path=db_path)
    return _vector_stores[patient_id]


def _get_agent(patient_id: str) -> MedicalAgent:
    """Get or create a patient-specific MedicalAgent."""
    if patient_id not in _agents:
        vs = _get_vector_store(patient_id)
        _agents[patient_id] = MedicalAgent(vector_store=vs, patient_id=patient_id)
    return _agents[patient_id]


def _resolve_patient(telegram_user_id: int) -> str | None:
    """Resolve telegram user ID to patient_id. Returns None if not registered."""
    return get_patient_id_by_telegram(telegram_user_id)

NOT_REGISTERED_TEXT = (
    "Вы ещё не зарегистрированы. Отправьте /start чтобы начать.\n"
    "You are not registered yet. Send /start to begin."
)


def _build_welcome(telegram_user_id: int, patient_id: str, lang: str) -> str:
    """Build personalized welcome message based on user role and access."""
    accessible = get_accessible_patients(telegram_user_id)
    role = get_user_role(telegram_user_id, patient_id)
    patient_name = get_patient_name(patient_id) or "пациент"
    user_name = get_user_name(telegram_user_id) or ""
    has_multiple = len(accessible) > 1
    is_patient = role == "owner"
    logger.info("_build_welcome: tg_id=%s user_name=%s role=%s patient_id=%s", telegram_user_id, user_name, role, patient_id)

    if lang == "lt":
        return _build_welcome_lt(user_name, patient_name, is_patient, has_multiple)

    # --- Russian ---
    lines = []

    if is_patient:
        greeting = f"Привет, {user_name}!" if user_name else "Привет!"
        lines.append(f"{greeting} Я Mano — ваш личный медицинский помощник.")
        lines.append("")
        lines.append("Что я умею:")
        lines.append("• Храню все ваши медицинские документы в одном месте")
        lines.append("• Отвечаю на вопросы по анализам и диагнозам простым языком")
        lines.append("• Слежу за динамикой показателей — увижу если что-то ухудшается")
        lines.append("• Проверяю каждый новый рецепт на совместимость с вашими лекарствами и диагнозами")
        lines.append("• Готовлю речь перед визитом к врачу — чтобы ничего не забыть")
        lines.append("• В экстренной ситуации покажу карточку для скорой со всей важной информацией")
    else:
        greeting = f"Привет, {user_name}!" if user_name else "Привет!"
        lines.append(f"{greeting} Я Mano — медицинский помощник для вашей семьи.")
        lines.append("")
        lines.append(f"Сейчас активный профиль: {patient_name}")
        lines.append("")
        lines.append("Вы можете помогать близкому человеку с медициной:")
        lines.append("• Загружайте документы (фото, PDF, выписки) — я сохраню в базу знаний")
        lines.append("• Спрашивайте что угодно по анализам и назначениям")
        lines.append("• Сфотографируйте рецепт — проверю безопасность всех препаратов")
        lines.append("• Перед визитом к врачу подготовлю речь с учётом всей истории")
        lines.append("• В экстренной ситуации — /sos покажет карточку для скорой")

    lines.append("")
    lines.append("Как пользоваться:")
    lines.append("• Просто напишите вопрос текстом или голосовым сообщением")
    lines.append("• Отправьте документ или фото — я обработаю и добавлю в базу")
    lines.append("• Запишите визит к врачу на диктофон — транскрибирую и проанализирую")
    lines.append("")
    lines.append("Команды:")
    lines.append("/sos — карточка для скорой помощи")
    lines.append("/doctor невролог — подготовить речь для врача")
    lines.append("/labs — анализы и тренды")
    lines.append("/meds — текущие лекарства")
    lines.append("/profile — что я знаю о пациенте")
    lines.append("/diary — дневник самочувствия")
    lines.append("/switch — переключить профиль пациента")
    lines.append("/me — создать свой медицинский профиль")
    lines.append("/newpatient — добавить профиль близкого")
    lines.append("/invite — пригласить человека к профилю")
    lines.append("/access — кто имеет доступ")

    lines.append("")
    lines.append("Я не ставлю диагнозы и не заменяю врача.")

    return "\n".join(lines)


def _build_welcome_lt(user_name: str, patient_name: str, is_patient: bool, has_multiple: bool) -> str:
    """Lithuanian welcome message."""
    lines = []
    greeting = f"Sveiki, {user_name}!" if user_name else "Sveiki!"

    if is_patient:
        lines.append(f"{greeting} Aš Mano — jūsų asmeninis medicininis asistentas.")
        lines.append("")
        lines.append("Ką galiu:")
        lines.append("• Saugau visus jūsų medicininius dokumentus vienoje vietoje")
        lines.append("• Atsakau į klausimus apie tyrimus ir diagnozes paprastai")
        lines.append("• Seku rodiklių dinamiką — pastebėsiu jei kas blogėja")
        lines.append("• Tikrinu kiekvieną naują receptą dėl suderinamumo")
        lines.append("• Paruošiu kalbą prieš vizitą pas gydytoją")
        lines.append("• Skubiu atveju — kortelė greitajai su visa svarbia informacija")
    else:
        lines.append(f"{greeting} Aš Mano — medicininis asistentas jūsų šeimai.")
        lines.append("")
        lines.append(f"Aktyvus profilis: {patient_name}")
        lines.append("")
        lines.append("Galite padėti artimam žmogui su medicina:")
        lines.append("• Siųskite dokumentus — išsaugosiu žinių bazėje")
        lines.append("• Klauskite bet ką apie tyrimus ir paskyrimus")
        lines.append("• Nufotografuokite receptą — patikrinsiu saugumą")
        lines.append("• Prieš vizitą pas gydytoją paruošiu kalbą")
        lines.append("• Skubiu atveju — /sos parodys kortelę greitajai")

    lines.append("")
    lines.append("Kaip naudotis:")
    lines.append("• Rašykite klausimą tekstu arba balso žinute")
    lines.append("• Siųskite dokumentą ar nuotrauką — apdorosiu ir pridėsiu")
    lines.append("• Įrašykite vizitą pas gydytoją — transkribuosiu ir analizuosiu")
    lines.append("")
    lines.append("Komandos:")
    lines.append("/sos — kortelė greitajai pagalbai")
    lines.append("/doctor neurologas — paruošti kalbą gydytojui")
    lines.append("/labs — tyrimai ir tendencijos")
    lines.append("/meds — dabartiniai vaistai")
    lines.append("/profile — ką žinau apie pacientą")
    lines.append("/diary — savijautos dienoraštis")

    lines.append("/switch — perjungti paciento profilį")
    lines.append("/newpatient — pridėti artimojo profilį")
    lines.append("/invite — pakviesti žmogų prie profilio")
    lines.append("/access — kas turi prieigą")

    lines.append("")
    lines.append("Nediagnozuoju ir nepakeičiu gydytojo.")

    return "\n".join(lines)


def _get_lang(patient_id: str) -> str:
    """Get patient language."""
    profile = load_profile(patient_id)
    return profile.get("language", "ru") if profile else "ru"


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

def _check_access(telegram_user_id: int) -> str | None:
    """Check if user is registered. Returns patient_id or None."""
    return _resolve_patient(telegram_user_id)


# --- Voice transcription ---

def transcribe_voice(ogg_path: str, language: str = "ru") -> str:
    """Convert voice message to text using local Whisper (no data sent to cloud)."""
    from src.audio_transcriber import transcribe_audio

    result = transcribe_audio(ogg_path, language=language)
    text = result["text"].strip()
    if not text:
        raise RuntimeError("Не удалось распознать речь. Попробуй сказать ещё раз.")
    return text


async def _ask_and_reply(
    update: Update, context: ContextTypes.DEFAULT_TYPE, question: str, patient_id: str
):
    """Common logic: send question to agent, reply with answer."""
    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
    except Exception:
        pass  # non-critical, continue even if typing indicator fails

    try:
        agent = _get_agent(patient_id)
        user_id = update.effective_user.id
        sender = get_user_name(user_id) or update.effective_user.first_name
        answer = agent.ask(question, sender_telegram_id=user_id, sender_name=sender)
        if len(answer) <= 4096:
            await update.message.reply_text(answer)
        else:
            for i in range(0, len(answer), 4096):
                await update.message.reply_text(answer[i : i + 4096])
    except Exception as e:
        logger.error("Error answering question: %s", e, exc_info=True)
        try:
            await update.message.reply_text(
                "Произошла ошибка при обработке вопроса. Попробуй позже."
            )
        except Exception:
            pass


# --- Handlers ---

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start — registration or welcome. Deep link: /start inv_TOKEN."""
    user_id = update.effective_user.id
    args = context.args

    # --- Deep link: invite acceptance ---
    if args and args[0].startswith("inv_"):
        token = args[0][4:]
        tg_name = update.effective_user.first_name or ""
        result = accept_invite(user_id, token, user_name=tg_name)
        if result:
            role_label = "владелец" if result["role"] == "owner" else "родственник"
            await update.message.reply_text(
                f"Вы подключены к профилю: {result['patient_name']} ({role_label}).\n\n"
                "Используйте /switch для переключения между профилями."
            )
            # If this is their first patient, also show welcome
            patient_id = get_patient_id_by_telegram(user_id)
            if patient_id:
                lang = _get_lang(patient_id)
                welcome = _build_welcome(user_id, patient_id, lang)
                await update.message.reply_text(welcome)
            return
        else:
            await update.message.reply_text(
                "Ссылка-приглашение недействительна или уже использована."
            )
            return

    # --- Already registered — show welcome ---
    if is_registered(user_id):
        patient_id = get_patient_id_by_telegram(user_id)
        if patient_id:
            # Always sync name from Telegram
            tg_name = update.effective_user.first_name or ""
            if tg_name:
                update_user_name(user_id, tg_name)
            lang = _get_lang(patient_id)
            logger.info("cmd_start: user_id=%s tg_name=%s patient_id=%s", user_id, tg_name, patient_id)
            welcome = _build_welcome(user_id, patient_id, lang)
            await update.message.reply_text(welcome)
            return

    # --- New user — start registration ---
    _pending_action[user_id] = {"state": "awaiting_name"}
    await update.message.reply_text(
        "Добро пожаловать в Mano — ваш личный медицинский помощник!\n\n"
        "Как вас зовут? (Имя или Имя Фамилия)"
    )


async def cmd_me(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /me — create own patient profile (for users who are only family)."""
    user_id = update.effective_user.id

    # Check if user already has own profile (is owner of any patient)
    accessible = get_accessible_patients(user_id)
    for pid, role in accessible.items():
        if role == "owner":
            await update.message.reply_text(
                f"У вас уже есть свой профиль: {get_patient_name(pid) or pid}\n"
                f"Переключиться: /switch {pid}"
            )
            return

    _pending_action[user_id] = {"state": "awaiting_own_profile_name"}
    await update.message.reply_text(
        "Создаём ваш личный медицинский профиль.\n"
        "Как вас зовут? (Имя или ФИО)"
    )


async def cmd_newpatient(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /newpatient — create a patient profile for a family member."""
    user_id = update.effective_user.id
    if not is_registered(user_id):
        await update.message.reply_text("Сначала зарегистрируйтесь: /start")
        return

    _pending_action[user_id] = {"state": "awaiting_newpatient_name"}
    await update.message.reply_text(
        "Создаём профиль для близкого человека.\n"
        "Как зовут пациента? (Имя или ФИО)"
    )


async def cmd_invite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /invite — generate invite link for current patient profile."""
    user_id = update.effective_user.id
    patient_id = _check_access(user_id)
    if not patient_id:
        return

    patient_name = get_patient_name(patient_id) or patient_id
    has_owner = patient_has_owner(patient_id)

    args = context.args
    role = "family"
    if args and args[0].lower() in ("owner", "владелец"):
        if has_owner:
            await update.message.reply_text(
                f"У профиля «{patient_name}» уже есть владелец. "
                "Пригласить можно только как родственника."
            )
            return
        role = "owner"

    token = create_invite(user_id, patient_id, role=role)
    if not token:
        await update.message.reply_text("Не удалось создать приглашение.")
        return

    link = f"https://t.me/{BOT_NAME}?start=inv_{token}"
    role_label = "владелец (администратор)" if role == "owner" else "родственник"

    lines = [
        f"Приглашение для профиля «{patient_name}» (роль: {role_label}):",
        "",
        f"{link}",
        "",
        "Отправьте эту ссылку человеку. После перехода он получит доступ.",
    ]
    if not has_owner:
        lines.append("")
        lines.append("Чтобы пригласить самого пациента как владельца:")
        lines.append("/invite владелец")

    await update.message.reply_text("\n".join(lines))


async def cmd_access(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /access — show who has access to current patient."""
    user_id = update.effective_user.id
    patient_id = _check_access(user_id)
    if not patient_id:
        return

    patient_name = get_patient_name(patient_id) or patient_id
    access_list = get_patient_access_list(patient_id)

    lines = [f"Доступ к профилю «{patient_name}»:\n"]
    for entry in access_list:
        role = entry["role"]
        name = entry["name"] or f"ID:{entry['telegram_id']}"
        role_label = "👑 владелец" if role == "owner" else "👥 родственник"
        lines.append(f"  {role_label} — {name}")

    user_is_owner = is_owner(user_id, patient_id)
    if user_is_owner and len(access_list) > 1:
        lines.append("")
        lines.append("Вы владелец. Чтобы отключить доступ:")
        lines.append("/revoke <имя или ID>")

    await update.message.reply_text("\n".join(lines))


async def cmd_revoke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /revoke — owner removes someone's access."""
    user_id = update.effective_user.id
    patient_id = _check_access(user_id)
    if not patient_id:
        return

    if not is_owner(user_id, patient_id):
        await update.message.reply_text("Только владелец профиля может управлять доступом.")
        return

    args = context.args
    if not args:
        # Show list of who can be revoked
        access_list = get_patient_access_list(patient_id)
        others = [e for e in access_list if e["telegram_id"] != str(user_id)]
        if not others:
            await update.message.reply_text("Кроме вас никто не имеет доступа.")
            return
        lines = ["Кого отключить?\n"]
        for entry in others:
            name = entry["name"] or f"ID:{entry['telegram_id']}"
            lines.append(f"  /revoke {entry['telegram_id']} — {name}")
        await update.message.reply_text("\n".join(lines))
        return

    target = args[0]
    # Try to find by name or telegram_id
    access_list = get_patient_access_list(patient_id)
    target_id = None
    for entry in access_list:
        if entry["telegram_id"] == target:
            target_id = int(target)
            break
        if entry["name"].lower() == target.lower():
            target_id = int(entry["telegram_id"])
            break

    if not target_id:
        await update.message.reply_text(f"Пользователь '{target}' не найден среди имеющих доступ.")
        return

    if revoke_access(user_id, patient_id, target_id):
        await update.message.reply_text("Доступ отключён.")
    else:
        await update.message.reply_text("Не удалось отключить доступ.")


async def cmd_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /switch — switch active patient for multi-patient users."""
    user_id = update.effective_user.id
    accessible = get_accessible_patients(user_id)

    if not accessible:
        await update.message.reply_text(NOT_REGISTERED_TEXT)
        return

    if len(accessible) == 1:
        pid = list(accessible.keys())[0]
        name = get_patient_name(pid) or pid
        await update.message.reply_text(
            f"У вас один профиль: {name}\n\n"
            "Чтобы добавить близкого: /newpatient"
        )
        return

    # If argument given — switch directly
    args = context.args
    if args:
        target = args[0].lower()
        if target in accessible:
            set_active_patient(user_id, target)
            name = get_patient_name(target) or target
            role = accessible[target]
            role_label = "вы владелец" if role == "owner" else "вы родственник"
            await update.message.reply_text(f"Переключено на: {name} ({role_label})")
            return
        else:
            await update.message.reply_text(f"Профиль '{target}' не найден.")

    # Show list
    current = get_patient_id_by_telegram(user_id)
    lines = ["Доступные профили:\n"]
    for pid, role in accessible.items():
        name = get_patient_name(pid) or pid
        role_label = "👑 владелец" if role == "owner" else "👥 родственник"
        marker = " ← сейчас" if pid == current else ""
        lines.append(f"  /switch {pid} — {name} ({role_label}){marker}")

    lines.append("")
    lines.append("Добавить близкого: /newpatient")
    await update.message.reply_text("\n".join(lines))


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear — reset conversation history."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return
    agent = _get_agent(patient_id)
    agent.clear_history()
    lang = _get_lang(patient_id)
    msg = "Dialogo istorija išvalyta." if lang == "lt" else "История диалога очищена."
    await update.message.reply_text(msg)


async def cmd_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /files — list documents in knowledge base."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return
    agent = _get_agent(patient_id)
    summary = agent.get_summary()
    await update.message.reply_text(summary)


async def cmd_diary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /diary — show health diary entries."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return

    args = context.args
    n = 10
    if args and args[0].isdigit():
        n = int(args[0])

    entries = load_entries(last_n=n, patient_id=patient_id)
    text = format_entries(entries)

    if len(text) <= 4096:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), 4096):
            await update.message.reply_text(text[i : i + 4096])


async def cmd_labs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /labs — show lab value trends. /labs креатинин for detail."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return

    args = context.args
    if args:
        param = " ".join(args)
        text = format_param_detail(param, patient_id=patient_id)
    else:
        text = format_trends(patient_id=patient_id)

    if len(text) <= 4096:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), 4096):
            await update.message.reply_text(text[i : i + 4096])


async def cmd_sos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /sos — instant emergency card for paramedics."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return
    card = build_emergency_card(patient_id)
    await update.message.reply_text(card)


async def cmd_hospital(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /hospital — full summary for ER admission."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return
    card = build_emergency_card(patient_id)
    if len(card) <= 4096:
        await update.message.reply_text(card)
    else:
        for i in range(0, len(card), 4096):
            await update.message.reply_text(card[i : i + 4096])


async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /profile — show what the bot knows about the patient."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return
    from src.patient_manager import build_medical_summary
    summary = build_medical_summary(patient_id)
    if not summary:
        lang = _get_lang(patient_id)
        msg = "Profilis tuščias. Siųskite dokumentus." if lang == "lt" else "Профиль пуст. Загрузите документы."
        await update.message.reply_text(msg)
        return
    if len(summary) <= 4096:
        await update.message.reply_text(summary)
    else:
        for i in range(0, len(summary), 4096):
            await update.message.reply_text(summary[i : i + 4096])


async def cmd_meds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /meds — medication tracker. /meds, /meds add ..., /meds stop ..."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return

    lang = _get_lang(patient_id)
    args_text = " ".join(context.args) if context.args else ""
    parsed = parse_meds_command(args_text)

    if parsed["action"] == "list":
        text = format_medications(patient_id, lang=lang)
        await update.message.reply_text(text)

    elif parsed["action"] == "add":
        entry = add_medication(
            patient_id,
            name=parsed["name"],
            dose=parsed.get("dose", ""),
            duration_weeks=parsed.get("weeks", 0),
        )
        if parsed.get("weeks", 0) > 0:
            msg = f"✓ Курс добавлен: {entry['name']} {entry.get('dose', '')} до {entry.get('end_date', '')}"
        else:
            msg = f"✓ Добавлено: {entry['name']} {entry.get('dose', '')}"
        await update.message.reply_text(msg)

    elif parsed["action"] == "stop":
        found = stop_medication(patient_id, parsed["name"], result=parsed.get("result", ""))
        if found:
            await update.message.reply_text(f"✓ {parsed['name']} — курс завершён/отменён.")
        else:
            await update.message.reply_text(f"Препарат '{parsed['name']}' не найден в текущих.")


async def cmd_doctor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /doctor — prepare speech for a doctor visit."""
    patient_id = _check_access(update.effective_user.id)
    if not patient_id:
        return

    args = context.args
    if not args:
        lang = _get_lang(patient_id)
        if lang == "lt":
            await update.message.reply_text(
                "Nurodykite specialybę po komandos.\n\n"
                "Pavyzdžiai:\n"
                "/doctor nefrologas\n"
                "/doctor urologas\n"
                "/doctor terapeutas"
            )
        else:
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
        agent = _get_agent(patient_id)
        prompt = DOCTOR_VISIT_PROMPT.format(specialty=specialty)
        user_id = update.effective_user.id
        sender = get_user_name(user_id) or update.effective_user.first_name
        answer = agent.ask(prompt, sender_telegram_id=user_id, sender_name=sender)
        if len(answer) <= 4096:
            await update.message.reply_text(answer)
        else:
            for i in range(0, len(answer), 4096):
                await update.message.reply_text(answer[i : i + 4096])
    except Exception as e:
        logger.error("Error preparing doctor visit: %s", e, exc_info=True)
        await update.message.reply_text("Ошибка при подготовке. Попробуй позже.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages — registration flow or Q&A."""
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        return

    # --- Handle pending registration / newpatient ---
    pending = _pending_action.get(user_id)
    if pending:
        if pending["state"] == "awaiting_name":
            del _pending_action[user_id]
            name = text
            patient_id = create_self_patient(user_id, name)
            # Update user name from input
            update_user_name(user_id, name)
            lang = _get_lang(patient_id)
            await update.message.reply_text(
                f"Профиль создан: {name}\n\n"
                "Теперь вы можете:\n"
                "• Отправить медицинский документ (фото, PDF) — я сохраню в вашу базу\n"
                "• Задать вопрос о здоровье\n"
                "• Создать профиль для близкого: /newpatient\n\n"
                "Команды: /sos /doctor /labs /meds /diary /profile"
            )
            return

        if pending["state"] == "awaiting_own_profile_name":
            del _pending_action[user_id]
            name = text
            patient_id = create_self_patient(user_id, name)
            update_user_name(user_id, name)
            set_active_patient(user_id, patient_id)
            await update.message.reply_text(
                f"Ваш профиль создан: {name}\n\n"
                "Теперь вы можете загружать свои медицинские документы.\n"
                "Переключиться между профилями: /switch"
            )
            return

        if pending["state"] == "awaiting_newpatient_name":
            del _pending_action[user_id]
            patient_name = text
            patient_id = create_patient_for_other(user_id, patient_name)
            # Auto-switch to the new patient
            set_active_patient(user_id, patient_id)
            await update.message.reply_text(
                f"Профиль «{patient_name}» создан. Вы — родственник-управляющий.\n\n"
                "Загружайте медицинские документы — я сохраню в базу знаний.\n\n"
                "Чтобы пригласить самого пациента как владельца:\n"
                "/invite владелец\n\n"
                "Чтобы пригласить другого родственника:\n"
                "/invite\n\n"
                "Переключиться обратно на свой профиль: /switch"
            )
            return

    # --- Normal flow: Q&A ---
    patient_id = _check_access(user_id)
    if not patient_id:
        await update.message.reply_text(NOT_REGISTERED_TEXT)
        return

    question = text

    role = get_user_role(user_id, patient_id)

    # Auto-detect doctor visit — prepare speech automatically
    doctor_specialty = _detect_doctor_visit(question)
    if doctor_specialty:
        await update.message.reply_text(f"Готовлю речь для {doctor_specialty}...")
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
        agent = _get_agent(patient_id)
        prompt = DOCTOR_VISIT_PROMPT.format(specialty=doctor_specialty)
        sender = get_user_name(user_id) or update.effective_user.first_name
        answer = agent.ask(prompt, sender_telegram_id=user_id, sender_name=sender)
        if len(answer) <= 4096:
            await update.message.reply_text(answer)
        else:
            for i in range(0, len(answer), 4096):
                await update.message.reply_text(answer[i : i + 4096])
        return

    # Auto-detect health status — diary
    if is_health_status(question):
        if role == "patient":
            entry = save_entry(question, patient_id=patient_id)
            await update.message.reply_text(f"📝 Записано в дневник ({entry['timestamp']})")
        elif role == "family":
            name = get_user_name(user_id) or "Родственник"
            entry = save_entry(question, reported_by=name, patient_id=patient_id)
            await update.message.reply_text(
                f"📝 Записано в дневник от {name} ({entry['timestamp']})"
            )

    await _ask_and_reply(update, context, question, patient_id)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages — transcribe and answer."""
    user_id = update.effective_user.id
    patient_id = _check_access(user_id)
    if not patient_id:
        return

    lang = _get_lang(patient_id)
    voice = update.message.voice
    dur = voice.duration or 0
    if dur > 15:
        eta = max(1, dur // 5)
        note = (
            f"Klausau... Garso atpažinimas užtruks ~{eta} min."
            if lang == "lt"
            else f"Слушаю... Расшифровка займёт ~{eta} мин."
        )
    else:
        note = "Klausau..." if lang == "lt" else "Слушаю..."
    await update.message.reply_text(note)

    tmp_path = None
    try:
        tg_file = await voice.get_file()

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await tg_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        text = transcribe_voice(tmp_path, language=lang)

        await update.message.reply_text(f"{'Atpažinta' if lang == 'lt' else 'Распознано'}: {text}")

        # Auto-detect health status and save to diary
        role = get_user_role(user_id, patient_id)
        if is_health_status(text):
            if role == "patient":
                entry = save_entry(text, patient_id=patient_id)
                await update.message.reply_text(f"📝 Записано в дневник ({entry['timestamp']})")
            elif role == "family":
                name = get_user_name(user_id) or "Родственник"
                entry = save_entry(text, reported_by=name, patient_id=patient_id)
                await update.message.reply_text(
                    f"📝 Записано в дневник от {name} ({entry['timestamp']})"
                )

        await _ask_and_reply(update, context, text, patient_id)

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
    user_id = update.effective_user.id
    patient_id = _check_access(user_id)
    if not patient_id:
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
        vs = _get_vector_store(patient_id)
        chunks = vs.add_document(
            result["text"],
            {"file_name": file_name, "file_type": result["file_type"]},
        )

        await update.message.reply_text(
            f"Готово! {file_name}: {chunks} фрагментов ({result['char_count']} символов) "
            f"добавлено в базу знаний."
        )

        # Auto-extract profile data from document
        if result["char_count"] > 50:
            updated = process_document_for_profile(patient_id, result["text"])
            if updated and updated.get("auto_extracted"):
                agent = _get_agent(patient_id)
                agent.reload_summary()
                await update.message.reply_text("📋 Профиль пациента обновлён из документа.")

    except Exception as e:
        logger.error("Error processing document: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка при обработке {file_name}: {e}")
    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle audio file uploads — transcribe with Whisper and analyze visit."""
    user_id = update.effective_user.id
    patient_id = _check_access(user_id)
    if not patient_id:
        return

    lang = _get_lang(patient_id)

    # Audio can come as audio message or as document
    if update.message.audio:
        tg_file = await update.message.audio.get_file()
        file_name = update.message.audio.file_name or "audio.mp3"
        duration = update.message.audio.duration or 0
    elif update.message.document:
        ext = Path(update.message.document.file_name or "").suffix.lower()
        if ext not in SUPPORTED_AUDIO:
            return
        tg_file = await update.message.document.get_file()
        file_name = update.message.document.file_name or "audio"
        duration = 0
    else:
        return

    duration_str = f" ({duration // 60}:{duration % 60:02d})" if duration else ""
    eta = max(1, duration // 5) if duration else None
    eta_str = f"\n⏱ Примерное время: ~{eta} мин." if eta else ""
    await update.message.reply_text(
        f"Транскрибирую {file_name}{duration_str}...{eta_str}"
    )

    tmp_path = None
    try:
        ext = Path(file_name).suffix.lower() or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            await tg_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        result = transcribe_audio(tmp_path, language=lang)
        duration_min = result["duration"] / 60

        transcript_path = save_transcript(result, file_name, patient_id=patient_id)

        await update.message.reply_text(
            f"Транскрибация завершена: {duration_min:.1f} мин, "
            f"{len(result['segments'])} сегментов.\n"
            f"Сохранено: {transcript_path.name}\n\n"
            "Анализирую визит..."
        )

        md_text = format_transcript_md(result, file_name)
        vs = _get_vector_store(patient_id)
        vs.add_document(
            md_text,
            {"file_name": transcript_path.name, "file_type": "transcript"},
        )

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )
        agent = _get_agent(patient_id)
        analysis_prompt = VISIT_ANALYSIS_PROMPT.format(transcript=md_text)
        sender = get_user_name(user_id) or update.effective_user.first_name
        answer = agent.ask(analysis_prompt, sender_telegram_id=user_id, sender_name=sender)

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
    user_id = update.effective_user.id
    patient_id = _check_access(user_id)
    if not patient_id:
        return

    await update.message.reply_text("Обрабатываю фото (OCR)...")

    tmp_path = None
    try:
        photo = update.message.photo[-1]
        tg_file = await photo.get_file()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            await tg_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        result = process_file(tmp_path)
        file_name = f"photo_{photo.file_unique_id}.jpg"
        vs = _get_vector_store(patient_id)
        chunks = vs.add_document(
            result["text"],
            {"file_name": file_name, "file_type": "image"},
        )

        preview = result["text"][:300]
        if len(result["text"]) > 300:
            preview += "..."

        await update.message.reply_text(
            f"Фото обработано: {chunks} фрагментов ({result['char_count']} символов).\n\n"
            f"Распознанный текст:\n{preview}"
        )

        # Auto-extract profile data from photo
        if result["char_count"] > 50:
            updated = process_document_for_profile(patient_id, result["text"])
            if updated and updated.get("auto_extracted"):
                agent_inst = _get_agent(patient_id)
                agent_inst.reload_summary()
                await update.message.reply_text("📋 Профиль пациента обновлён из документа.")

        # Auto-extract lab values and save to tracker
        lab_values = extract_lab_values(result["text"])
        if lab_values:
            saved = save_lab_values(lab_values, patient_id=patient_id)
            if saved:
                param_list = ", ".join(f"{v['param']} {v['value']}" for v in lab_values[:5])
                await update.message.reply_text(
                    f"📊 Найдено {len(lab_values)} показателей, сохранено {saved} новых: {param_list}"
                )

        # Auto-check for prescriptions/procedures
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
            agent = _get_agent(patient_id)
            check_prompt = PRESCRIPTION_CHECK_PROMPT.format(text=result["text"])
            sender = get_user_name(user_id) or update.effective_user.first_name
            answer = agent.ask(check_prompt, sender_telegram_id=user_id, sender_name=sender)
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

    from src.patient_manager import get_all_patients
    patients = get_all_patients()
    if not patients:
        logger.warning("No patients registered. Use register_patient() to add patients.")
    else:
        logger.info("Registered patients: %s", list(patients.keys()))

    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("switch", cmd_switch))
    app.add_handler(CommandHandler("me", cmd_me))
    app.add_handler(CommandHandler("newpatient", cmd_newpatient))
    app.add_handler(CommandHandler("invite", cmd_invite))
    app.add_handler(CommandHandler("access", cmd_access))
    app.add_handler(CommandHandler("revoke", cmd_revoke))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("files", cmd_files))
    app.add_handler(CommandHandler("labs", cmd_labs))
    app.add_handler(CommandHandler("sos", cmd_sos))
    app.add_handler(CommandHandler("hospital", cmd_hospital))
    app.add_handler(CommandHandler("doctor", cmd_doctor))
    app.add_handler(CommandHandler("diary", cmd_diary))
    app.add_handler(CommandHandler("profile", cmd_profile))
    app.add_handler(CommandHandler("meds", cmd_meds))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot started (multi-patient mode). Patients: %s", list(patients.keys()) if patients else "none")
    app.run_polling()


if __name__ == "__main__":
    main()
