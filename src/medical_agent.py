"""Mano — medical AI agent powered by Claude for Q&A over patient documents.

Supports multi-patient: pass patient_id to constructor for isolated data.
Legacy mode (no patient_id) uses the original med_docs_olga/ MEDICAL_SUMMARY.md.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from src.vector_store import VectorStore
from src.health_diary import get_diary_context

load_dotenv()

# Legacy single-patient summary (backward compat)
_SUMMARY_PATH = Path(__file__).resolve().parent.parent / "med_docs_olga" / "MEDICAL_SUMMARY.md"
_MEDICAL_SUMMARY = ""
if _SUMMARY_PATH.exists():
    _MEDICAL_SUMMARY = _SUMMARY_PATH.read_text(encoding="utf-8")

SYSTEM_PROMPT = """Ты — Mano, персональный медицинский помощник. Твоя задача — помогать разобраться
в медицинских документах, анализах и истории лечения конкретного пациента.

ВАЖНЕЙШЕЕ ПРАВИЛО — ПРОВЕРКА ЛЕКАРСТВ:
Врачи узконаправлены и часто НЕ видят полную картину. Приём длится 12 минут — невозможно
учесть все диагнозы. Ты ОБЯЗАН проверять КАЖДОЕ назначение:
1. Совместимость со ВСЕМИ текущими лекарствами пациента
2. Противопоказания по ВСЕМ диагнозам (не только по профилю назначившего врача)
3. Нагрузку на почки (единственная почка, рСКФ 32 мл/мин — критически низкая!)
4. Нагрузку на печень (АЛТ на верхней границе)
5. Проверку по списку запрещённых препаратов (аллергии и непереносимости)
6. Допускай мысль что врач мог ошибиться
Если пациентка сообщает о новом назначении — СРАЗУ проведи полную проверку и объясни
простым языком что будет происходить, какие побочки возможны, на что обратить внимание.

РЕАЛЬНЫЙ ИНЦИДЕНТ: нефролог заменил габапентин+дулоксетин на карбамазепин+венлафаксин
без учёта аллергоанамнеза и синдрома отмены → аллергия, панические атаки, вызов скорой.
Такого больше быть НЕ должно.

ТВОИ ВОЗМОЖНОСТИ:
- Отвечать на вопросы по конкретным анализам и показателям из документов
- Объяснять медицинские термины простым языком
- Сравнивать показатели в динамике (как менялись со временем)
- Находить аналоги лекарств (с тем же действующим веществом)
- Формулировать вопросы для врача на основе данных
- Структурировать информацию из разных документов
- Давать рекомендации по образу жизни (питание, напитки, физическая активность, сон)
  с учётом ВСЕХ диагнозов, лекарств, ограничений и рисков пациента
- ПРОВЕРЯТЬ назначения врачей на безопасность для ЭТОГО конкретного пациента

ТВОИ ОГРАНИЧЕНИЯ:
- Ты НЕ ставишь диагнозы
- Ты НЕ отменяешь и НЕ заменяешь назначения врача — но ты ПРЕДУПРЕЖДАЕШЬ о рисках
- При подборе аналогов лекарств ВСЕГДА уточняй: "согласуйте с врачом перед заменой"
- При рекомендациях по образу жизни ВСЕГДА учитывай взаимодействие с текущими лекарствами
  и все диагнозы одновременно — не рассматривай каждый диагноз отдельно
- Если данных недостаточно — честно говори об этом

СТИЛЬ ОТВЕТОВ — САМОЕ ВАЖНОЕ:
- КОРОТКО. 2-4 предложения. Как тёплый друг-врач, не как энциклопедия.
- Сначала ПРЯМОЙ ответ на вопрос. Без предисловий, без перечисления всего что знаешь.
- Детали — ТОЛЬКО если попросят. Не вываливай всё сразу.
- Если вопрос неясен — СПРОСИ уточнение, не угадывай.
- Обращайся на «ты». Тепло, по-человечески.
- Простой язык, без жаргона.
- Если нашёл что-то опасное — скажи прямо и коротко.
- НИКОГДА не перечисляй все диагнозы, все лекарства, всю историю в ответе.
  Пациентка устаёт, ей плохо. Длинный ответ она не прочитает = бесполезный ответ.
- Отвечай на русском языке

ДОКУМЕНТЫ ПАЦИЕНТА:
Ниже будет предоставлен контекст из медицинских документов пациента. Используй только эту
информацию для ответов о конкретных показателях. Не придумывай данные."""

MEDICATION_PROMPT_ADDITION = """

ПРИ ЗАПРОСЕ АНАЛОГОВ ЛЕКАРСТВ:
1. Определи действующее вещество (МНН) препарата
2. Назови другие торговые названия с тем же МНН
3. Укажи форму выпуска и дозировку оригинала
4. Предупреди о необходимости консультации с врачом
5. Отметь если есть дженерики — они обычно дешевле"""

MEDICATION_KEYWORDS = [
    "аналог", "замен", "препарат", "лекарств", "таблетк", "капсул",
    "мг", "мкг", "действующее вещество", "дженерик",
]


class MedicalAgent:
    """Claude-powered medical Q&A agent with RAG.

    Multi-patient: pass patient_id to use patient-specific data.
    Legacy: no patient_id uses global med_docs_olga/MEDICAL_SUMMARY.md.
    """

    def __init__(self, vector_store: VectorStore = None, patient_id: str = None):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Create a .env file with your key.")

        self.client = Anthropic(api_key=api_key, timeout=120.0, max_retries=3)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.patient_id = patient_id
        self.vector_store = vector_store or VectorStore()
        self.conversation_history: List[Dict[str, str]] = []

        # Load medical summary — dynamic from profile or legacy static file
        if patient_id:
            from src.patient_manager import build_medical_summary
            self.medical_summary = build_medical_summary(patient_id)
        else:
            self.medical_summary = _MEDICAL_SUMMARY

    def _retrieve_context(self, query: str, n_results: int = 8) -> str:
        """Retrieve relevant document chunks for the query."""
        results = self.vector_store.search(query, n_results=n_results)

        if not results:
            return "Документы по данному запросу не найдены в базе."

        context_parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            header = f"[Документ {i}: {meta.get('file_name', '?')}"
            if meta.get("detected_date"):
                header += f", дата: {meta['detected_date']}"
            if meta.get("document_type"):
                header += f", тип: {meta['document_type']}"
            header += "]"
            context_parts.append(f"{header}\n{r['text']}")

        return "\n\n---\n\n".join(context_parts)

    def _is_medication_query(self, question: str) -> bool:
        """Detect if question is about medications/alternatives."""
        q = question.lower()
        return any(kw in q for kw in MEDICATION_KEYWORDS)

    def reload_summary(self):
        """Reload medical summary from profile (call after profile update)."""
        if self.patient_id:
            from src.patient_manager import build_medical_summary
            self.medical_summary = build_medical_summary(self.patient_id)

    def ask(self, question: str, sender_telegram_id: int = None, sender_name: str = None) -> str:
        """Ask a question about the patient's medical history."""
        context = self._retrieve_context(question)

        # Build system prompt — extend for medication queries
        system = SYSTEM_PROMPT
        if self._is_medication_query(question):
            system += MEDICATION_PROMPT_ADDITION

        # For multi-patient: adapt language and healthcare system
        if self.patient_id:
            from src.patient_manager import load_profile
            profile = load_profile(self.patient_id)
            if profile:
                if profile.get("language") == "lt":
                    system += "\n\nОТВЕЧАЙ НА ЛИТОВСКОМ ЯЗЫКЕ (lietuvių kalba). Пациент говорит по-литовски."

                # Load healthcare system context
                hs = profile.get("healthcare_system", "")
                if hs:
                    hs_path = Path(__file__).resolve().parent.parent / "config" / "healthcare_systems" / f"{hs}.md"
                    if hs_path.exists():
                        hs_text = hs_path.read_text(encoding="utf-8")
                        system += f"\n\nКОНТЕКСТ СИСТЕМЫ ЗДРАВООХРАНЕНИЯ:\n{hs_text}"

        # Build message with medical summary + diary + RAG context
        parts = []
        if self.medical_summary:
            parts.append(f"МЕДИЦИНСКАЯ СВОДКА ПАЦИЕНТА:\n{self.medical_summary}")
        diary = get_diary_context(last_n=10, patient_id=self.patient_id)
        if diary:
            parts.append(diary)

        # Include current medications context
        if self.patient_id:
            from src.medication_tracker import get_current_medications, get_active_courses
            current_meds = get_current_medications(self.patient_id)
            active_courses = get_active_courses(self.patient_id)
            if current_meds or active_courses:
                med_lines = ["ТЕКУЩИЕ ЛЕКАРСТВА (из трекера):"]
                for m in current_meds:
                    line = f"• {m['name']} {m.get('dose', '')} {m.get('frequency', '')}"
                    if m.get("critical"):
                        line += " [КРИТИЧЕСКИ ВАЖНО — НЕ ОТМЕНЯТЬ]"
                    med_lines.append(line)
                for c in active_courses:
                    line = f"• {c['name']} {c.get('dose', '')} (курс до {c.get('end_date', '?')})"
                    med_lines.append(line)
                parts.append("\n".join(med_lines))

        # Load persistent conversation context for continuity across sessions
        if self.patient_id and not self.conversation_history:
            from src.patient_manager import get_conversation_context
            conv_ctx = get_conversation_context(self.patient_id, limit=10)
            if conv_ctx:
                parts.append(conv_ctx)

        parts.append(f"КОНТЕКСТ ИЗ ДОКУМЕНТОВ ПАЦИЕНТА:\n{context}")
        parts.append(f"ВОПРОС: {question}")

        message_with_context = "\n\n---\n\n".join(parts)

        # Send full history + new message to API
        api_messages = self.conversation_history.copy()
        api_messages.append({"role": "user", "content": message_with_context})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=api_messages,
        )

        answer = response.content[0].text

        # Store clean question in history (without RAG context)
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Keep history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        # Persist to conversation DB
        if self.patient_id:
            from src.patient_manager import save_message
            save_message(self.patient_id, "user", question,
                         sender_telegram_id=sender_telegram_id,
                         sender_name=sender_name)
            save_message(self.patient_id, "assistant", answer)

        return answer

    def analyze_dynamics(self, indicator: str) -> str:
        """Track a specific indicator across all documents over time."""
        query = f"{indicator} показатель значение результат анализ"
        context = self._retrieve_context(query, n_results=15)

        prompt = f"""Из контекста медицинских документов извлеки все упоминания показателя "{indicator}".

Для каждого упоминания определи:
1. Дата документа (если есть)
2. Значение показателя и единицы измерения
3. Референсные значения (норма), если указаны
4. Отклонение от нормы (если есть)

После извлечения данных:
- Опиши динамику: как менялся показатель со временем
- Выдели тревожные значения (если есть)
- Сделай вывод о тренде (улучшение / ухудшение / стабильно)

КОНТЕКСТ:
{context}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history = []

    def get_summary(self) -> str:
        """Generate a summary of all loaded documents."""
        files = self.vector_store.get_all_files()
        count = self.vector_store.get_document_count()

        if not files:
            return "База знаний пуста. Загрузите документы через вкладку 'Загрузка документов'."

        file_list = "\n".join(f"  • {f}" for f in files)
        return f"В базе знаний: {len(files)} документов, {count} фрагментов.\n\nФайлы:\n{file_list}"
