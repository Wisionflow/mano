"""Medical AI agent powered by Claude for Q&A over patient documents."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from src.vector_store import VectorStore
from src.health_diary import get_diary_context

load_dotenv()

# Load medical summary if available — gives the agent full context
_SUMMARY_PATH = Path(__file__).resolve().parent.parent / "med_docs_olga" / "MEDICAL_SUMMARY.md"
_MEDICAL_SUMMARY = ""
if _SUMMARY_PATH.exists():
    _MEDICAL_SUMMARY = _SUMMARY_PATH.read_text(encoding="utf-8")

SYSTEM_PROMPT = """Ты — персональный медицинский ассистент. Твоя задача — помогать разобраться
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

СТИЛЬ ОТВЕТОВ:
- Отвечай на русском языке
- Говори тепло, по-человечески — ты разговариваешь с пациенткой, а не пишешь медицинскую статью
- Обращайся на «ты» — это близкий человек, не незнакомка в поликлинике
- Объясняй простым языком, без жаргона (но если нужен термин — сразу объясни его)
- Структурируй ответы: сначала короткий прямой ответ, потом детали
- Ссылайся на конкретные документы и даты, когда это возможно
- Если нашёл что-то опасное — скажи прямо, не смягчай

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
    """Claude-powered medical Q&A agent with RAG."""

    def __init__(self, vector_store: VectorStore = None):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Create a .env file with your key.")

        self.client = Anthropic(api_key=api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.vector_store = vector_store or VectorStore()
        self.conversation_history: List[Dict[str, str]] = []
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

    def ask(self, question: str) -> str:
        """Ask a question about the patient's medical history."""
        context = self._retrieve_context(question)

        # Build system prompt — extend for medication queries
        system = SYSTEM_PROMPT
        if self._is_medication_query(question):
            system += MEDICATION_PROMPT_ADDITION

        # Build message with medical summary + diary + RAG context
        parts = []
        if self.medical_summary:
            parts.append(f"МЕДИЦИНСКАЯ СВОДКА ПАЦИЕНТА:\n{self.medical_summary}")
        diary = get_diary_context(last_n=10)
        if diary:
            parts.append(diary)
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
