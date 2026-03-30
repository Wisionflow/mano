"""Main Gradio web application for Mano — personal medical assistant."""

import os
import tempfile
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from src.document_processor import process_file, SUPPORTED_EXTENSIONS
from src.vector_store import VectorStore
from src.medical_agent import MedicalAgent
from src.medication_lookup import lookup_medication
from src.analytics import search_parameter_history, create_parameter_chart

load_dotenv()

# Initialize components
vector_store = VectorStore()
agent = MedicalAgent(vector_store=vector_store)

DISCLAIMER = """⚠️ **Медицинский дисклеймер**

Этот ассистент — инструмент для организации и понимания вашей медицинской информации.
Он **НЕ заменяет консультацию врача** и не ставит диагнозы.
Все данные хранятся **локально на вашем устройстве**."""


# --- Chat Tab ---

def chat_respond(message: str, history: list) -> tuple:
    """Handle chat messages."""
    if not message.strip():
        return "", history

    response = agent.ask(message)
    history.append((message, response))
    return "", history


def clear_chat():
    """Clear chat history."""
    agent.clear_history()
    return [], ""


# --- Upload Tab ---

def upload_files(files) -> str:
    """Process and ingest uploaded files."""
    if not files:
        return "Файлы не выбраны."

    results = []
    for file in files:
        file_path = file.name if hasattr(file, "name") else str(file)
        ext = Path(file_path).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            results.append(f"✗ {Path(file_path).name}: формат {ext} не поддерживается")
            continue

        try:
            doc = process_file(file_path)
            chunks = vector_store.add_document(doc["text"], {
                "file_name": doc["file_name"],
                "file_type": doc["file_type"],
            })
            results.append(f"✓ {doc['file_name']}: {chunks} фрагментов ({doc['char_count']} символов)")
        except Exception as e:
            results.append(f"✗ {Path(file_path).name}: ошибка — {e}")

    summary = agent.get_summary()
    return "\n".join(results) + f"\n\n{summary}"


def get_files_list() -> str:
    """Get list of files in the knowledge base."""
    return agent.get_summary()


def delete_file(file_name: str) -> str:
    """Delete a file from the knowledge base."""
    if not file_name.strip():
        return "Введите имя файла."
    deleted = vector_store.delete_file(file_name.strip())
    if deleted:
        return f"Удалено {deleted} фрагментов файла '{file_name}'.\n\n{agent.get_summary()}"
    return f"Файл '{file_name}' не найден в базе."


# --- Medication Tab ---

def search_medication(name: str) -> str:
    """Search for medication alternatives."""
    if not name.strip():
        return "Введите название препарата."
    return lookup_medication(name.strip())


# --- Analytics Tab ---

def analyze_parameter(parameter_name: str):
    """Search for lab parameter and create chart."""
    if not parameter_name.strip():
        return "Введите название параметра (например: гемоглобин, глюкоза, холестерин).", None

    entries = search_parameter_history(vector_store, parameter_name.strip())

    if not entries:
        return f"Значения параметра '{parameter_name}' не найдены в документах.", None

    text_results = []
    for i, e in enumerate(entries, 1):
        text_results.append(
            f"{i}. **{e['value']}** {e['unit']} — источник: {e['source']}"
        )

    chart = create_parameter_chart(entries, parameter_name.strip())
    return "\n".join(text_results), chart


def analyze_dynamics_ai(indicator: str) -> str:
    """Use Claude to analyze indicator dynamics across documents."""
    if not indicator.strip():
        return "Введите название показателя."
    return agent.analyze_dynamics(indicator.strip())


# --- Build UI ---

def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="Медицинский ассистент",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# 🏥 Медицинский ассистент")
        gr.Markdown(DISCLAIMER)

        with gr.Tabs():
            # Tab 1: Chat
            with gr.Tab("💬 Чат с ассистентом"):
                chatbot = gr.Chatbot(
                    label="Диалог",
                    height=500,
                    show_copy_button=True,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ваш вопрос",
                        placeholder="Например: Какие были последние результаты анализа крови?",
                        scale=4,
                    )
                    send_btn = gr.Button("Отправить", variant="primary", scale=1)

                clear_btn = gr.Button("🗑 Очистить диалог")

                msg.submit(chat_respond, [msg, chatbot], [msg, chatbot])
                send_btn.click(chat_respond, [msg, chatbot], [msg, chatbot])
                clear_btn.click(clear_chat, outputs=[chatbot, msg])

            # Tab 2: Upload
            with gr.Tab("📄 Загрузка документов"):
                gr.Markdown("### Загрузите медицинские документы")
                gr.Markdown(
                    f"Поддерживаемые форматы: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )

                file_upload = gr.File(
                    label="Выберите файлы",
                    file_count="multiple",
                    type="filepath",
                )
                upload_btn = gr.Button("📥 Загрузить в базу знаний", variant="primary")
                upload_output = gr.Textbox(label="Результат", lines=10, interactive=False)

                upload_btn.click(upload_files, inputs=[file_upload], outputs=[upload_output])

                gr.Markdown("---")
                gr.Markdown("### Управление базой знаний")
                files_info = gr.Textbox(label="Файлы в базе", lines=5, interactive=False)
                refresh_btn = gr.Button("🔄 Обновить список")
                refresh_btn.click(get_files_list, outputs=[files_info])

                with gr.Row():
                    del_name = gr.Textbox(label="Имя файла для удаления", scale=3)
                    del_btn = gr.Button("🗑 Удалить", variant="stop", scale=1)
                del_output = gr.Textbox(label="Результат удаления", interactive=False)
                del_btn.click(delete_file, inputs=[del_name], outputs=[del_output])

            # Tab 3: Medications
            with gr.Tab("💊 Поиск аналогов лекарств"):
                gr.Markdown("### Поиск аналогов препаратов")
                gr.Markdown(
                    "Введите название препарата, чтобы найти аналоги с тем же действующим веществом."
                )

                med_name = gr.Textbox(
                    label="Название препарата",
                    placeholder="Например: Аторвастатин, Омепразол, Метформин",
                )
                med_btn = gr.Button("🔍 Найти аналоги", variant="primary")
                med_output = gr.Markdown(label="Результат")

                med_btn.click(search_medication, inputs=[med_name], outputs=[med_output])

            # Tab 4: Analytics
            with gr.Tab("📊 Динамика показателей"):
                gr.Markdown("### Отслеживание показателей здоровья")
                gr.Markdown(
                    "Введите название лабораторного показателя для поиска его значений в документах."
                )

                param_name = gr.Textbox(
                    label="Название показателя",
                    placeholder="Например: гемоглобин, глюкоза, холестерин, СОЭ",
                )
                with gr.Row():
                    param_btn = gr.Button("📈 Показать динамику", variant="primary", scale=1)
                    dynamics_btn = gr.Button("🤖 AI-анализ динамики", variant="secondary", scale=1)

                param_text = gr.Markdown(label="Найденные значения")
                param_chart = gr.Plot(label="График")
                dynamics_output = gr.Markdown(label="AI-анализ")

                param_btn.click(
                    analyze_parameter,
                    inputs=[param_name],
                    outputs=[param_text, param_chart],
                )
                dynamics_btn.click(
                    analyze_dynamics_ai,
                    inputs=[param_name],
                    outputs=[dynamics_output],
                )

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    # Use uvicorn directly to avoid Gradio's startup self-check
    # which can fail on some Windows configurations
    gradio_app = gr.routes.App.create_app(app)
    print("* Медицинский ассистент запущен: http://127.0.0.1:7860")
    uvicorn.run(gradio_app, host="127.0.0.1", port=7860)
