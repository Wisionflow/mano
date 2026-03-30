"""Audio transcription for medical visit recordings using Whisper.

Transcribes long audio files (doctor visits, consultations) using
OpenAI Whisper large-v3 locally with CUDA. No diarization — speaker
identification is done by the medical agent via context analysis.
"""

import os
import logging
from pathlib import Path
from datetime import datetime

import whisper

logger = logging.getLogger(__name__)

# Whisper model — loaded lazily
_model = None
_model_name = os.getenv("WHISPER_MODEL", "large-v3")

SUPPORTED_AUDIO = {".mp3", ".m4a", ".ogg", ".oga", ".wav", ".flac", ".webm", ".mp4"}

# Legacy output directory for transcripts
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "med_docs_olga"


def _transcript_dir(patient_id: str = None) -> Path:
    """Get transcript output directory — patient-specific or legacy."""
    if patient_id:
        from src.patient_manager import get_patient_dir
        return get_patient_dir(patient_id) / "documents"
    return TRANSCRIPT_DIR


def _get_model():
    """Lazy-load Whisper model."""
    global _model
    if _model is None:
        logger.info("Loading Whisper model '%s'...", _model_name)
        _model = whisper.load_model(_model_name)
        logger.info("Whisper model loaded.")
    return _model


def transcribe_audio(file_path: str, language: str = "ru") -> dict:
    """Transcribe an audio file using Whisper.

    Returns dict with keys:
        - text: full transcription text
        - segments: list of {start, end, text} segments
        - language: detected language
        - duration: audio duration in seconds
    """
    model = _get_model()

    logger.info("Transcribing: %s", file_path)
    result = model.transcribe(
        file_path,
        language=language,
        verbose=False,
        word_timestamps=False,
    )

    duration = result["segments"][-1]["end"] if result["segments"] else 0

    return {
        "text": result["text"].strip(),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ],
        "language": result.get("language", language),
        "duration": duration,
    }


def format_transcript_md(transcription: dict, source_filename: str = "") -> str:
    """Format transcription result as markdown with timestamps."""
    lines = []
    lines.append(f"# Транскрипт: {source_filename or 'аудиозапись'}")
    lines.append("")
    lines.append(f"- **Дата транскрибации:** {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    duration_min = transcription["duration"] / 60
    lines.append(f"- **Длительность:** {duration_min:.1f} мин")
    lines.append(f"- **Язык:** {transcription['language']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for seg in transcription["segments"]:
        start_m, start_s = divmod(int(seg["start"]), 60)
        end_m, end_s = divmod(int(seg["end"]), 60)
        ts = f"[{start_m:02d}:{start_s:02d}-{end_m:02d}:{end_s:02d}]"
        lines.append(f"{ts} {seg['text']}")

    lines.append("")
    return "\n".join(lines)


def save_transcript(transcription: dict, source_filename: str, patient_id: str = None) -> Path:
    """Save transcription as markdown file."""
    out_dir = _transcript_dir(patient_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    stem = Path(source_filename).stem
    # Sanitize filename
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)
    out_name = f"{date_str}_transcript_{safe_stem}.md"
    out_path = out_dir / out_name

    md = format_transcript_md(transcription, source_filename)
    out_path.write_text(md, encoding="utf-8")
    logger.info("Transcript saved: %s", out_path)
    return out_path
