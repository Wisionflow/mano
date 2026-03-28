"""Document processor: extracts text from PDFs, images (OCR), Excel files, and plain text."""

import os
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
import pytesseract
import easyocr
from PIL import Image
import pandas as pd


# Initialize EasyOCR reader (Russian + English) — lazy loaded
_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["ru", "en"], gpu=False)
    return _easyocr_reader


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

            # Also extract tables
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    cleaned = [str(cell) if cell else "" for cell in row]
                    text_parts.append(" | ".join(cleaned))

    return "\n".join(text_parts)


def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image using pytesseract first, fallback to EasyOCR."""
    # Try pytesseract (faster, good for clear scans)
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang="rus+eng")
        if text.strip() and len(text.strip()) > 20:
            return text.strip()
    except Exception:
        pass

    # Fallback to EasyOCR (better for photos, handwriting)
    reader = _get_easyocr_reader()
    results = reader.readtext(file_path, detail=0)
    return "\n".join(results)


def extract_text_from_excel(file_path: str) -> str:
    """Extract text from Excel files, converting all sheets to readable text."""
    text_parts = []
    xls = pd.ExcelFile(file_path)

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        text_parts.append(f"=== Лист: {sheet_name} ===")
        text_parts.append(df.to_string(index=False))

    return "\n\n".join(text_parts)


def extract_text_from_text_file(file_path: str) -> str:
    """Read plain text files."""
    encodings = ["utf-8", "cp1251", "latin-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


# Map extensions to processors
PROCESSORS = {
    ".pdf": extract_text_from_pdf,
    ".png": extract_text_from_image,
    ".jpg": extract_text_from_image,
    ".jpeg": extract_text_from_image,
    ".tiff": extract_text_from_image,
    ".tif": extract_text_from_image,
    ".bmp": extract_text_from_image,
    ".xlsx": extract_text_from_excel,
    ".xls": extract_text_from_excel,
    ".txt": extract_text_from_text_file,
    ".csv": extract_text_from_text_file,
    ".md": extract_text_from_text_file,
}

SUPPORTED_EXTENSIONS = set(PROCESSORS.keys())


def process_file(file_path: str) -> Dict[str, Any]:
    """Process a single file and return extracted text with metadata."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in PROCESSORS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    text = PROCESSORS[ext](file_path)

    return {
        "file_name": path.name,
        "file_path": str(path.absolute()),
        "file_type": ext,
        "text": text,
        "char_count": len(text),
    }


def process_directory(dir_path: str) -> List[Dict[str, Any]]:
    """Process all supported files in a directory."""
    results = []
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                result = process_file(str(file_path))
                results.append(result)
                print(f"  [OK] {file_path.name} ({result['char_count']} chars)")
            except Exception as e:
                print(f"  [ERR] {file_path.name}: {e}")

    return results
