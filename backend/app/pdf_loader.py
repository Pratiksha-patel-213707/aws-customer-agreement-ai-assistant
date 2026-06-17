from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = _extract_with_pypdf(pdf_path)
    if sum(len(page["text"].strip()) for page in pages) > 100:
        return pages
    return _extract_with_ocr(pdf_path)


def _extract_with_pypdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    return [
        {"page": page_number, "text": (page.extract_text() or "").strip()}
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def _extract_with_ocr(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages: list[dict] = []
    for index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(image)
        pages.append({"page": index, "text": text.strip()})
    return pages
