import logging
import os
from pathlib import Path
from typing import List

from langchain.schema import Document

from src.ingestion import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# Prototype heuristic, not a clinical quality gate.
# Override with OCR_MIN_EXTRACTED_CHARS. Tune later with evaluation data.
DEFAULT_MIN_EXTRACTED_CHARS = 50


def get_min_extracted_chars() -> int:
    return int(os.environ.get("OCR_MIN_EXTRACTED_CHARS", DEFAULT_MIN_EXTRACTED_CHARS))


def apply_ocr_fallback(docs: List[Document]) -> List[Document]:
    """Keep digital PDF text when it is long enough; otherwise OCR that page.

    Images are always sent through OCR.
    """
    threshold = get_min_extracted_chars()
    processed: List[Document] = []
    for doc in docs:
        source = str(doc.metadata.get("source", ""))
        page = int(doc.metadata.get("page", 1))
        suffix = Path(source).suffix.lower()
        native_text = (doc.page_content or "").strip()

        if suffix in IMAGE_EXTENSIONS:
            text = _ocr_image(source) or native_text
            ocr_used = True
        elif suffix == ".pdf" and len(native_text) < threshold:
            ocr_text = _ocr_pdf_page(source, page)
            text = ocr_text if ocr_text else native_text
            ocr_used = True
        else:
            text = doc.page_content or ""
            ocr_used = False

        metadata = {
            **doc.metadata,
            "source": source,
            "page": page,
            "ocr_used": ocr_used,
        }
        processed.append(Document(page_content=text, metadata=metadata))
    return processed


def _ocr_image(path: str) -> str:
    try:
        import pytesseract
        from PIL import Image

        with Image.open(path) as image:
            return pytesseract.image_to_string(image) or ""
    except Exception as exc:
        logger.warning("OCR failed for image %s: %s", path, exc)
        return ""


def _ocr_pdf_page(path: str, page: int) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(
            path,
            dpi=300,
            first_page=page,
            last_page=page,
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0]) or ""
    except Exception as exc:
        logger.warning("OCR failed for PDF %s page %s: %s", path, page, exc)
        return ""
