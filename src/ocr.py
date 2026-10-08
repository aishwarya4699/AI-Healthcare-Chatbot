import logging
import os
from pathlib import Path
from typing import List

from langchain.schema import Document

from src.ingestion import IMAGE_EXTENSIONS

# Logger used to print a warning if OCR fails
logger = logging.getLogger(__name__)

# A PDF page with fewer than 50 characters is treated as a scan and gets OCR.
# Prototype rule of thumb; can be changed with OCR_MIN_EXTRACTED_CHARS.
DEFAULT_MIN_EXTRACTED_CHARS = 50


# Read the threshold from the environment, or use 50 by default
def get_min_extracted_chars() -> int:
    return int(os.environ.get("OCR_MIN_EXTRACTED_CHARS", DEFAULT_MIN_EXTRACTED_CHARS))


def apply_ocr_fallback(docs: List[Document]) -> List[Document]:
    """Keep digital PDF text when it is long enough; otherwise OCR that page.

    Images are always sent through OCR.
    """
    # Get the threshold (50) and prepare the result list
    threshold = get_min_extracted_chars()
    processed: List[Document] = []
    for doc in docs:
        # Get the file path, page number, file type and any text already found
        source = str(doc.metadata.get("source", ""))
        page = int(doc.metadata.get("page", 1))
        suffix = Path(source).suffix.lower()
        native_text = (doc.page_content or "").strip()

        # Image: always run OCR (fall back to existing text if OCR finds nothing)
        if suffix in IMAGE_EXTENSIONS:
            text = _ocr_image(source) or native_text
            ocr_used = True
        # Scanned PDF page (fewer than 50 chars): run OCR on just this page
        elif suffix == ".pdf" and len(native_text) < threshold:
            ocr_text = _ocr_pdf_page(source, page)
            text = ocr_text if ocr_text else native_text
            ocr_used = True
        # Normal text page: keep the text as it is, no OCR
        else:
            text = doc.page_content or ""
            ocr_used = False

        # Copy the old info and record whether OCR was used (shown as a badge)
        metadata = {
            **doc.metadata,
            "source": source,
            "page": page,
            "ocr_used": ocr_used,
        }
        processed.append(Document(page_content=text, metadata=metadata))
    return processed


# Read text from an image with Tesseract; return "" instead of crashing on errors
def _ocr_image(path: str) -> str:
    try:
        import pytesseract
        from PIL import Image

        # Open the image and let Tesseract read the text
        with Image.open(path) as image:
            return pytesseract.image_to_string(image) or ""
    except Exception as exc:
        logger.warning("OCR failed for image %s: %s", path, exc)
        return ""


# Read text from one scanned PDF page; return "" instead of crashing on errors
def _ocr_pdf_page(path: str, page: int) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path

        # Turn only this one page into a sharp image (300 dpi)
        images = convert_from_path(
            path,
            dpi=300,
            first_page=page,
            last_page=page,
        )
        if not images:
            return ""
        # Let Tesseract read the text from that image
        return pytesseract.image_to_string(images[0]) or ""
    except Exception as exc:
        logger.warning("OCR failed for PDF %s page %s: %s", path, page, exc)
        return ""
