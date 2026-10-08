"""Two lanes through the Document AI pipeline.

Lane A - Knowledge base (offline, store_index.py):
    trusted reference documents (Medical_book.pdf, payer policies)
    -> native text / OCR fallback -> chunk -> embed -> Pinecone
    No classifier: their type is known when we curate them.

Lane B - Case documents (online, /upload):
    unknown inbound documents (faxes, notes, forms)
    -> native text / OCR fallback -> classifier -> doc_type + confidence
    -> low confidence is flagged for human review
    -> text is used as case context for the chat (never written to Pinecone)
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from langchain.schema import Document

from src.classification import load_classifier, predict_page
from src.ingestion import load_file
from src.ocr import apply_ocr_fallback

# Below this confidence, a reviewer must confirm the document type.
REVIEW_CONFIDENCE_THRESHOLD = float(os.environ.get("REVIEW_CONFIDENCE_THRESHOLD", 0.5))


@lru_cache(maxsize=1)
def get_classifier():
    return load_classifier()


def _extraction_method(pages: List[Document]) -> str:
    ocr_flags = [bool(p.metadata.get("ocr_used")) for p in pages]
    if ocr_flags and all(ocr_flags):
        return "ocr"
    if any(ocr_flags):
        return "mixed"
    return "native"


# ---------- Lane A: knowledge base ----------
def prepare_reference_documents(
    file_path: str, source_type: str, doc_type: str
) -> List[Document]:
    """Extract a trusted knowledge-base file. Type is assigned, not predicted."""
    pages = apply_ocr_fallback(load_file(file_path))
    docs: List[Document] = []
    for page in pages:
        if not page.page_content.strip():
            continue
        docs.append(
            Document(
                page_content=page.page_content,
                metadata={
                    "source": Path(file_path).name,
                    "page": int(page.metadata.get("page", 1)),
                    "source_type": source_type,  # "reference" or "policy"
                    "doc_type": doc_type,
                    "extraction_method": "ocr" if page.metadata.get("ocr_used") else "native",
                },
            )
        )
    return docs


# ---------- Lane B: uploaded case documents ----------
def process_upload(file_path: str) -> Dict:
    """Extract text from an uploaded case document and classify it."""
    pages = apply_ocr_fallback(load_file(file_path))
    text = "\n\n".join(p.page_content.strip() for p in pages if p.page_content.strip())

    result = {
        "pages": len(pages),
        "extraction_method": _extraction_method(pages),
        "char_count": len(text),
        "text": text,
        "preview": text[:600],
    }

    if not text.strip():
        result.update(
            doc_type=None,
            confidence=0.0,
            needs_review=True,
            warning="No text could be extracted from this document.",
        )
        return result

    label, confidence = predict_page(get_classifier(), text)
    result.update(
        doc_type=label,
        confidence=round(confidence, 3),
        needs_review=confidence < REVIEW_CONFIDENCE_THRESHOLD,
    )
    return result
