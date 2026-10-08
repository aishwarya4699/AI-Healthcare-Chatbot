from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt"}
SUPPORTED_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS | TEXT_EXTENSIONS


def load_file(file_path) -> List[Document]:
    """Load a single file into page-level Documents.

    - PDF: native text per page (OCR fallback is applied later in src.ocr)
    - Image: empty page_content (always OCR'd in src.ocr)
    - TXT: text as-is (already digital)
    """
    # Get the file path and its extension (lowercase)
    path = Path(file_path)
    suffix = path.suffix.lower()

    # PDF: read text page by page (scanned pages get OCR later)
    if suffix == ".pdf":
        return _load_pdf(path)

    # Image: return an empty page with the file path; ocr.py fills in the text
    if suffix in IMAGE_EXTENSIONS:
        return [Document(page_content="", metadata={"source": str(path), "page": 1})]

    # Text file: read the text directly
    if suffix in TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": str(path), "page": 1})]

    # Any other file type is not supported
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> List[Document]:
    pages = PyPDFLoader(str(path)).load()
    loaded: List[Document] = []
    for doc in pages:
        # PyPDFLoader uses 0-based page indexes; store 1-based page numbers.
        page_index = int(doc.metadata.get("page", 0))
        loaded.append(
            Document(
                page_content=doc.page_content or "",
                metadata={"source": str(path), "page": page_index + 1},
            )
        )
    return loaded
