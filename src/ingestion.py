from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_documents(data_dir: str = "data/") -> List[Document]:
    """Load PDFs (native text per page) and image files from data_dir.

    Image page_content is left empty; OCR fills it in src.ocr.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Ingestion directory not found: {root}")

    documents: List[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(_load_pdf(path))
        elif suffix in IMAGE_EXTENSIONS:
            documents.append(
                Document(
                    page_content="",
                    metadata={"source": str(path), "page": 1},
                )
            )
    return documents


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
