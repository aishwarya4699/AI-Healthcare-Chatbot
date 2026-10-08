from src.classification import classify_documents
from src.helper import filter_to_minimal_docs
from src.ingestion import load_documents
from src.ocr import apply_ocr_fallback


def prepare_documents(data_dir: str = "data/"):
    """Ingest, OCR-fallback, and classify pages before chunking/embeddings/RAG."""
    docs = load_documents(data_dir)
    docs = apply_ocr_fallback(docs)
    docs = classify_documents(docs)
    return filter_to_minimal_docs(docs)
