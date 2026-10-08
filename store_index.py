"""Build the knowledge base (Lane A) in Pinecone.

Knowledge-base documents are curated, so they are NOT classified:
  - data/Medical_book.pdf      -> source_type="reference", doc_type="medical_reference"
  - data/policies/*.txt|*.pdf  -> source_type="policy",    doc_type="policy_document"

Usage:
  python store_index.py --policies-only   # fast: add/refresh payer policies only
  python store_index.py --reset           # full rebuild: wipe index, re-index everything
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import download_hugging_face_embeddings, text_split
from src.pipeline import prepare_reference_documents

load_dotenv()

INDEX_NAME = "medical-chatbot"
REFERENCE_FILES = [Path("data/Medical_book.pdf")]
POLICY_DIR = Path("data/policies")


def chunk_ids(chunks, prefix):
    """Deterministic IDs so re-running does not create duplicate vectors."""
    counters = {}
    ids = []
    for chunk in chunks:
        key = (chunk.metadata["source"], chunk.metadata.get("page", 1))
        counters[key] = counters.get(key, 0) + 1
        ids.append(f"{prefix}::{key[0]}::p{key[1]}::c{counters[key]}")
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies-only", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(INDEX_NAME)
    if args.reset:
        print("Deleting all existing vectors...")
        index.delete(delete_all=True)

    embeddings = download_hugging_face_embeddings()
    store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Payer policies
    policy_docs = []
    for path in sorted(POLICY_DIR.glob("*")):
        if path.suffix.lower() in {".txt", ".pdf"}:
            policy_docs += prepare_reference_documents(str(path), "policy", "policy_document")
    # Policies are short and every criterion matters, so keep each policy whole
    # (one chunk) instead of splitting at 500 chars like the reference book.
    policy_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    policy_chunks = policy_splitter.split_documents(policy_docs)
    store.add_documents(policy_chunks, ids=chunk_ids(policy_chunks, "policy"))
    print(f"Indexed {len(policy_chunks)} policy chunks from {POLICY_DIR}/")

    if args.policies_only:
        return

    # Medical reference book(s)
    for path in REFERENCE_FILES:
        docs = prepare_reference_documents(str(path), "reference", "medical_reference")
        ocr_pages = sum(d.metadata["extraction_method"] == "ocr" for d in docs)
        chunks = text_split(docs)
        store.add_documents(chunks, ids=chunk_ids(chunks, "reference"))
        print(f"Indexed {len(chunks)} chunks from {path.name} ({ocr_pages} pages via OCR)")


if __name__ == "__main__":
    main()
