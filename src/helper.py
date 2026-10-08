from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep page_content plus indexing metadata used by the Document AI pipeline."""
    kept_keys = ("source", "page", "ocr_used", "doc_type", "doc_type_confidence")
    minimal_docs: List[Document] = []
    for doc in docs:
        metadata = {key: doc.metadata[key] for key in kept_keys if key in doc.metadata}
        if "source" not in metadata:
            metadata["source"] = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings