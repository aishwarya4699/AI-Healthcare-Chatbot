import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from werkzeug.utils import secure_filename

from src.classification import DOCUMENT_CLASSES
from src.helper import download_hugging_face_embeddings
from src.ingestion import SUPPORTED_EXTENSIONS
from src.pipeline import process_upload
from src.prompt import case_system_prompt, system_prompt

# Load API keys from the .env file
load_dotenv()

# Create the Flask web app
app = Flask(__name__)
# Secret key used to sign the browser session cookie
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads

# Temporary folder for uploaded files (each file is deleted right after OCR)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Demo-only in-memory case store, keyed by browser session.
# Patient documents are NOT written to Pinecone (shared index) - they live
# only for the session. Production would use an encrypted, access-controlled store.
CASES = {}

# Load MiniLM and connect to the existing Pinecone index
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot", embedding=embeddings
)

# Reference retriever: medical book only (top 3 chunks), like the original chatbot.
reference_retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3, "filter": {"source_type": "reference"}}
)
# Policy retriever: the single best-matching payer policy (metadata filter), used when a case is open.
policy_retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 1, "filter": {"source_type": "policy"}}
)

# GPT-4o with temperature 0 so answers are consistent
chatModel = ChatOpenAI(model="gpt-4o", temperature=0)


# Make a readable source name, e.g. "Policy: isotretinoin_acne_policy" or "Medical_book.pdf p. 38"
def source_label(doc):
    meta = doc.metadata
    source = Path(str(meta.get("source", "unknown"))).name
    if meta.get("source_type") == "policy":
        return f"Policy: {Path(source).stem}"
    page = meta.get("page")
    return f"{source} p. {int(page)}" if page is not None else source


# Join the retrieved chunks into one context text (skip duplicates) and list their sources
def build_context(docs):
    seen, parts, sources = set(), [], []
    for doc in docs:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        label = source_label(doc)
        parts.append(f"[{label}]\n{doc.page_content}")
        if label not in sources:
            sources.append(label)
    return "\n\n---\n\n".join(parts), sources


# Get this browser's open patient case (or None if no file is uploaded)
def current_case():
    return CASES.get(session.get("case_id"))


# Show the chat page
@app.route("/")
def index():
    return render_template("chat.html", doc_classes=DOCUMENT_CLASSES)


# Answer a chat question
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    case = current_case()
    try:
        if case is None:
            # Original RAG behavior: question -> top-3 chunks -> GPT-4o.
            docs = reference_retriever.invoke(msg)
            context, sources = build_context(docs)
            system = system_prompt.replace("{context}", context)
        else:
            # Case mode: retrieve using the question + the case text, from both
            # the medical reference and the payer policies.
            query = f"{msg}\n\n{case['text'][:1500]}"
            docs = policy_retriever.invoke(query) + reference_retriever.invoke(query)
            context, sources = build_context(docs)
            # Fill the review prompt with the document type, patient text and retrieved context
            system = (
                case_system_prompt.replace("{doc_type}", case["doc_type"] or "unknown")
                .replace("{case_text}", case["text"])
                .replace("{context}", context)
            )
        # Ask GPT-4o and send the answer and sources back to the page
        answer = chatModel.invoke([SystemMessage(content=system), HumanMessage(content=msg)]).content
        return jsonify(answer=answer, sources=sources, case_mode=case is not None)
    except Exception as exc:
        # Log the error and return a clear message instead of crashing
        app.logger.exception("Chat failed")
        return jsonify(error=str(exc)), 500


# Handle a patient document upload
@app.route("/upload", methods=["POST"])
def upload():
    # Check a file was sent
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="No file uploaded."), 400
    # Clean the file name and allow only supported types
    filename = secure_filename(file.filename)
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return jsonify(error="Unsupported file type. Use PDF, PNG, JPG, TIFF or TXT."), 400

    # Save it temporarily with a random prefix so names never clash
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    file.save(path)
    try:
        # Read (OCR if needed) and classify the document
        result = process_upload(str(path))
    except Exception as exc:
        app.logger.exception("Upload processing failed")
        return jsonify(error=f"Could not process document: {exc}"), 500
    finally:
        path.unlink(missing_ok=True)  # don't keep raw patient files on disk

    # Store the case in memory and link it to this browser's session
    case_id = uuid.uuid4().hex
    CASES[case_id] = {
        "filename": filename,
        "text": result["text"],
        "doc_type": result["doc_type"],
        "confidence": result["confidence"],
        "confirmed": False,
    }
    session["case_id"] = case_id

    # Send the card info back to the page (without the full text)
    response = {k: v for k, v in result.items() if k != "text"}
    response["filename"] = filename
    return jsonify(response)


@app.route("/confirm_type", methods=["POST"])
def confirm_type():
    """Human-in-the-loop: reviewer confirms or corrects the predicted type."""
    case = current_case()
    doc_type = request.form.get("doc_type")
    # Reject if there's no open case or the type isn't one of the 4 allowed types
    if case is None or doc_type not in DOCUMENT_CLASSES:
        return jsonify(error="No active case or invalid type."), 400
    case["doc_type"] = doc_type
    case["confirmed"] = True
    return jsonify(doc_type=doc_type, confirmed=True)


# Clear the open case (the "Clear case" button, and on every page load)
@app.route("/reset", methods=["POST"])
def reset():
    CASES.pop(session.pop("case_id", None), None)
    return jsonify(ok=True)


# Start the web server on port 8080
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
