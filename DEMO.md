# Clinical Review Assistant – Demo Guide

## The story (30 seconds)
"I started with a medical RAG chatbot grounded in a medical reference book. In real healthcare
operations, the work isn't open questions – it's reviewers (e.g. a utilization-management nurse)
processing messy document packets against payer policies. So I extended it into a Clinical Review
Assistant: OCR for scanned faxes, a classifier to triage document types, and RAG over both the
medical reference and payer policies, with low-confidence predictions routed to human review."

## Architecture – two lanes

```
LANE A  Knowledge base (offline: python store_index.py)
  Medical_book.pdf, data/policies/*  -> native text / OCR fallback -> chunk -> embed (MiniLM) -> Pinecone
  metadata: source, page, source_type (reference | policy), doc_type, extraction_method
  NOT classified – curated documents have a known type.
  Policies are kept as one chunk each so all criteria are retrieved together.

LANE B  Case documents (online: POST /upload)
  PDF / image / txt -> native text, OCR fallback (<50 chars/page) -> TF-IDF + LogReg classifier
  -> doc_type + confidence -> confidence < 0.5 => "Needs review" (reviewer confirms type)
  -> text kept in the session as case context – never written to the shared vector index.

CHAT  POST /get
  no case : question -> top-3 chunks -> GPT-4o           (original chatbot behavior)
  case    : question + case text -> policy chunks (metadata filter) + reference chunks
            -> GPT-4o with case prompt -> criterion-by-criterion answer with citations
            Never denies; recommends "route for approval" or "pend for clinical review".
```

## Setup (once)
```bash
python store_index.py --policies-only   # adds the 2 payer policies (fast)
python app.py                            # http://localhost:8080
```
(Full rebuild with new metadata on the book: `python store_index.py --reset` – slower.)

## Demo script (~5 min) – one story: acne
1. **Baseline RAG** – ask "What is acne?" → answer from Medical_book p. 38 (almost word for word).
2. **Scanned fax** – upload `samples/Acne_Case1_Clinical_Note_Scanned.pdf`
   (a dermatologist asking the insurer to cover isotretinoin / Accutane).
   → card shows OCR (no native text in the PDF), predicted *request form*, low confidence, *Needs review*.
   It's really a clinical note that contains a request → change dropdown to *clinical note* → Confirm.
   ("This is exactly why low-confidence predictions go to a human.")
3. Click **"Does this patient meet the policy criteria?"**
   → isotretinoin policy retrieved; severity, 3+ months antibiotic + topical, iPLEDGE met; pregnancy N/A (male)
   → "route for approval" [Policy: isotretinoin_acne_policy].
4. **Clear case**, upload `samples/Acne_Case2_Clinical_Note_Scanned.png` (16 y/o, moderate acne, only 2 weeks OTC gel)
   → same question → severity / prior treatment / iPLEDGE / pregnancy test not met → "pend / request more info".
5. Show `reports/` (classification report + confusion matrix) and be honest about the limits.

Backup case (knee): `samples/Knee_Case1_Clinical_Note_Scanned.pdf` (meets) and `samples/Knee_Case2_Clinical_Note_Scanned.png` (not met).

## Honest limitations → next steps
| Today | Next step |
|---|---|
| 48 synthetic docs, 100% held-out accuracy, ~30% confidence on new docs | Real de-identified docs, calibration, LLM/embedding classifier fallback |
| No eval for RAG answers | Labeled case set: retrieval hit rate, criteria accuracy, faithfulness (LLM-as-judge / RAGAS) |
| Single linear flow | LangGraph: extract → retrieve policy → check criteria → decide, with review branch |
| Tesseract OCR | VLM extraction for handwriting/tables, structured fields (ICD-10, CPT) via schema |
| In-memory case store | PHI redaction (Presidio), audit logging, encrypted storage |
| No monitoring | Latency / token cost / quality tracking (LangSmith or MLflow) |

## Likely interview questions
- *Why not classify Medical_book.pdf?* Classification is intake triage for unknown documents; curated KB docs have known provenance, so predicting their type only adds error.
- *Why not put patient notes in Pinecone?* PHI shouldn't go into a shared index; the case is session-scoped context.
- *Why is confidence low?* 4 classes → 25% is chance; tiny training set gives flat probabilities. The system handles it with a review threshold instead of trusting it blindly.
- *Why 50 chars for OCR fallback?* Prototype heuristic, configurable via `OCR_MIN_EXTRACTED_CHARS`; would tune on real scanned data.
