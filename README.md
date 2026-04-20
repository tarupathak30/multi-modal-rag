# Multimodal RAG

Local CLI pipeline that ingests PDFs and images, extracts text, tables, charts, and equations, indexes them in a vector store, and answers queries with LLM-generated explanations and Mermaid flowcharts.

Tested on *Attention Is All You Need* (1706.03762) — correctly retrieved BLEU 28.4 and reconstructed the full attention formula.

## Results

| Query | Result | Similarity |
|---|---|---|
| BLEU score on WMT 2014 English-German? | 28.4 ✓ | 0.8766 |
| Scaled dot-product attention formula? | Full LaTeX reconstructed ✓ | 0.9385 |
| What does this paper tell about? | Transformer, attention-only arch, benchmarks ✓ | 0.8225 |

Ingested 1706.03762v7.pdf → 39,525 chars · 56 tables · 3 charts · 74 chunks indexed.

## Stack

| | |
|---|---|
| **pymupdf** | PDF/image extraction |
| **sentence-transformers** | Embeddings (BAAI/bge-large-en) |
| **chromadb** | Vector store |
| **groq** | openai/gpt-oss-20b inference |
| **sqlite3** | Structured storage |
| **rich** | Terminal UI |

## Setup

```bash
python -m venv venv && venv\Scripts\activate   # Windows
# source venv/bin/activate                      # macOS/Linux

pip install -r requirements.txt

cp .env.example .env   # add GROQ_API_KEY
```

Get a free Groq key at [groq.com](https://groq.com).

## Usage

```bash
# Ingest
python app.py ingest yourfile.pdf

# Query (use the doc_id printed after ingest)
python app.py query <doc_id> "your question"

# List documents
python app.py list
```

## Pipeline

```
Extract → Normalize → SQLite → Embed → Retrieve → Reason → Flowchart → Log
```

1. **Extract** — PyMuPDF parses text, tables, images, and equations
2. **Embed** — chunks encoded and stored in ChromaDB
3. **Retrieve** — top-6 chunks by cosine similarity
4. **Reason** — openai/gpt-oss-20b generates a structured explanation
5. **Flowchart** — second LLM call produces a Mermaid diagram saved to `output/`

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Required. [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | Optional. Suppresses HuggingFace rate-limit warnings |

## Known Limitations

- Equation extraction is heuristic — LaTeX rendered as PDF glyphs is not captured
- Chart understanding is metadata-only (no vision LLM)
- No conversation memory between queries
- Embedding model reloads on every invocation (~2s startup)

## Planned

- Vision-based chart understanding via LLaVA / GPT-4o
- Multi-document comparison CLI command
- Incremental re-indexing (page-level content hashing)