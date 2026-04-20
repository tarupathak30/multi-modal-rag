"""LLM reasoning via Groq API."""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv() 


_MODEL = "openai/gpt-oss-20b"  # Best OSS 70B on Groq; fallback below
_FALLBACK_MODEL = "openai/gpt-oss-120b"


def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment.")
    return Groq(api_key=api_key)


def _chat(messages: list[dict], model: str = _MODEL) -> str:
    client = _client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        if "model" in str(e).lower() and model != _FALLBACK_MODEL:
            return _chat(messages, model=_FALLBACK_MODEL)
        raise


def explain_document(
    query: str,
    retrieved_chunks: list[dict],
    doc_metadata: dict,
) -> str:
    """Generate a structured explanation from retrieved context."""

    context_parts = []
    for chunk in retrieved_chunks:
        tag = chunk["modality"].upper()
        context_parts.append(f"[{tag}] {chunk['text']}")

    context = "\n\n".join(context_parts)

    system = (
        "You are a technical document analyst. Given retrieved context from a multimodal document "
        "(text, tables, charts, equations), provide a clear, structured explanation. "
        "Be concise. Use bullet points where appropriate. Reference modality types when relevant."
    )

    user = (
        f"Document: {doc_metadata.get('source_file', 'unknown')}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Query: {query}\n\n"
        "Provide a clear explanation addressing the query based on the retrieved context."
    )

    return _chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


def generate_mermaid_spec(
    query: str,
    explanation: str,
    retrieved_chunks: list[dict],
) -> str:
    """Ask LLM to produce a Mermaid flowchart spec from the analysis."""

    modalities = list({c["modality"] for c in retrieved_chunks})

    system = (
        "You are a technical diagram generator. Output ONLY valid Mermaid flowchart syntax. "
        "No explanations, no markdown fences, no extra text. Start with 'flowchart TD'."
    )

    user = (
        f"Query: {query}\n\n"
        f"Analysis summary: {explanation[:800]}\n\n"
        f"Modalities involved: {', '.join(modalities)}\n\n"
        "Generate a Mermaid flowchart (flowchart TD) showing the document processing pipeline "
        "and key concepts found. Keep it to 8-12 nodes."
    )

    raw = _chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    # Strip accidental fences
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    if not raw.startswith("flowchart"):
        raw = "flowchart TD\n" + raw

    return raw