#!/usr/bin/env python3
"""
Multimodal RAG CLI
Usage:
  python app.py ingest <file>             # Process a PDF or image
  python app.py query  <doc_id> <query>   # Query an ingested document
  python app.py list                      # List ingested documents
"""

import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

import database
import extractor
import vector_store
import llm
import flowchart

load_dotenv()
console = Console()


def cmd_ingest(file_path: str) -> None:
    console.rule("[bold cyan]Ingesting Document")

    console.print(f"[yellow]Extracting:[/] {file_path}")
    doc = extractor.extract(file_path)

    console.print(f"[green]✓[/] Extracted {len(doc.text)} chars of text")
    console.print(f"[green]✓[/] Found {len(doc.tables)} tables, {len(doc.charts)} charts, {len(doc.equations)} equations")

    console.print("[yellow]Storing in SQLite...[/]")
    database.init_db()
    database.upsert_document(
        document_id=doc.document_id,
        source_file=doc.source_file,
        extracted_text=doc.text,
        tables=doc.tables,
        charts=doc.charts,
        equations=doc.equations,
    )
    console.print("[green]✓[/] SQLite storage complete")

    console.print("[yellow]Generating embeddings and indexing in ChromaDB...[/]")
    n_chunks = vector_store.index_document(
        document_id=doc.document_id,
        text=doc.text,
        tables=doc.tables,
        charts=doc.charts,
        equations=doc.equations,
    )
    console.print(f"[green]✓[/] Indexed {n_chunks} chunks in vector store")

    console.print(Panel(
        f"[bold]Document ID:[/] {doc.document_id}\n"
        f"[bold]Source:[/] {doc.source_file}\n"
        f"[bold]Text length:[/] {len(doc.text):,} chars\n"
        f"[bold]Tables:[/] {len(doc.tables)}\n"
        f"[bold]Charts:[/] {len(doc.charts)}\n"
        f"[bold]Equations:[/] {len(doc.equations)}\n"
        f"[bold]Chunks indexed:[/] {n_chunks}",
        title="[bold green]Ingestion Complete",
    ))


def cmd_query(document_id: str, query: str) -> None:
    console.rule("[bold cyan]Querying Document")

    database.init_db()
    row = database.get_document(document_id)
    if not row:
        console.print(f"[red]Error:[/] Document {document_id} not found. Run `ingest` first.")
        sys.exit(1)

    doc_meta = {"source_file": row["source_file"]}

    console.print(f"[yellow]Retrieving relevant chunks for:[/] {query}")
    chunks = vector_store.retrieve(query=query, document_id=document_id, top_k=6)

    if not chunks:
        console.print("[red]No chunks retrieved. The document may not be indexed.[/]")
        sys.exit(1)

    tbl = Table(title="Retrieved Chunks", show_lines=True)
    tbl.add_column("Modality", style="cyan", width=10)
    tbl.add_column("Score", style="green", width=8)
    tbl.add_column("Preview", width=60)
    for c in chunks:
        tbl.add_row(c["modality"], str(c["score"]), c["text"][:100] + "...")
    console.print(tbl)

    console.print("[yellow]Generating explanation via LLM...[/]")
    explanation = llm.explain_document(query=query, retrieved_chunks=chunks, doc_metadata=doc_meta)
    console.print(Panel(explanation, title="[bold green]Explanation"))

    console.print("[yellow]Generating Mermaid flowchart...[/]")
    mermaid_spec = llm.generate_mermaid_spec(query=query, explanation=explanation, retrieved_chunks=chunks)

    chart_path = flowchart.save_flowchart(mermaid_spec, document_id)
    ascii_preview = flowchart.render_ascii_preview(mermaid_spec)

    console.print(Panel(
        f"[bold]Mermaid spec saved to:[/] {chart_path}\n\n"
        f"[bold]Flow preview:[/]\n{ascii_preview}\n\n"
        f"[bold]Raw spec (first 600 chars):[/]\n[dim]{mermaid_spec[:600]}[/]",
        title="[bold green]Flowchart Output",
    ))

    database.log_comparison(document_id=document_id, query=query, response=explanation)
    console.print("[green]✓[/] Logged to comparison history")


def cmd_list() -> None:
    database.init_db()
    rows = database.list_documents()
    if not rows:
        console.print("[yellow]No documents ingested yet.[/]")
        return

    tbl = Table(title="Ingested Documents")
    tbl.add_column("Document ID", style="cyan")
    tbl.add_column("Source File", style="white")
    tbl.add_column("Ingested At", style="dim")
    for r in rows:
        tbl.add_row(r["document_id"], r["source_file"], r["created_at"])
    console.print(tbl)


def main():
    if len(sys.argv) < 2:
        console.print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        if len(sys.argv) < 3:
            console.print("[red]Usage:[/] python app.py ingest <file>")
            sys.exit(1)
        cmd_ingest(sys.argv[2])

    elif cmd == "query":
        if len(sys.argv) < 4:
            console.print("[red]Usage:[/] python app.py query <doc_id> <query>")
            sys.exit(1)
        cmd_query(sys.argv[2], " ".join(sys.argv[3:]))

    elif cmd == "list":
        cmd_list()

    else:
        console.print(f"[red]Unknown command:[/] {cmd}")
        console.print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()