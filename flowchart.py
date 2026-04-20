"""Save and display Mermaid flowcharts."""

from pathlib import Path
from datetime import datetime


def save_flowchart(mermaid_spec: str, document_id: str, output_dir: str = "./output") -> str:
    """Save Mermaid spec to .mmd file and return path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"flowchart_{document_id[:8]}_{ts}.mmd"
    filepath = out / filename

    filepath.write_text(mermaid_spec, encoding="utf-8")
    return str(filepath)


def render_ascii_preview(mermaid_spec: str) -> str:
    """Return a minimal ASCII representation of the flowchart nodes."""
    lines = []
    for line in mermaid_spec.splitlines():
        line = line.strip()
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                src = parts[0].strip().split("[")[0].split("(")[0].strip()
                dst = parts[1].strip().split("[")[0].split("(")[0].strip()
                # Extract label from brackets
                label = ""
                if "|" in parts[1]:
                    label_part = parts[1].split("|")
                    if len(label_part) >= 2:
                        label = f" [{label_part[1].strip()}]"
                lines.append(f"  {src} ──{label}──► {dst}")
    return "\n".join(lines) if lines else "(no edges parsed)"