"""Extract text, tables, charts and equations from PDF/image files."""

import re 
import base64
import hashlib 
from pathlib import Path 
from dataclasses import dataclass, field
from typing import Optional 
import fitz 
from PIL import Image 
import io 


@dataclass 
class ExtractedDocument: 
    document_id : str 
    source_file : str
    text : str 
    tables : list[dict]
    charts : list[dict]
    equations : list[str] 
    page_images : list[bytes] = field(default_factory=list) 

def _file_id(path : Path) -> str: 
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] 

def _extract_equations(text : str) -> list[str]: 
    """Heuristic : lines with latex-like pattern or isolated math symbols."""
    patterns = [
        r"\$[^$]+\$",                          # inline LaTeX
        r"\\\[.*?\\\]",                         # display LaTeX
        r"[A-Za-z]\s*=\s*[-+]?[\d.]+\s*[\*\/]",  # simple assignment equations
        r"∑|∫|∂|∇|√|≈|≠|≤|≥|±",              # unicode math symbols
    ]
    found = [] 
    for pat in patterns: 
        found.extend(re.findall(pat, text, re.DOTALL))
    
    return list(set(found)) 

def _parse_text_table(block_text : str) -> Optional[dict]: 
    """Detect simple pipe-separated or whitespace-aligned tables"""
    lines = [l.strip() for l in block_text.strip().splitlines() if l.strip()] 

    if(len(lines) < 2): 
        return None #you just can't have a table with just one row

    # pipe-separated 
    if all("|" in l for l in lines[:3]): 
        rows = [[c.strip() for c in l.split("|")] for l in lines if "|" in l]
        if rows: 
            return {"type" : "pipe_table", "headers" : rows[0], "rows" : rows[1:]}

    # whitespaced-aligned (>=3 columns detected)
    split_rows = [l.split() for l in lines]
    col_counts = [len(r) for r in split_rows]
    if len(set(col_counts)) <= 2 and col_counts[0] >= 2: 
        return {"type" : "space_table", "headers" : split_rows[0], "rows" : split_rows[1:]}

    return None

def extract_pdf(path:Path) -> ExtractedDocument: 
    doc = fitz.open(str(path))
    doc_id = _file_id(path)

    all_text_parts : list[str] = [] 
    tables : list[dict] = [] 
    charts : list[dict] = [] 
    page_images : list[bytes] = [] 


    for page_num, page in enumerate(doc): 
        page_text = page.get_text("text")
        all_text_parts.append(page_text)

        # attempting structured table extraction via pymupdf 
        try: 
            found_tables = page.find_tables() 
            for t in found_tables.tables(): 
                data = t.extract() 
                if data and len(data) > 1: 
                    tables.append({
                        "page" : page_num + 1, 
                        "type" : "pdf_table", 
                        "headers" : data[0], 
                        "rows" : data[1:], 
                    })

        except Exception: 
            pass
        

        # parse text blocks for table-like content 
        # blocks = PDF text grouped into rectangular chunks
        for block in page.get_text("blocks"): 
            # each block is usually a tuple like 
            # (x0, y0, x1, y1, text, block_no, block_type)

            block_text = block[4] if len(block) > 4 else ""
            tbl = _parse_text_table(block_text)

            if tbl: 
                tbl['page'] = page_num + 1
                # avoid duplicates from find_tables 
                tables.append(tbl)
        
        # extract embedded images(potential charts/figures)
        img_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            try: 
                base_img = doc.extract_image(xref)
                img_bytes = base_img['image']
                img_b64 = base64.b64encode(img_bytes).decode()
                charts.append({
                    "page" : page_num + 1, 
                    "index" : img_idx, 
                    "format" : base_img.get("ext", "png"), 
                    "description" : f"Image on page{page_num + 1}, index {img_idx}", 
                    "data_b64" : img_b64[:200] + "...", 
                })
            except Exception: 
                pass
            
        # full-page render for vision based analysis
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        page_images.append(pix.tobytes("png"))

    full_text = "\n\n".join(all_text_parts).strip()
    equations = _extract_equations(full_text)


    return ExtractedDocument(
        document_id = doc_id, 
        source_file = str(path), 
        text = full_text, 
        tables = tables, 
        charts  = charts, 
        equations = equations, 
        page_images = page_images, 
    )

def extract_image(path:Path) -> ExtractedDocument: 
    """treat a standalone image as a single-page document"""

    doc_id = _file_id(path)
    img = Image.open(path)
    buf = io.BytesIO() 
    img.save(buf, format="PNG")
    page_images = [buf.getvalue()]

    return ExtractedDocument(
        document_id = doc_id, 
        source_file = str(path), 
        text="", 
        tables = [], 
        charts = [{
            "page" : 1, 
            "index" : 0, 
            "format" : path.suffix.lstrup("."), 
            "description" : f"standalone image : {path.name}", 
            "data_b64" : base64.b64encode(buf.getvalue()).decode()[:200] + "...", 
        }], 
        equations = [], 
        page_images=page_images,
    )

def extract(file_path : str) -> ExtractedDocument: 
    path = Path(file_path)
    if not path.exists(): 
        raise FileNotFoundError(f"File not found : {file_path}")

    suffix = path.suffix.lower() 
    if suffix == ".pdf": 
        return extract_pdf(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}: 
        return extract_image(path)
    else: 
        raise ValueError(f"Unsupported file type : {suffix}")