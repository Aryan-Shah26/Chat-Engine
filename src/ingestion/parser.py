import re
from pathlib import Path

import fitz
from bs4 import BeautifulSoup

IMAGE_DIR = "data/extracted_images"


def clean_extracted_text(text: str) -> str:
    """Normalizes glyphs, private-use bullets, whitespace, and line breaks from extracted text."""
    if not text:
        return ""
    # Normalize private-use bullets and common bullet glyphs
    text = re.sub(r"[\uf0b7\uf0a7\u2022\u25aa\u25cf\u25e6\u2043\u2219]", "• ", text)
    # Normalize non-breaking spaces and zero-width spaces
    text = re.sub(r"[\u202f\xa0\u200b\u200e\u200f\ufeff]", " ", text)
    # Fix hyphenated words broken across lines
    text = re.sub(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)", r"\1\2", text)
    # Normalize multiple whitespace within lines while preserving linebreaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Normalize multiple empty lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    doc = None
    try:
        pages = []
        doc = fitz.open(file_path)
        for page in doc:
            try:
                raw_text = page.get_text()
                text = clean_extracted_text(raw_text)
            except Exception:
                # Skip pages that fail text extraction (e.g. corrupt graphic streams)
                continue
            if not text:
                continue
            pages.append({"text": text, "metadata": {"source": file_path.name, "page": page.number + 1}})
        return pages
    except Exception as e:
        raise ValueError(f"Error parsing '{file_path.name}': {e}")
    finally:
        if doc is not None:
            doc.close()


def extract_tables(file_path: str | Path) -> list[dict]:
    """Extracts tables as markdown chunks tagged content_type=table, for the retrieval index."""
    file_path = Path(file_path)
    doc = None
    try:
        doc = fitz.open(file_path)
        tables = []
        for page in doc:
            try:
                found = page.find_tables()
                for i, table in enumerate(found.tables):
                    markdown = table.to_markdown()
                    cleaned_md = clean_extracted_text(markdown)
                    if not cleaned_md:
                        continue
                    tables.append({
                        "text": cleaned_md,
                        "metadata": {
                            "source": file_path.name, "page": page.number + 1,
                            "content_type": "table", "table_index": i,
                        },
                    })
            except Exception:
                # Skip pages where table extraction fails (e.g. complex vector graphics)
                continue
        return tables
    except Exception as e:
        raise ValueError(f"Error extracting tables from '{file_path.name}': {e}")
    finally:
        if doc is not None:
            doc.close()


def extract_images(file_path: str | Path) -> list[dict]:
    """
    Extracts embedded images to disk and returns pointer metadata. NOT wired
    into the retrieval index yet — no captioning/OCR model is in the loop,
    so an image extracted here has no searchable text representation.
    """
    file_path = Path(file_path)
    out_dir = Path(IMAGE_DIR) / file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(file_path)
    images = []
    try:
        for page in doc:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base = doc.extract_image(xref)
                out_path = out_dir / f"page{page.number + 1}_img{img_index}.{base['ext']}"
                out_path.write_bytes(base["image"])
                images.append({"path": str(out_path), "source": file_path.name, "page": page.number + 1})
        return images
    finally:
        doc.close()


def parse_html(file_path: str | Path) -> list[dict]:
    try:
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            raw_text = soup.get_text(separator="\n", strip=True)
            text = clean_extracted_text(raw_text)
        return [{"text": text, "metadata": {"source": file_path.name, "page": 1}}]
    except Exception as e:
        raise ValueError(f"Error parsing '{Path(file_path).name}': {e}")


def parse_text(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = clean_extracted_text(raw_text)
    return [{"text": text, "metadata": {"source": file_path.name, "page": 1}}]


def parse_file(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix == ".html":
        return parse_html(file_path)
    elif suffix in {".txt", ".md"}:
        return parse_text(file_path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

