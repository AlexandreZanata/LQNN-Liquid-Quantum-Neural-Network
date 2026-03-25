"""Content extractors for different source types.

Supports:
- PDF files (text + embedded images) -- page-by-page for large files
- Plain text / markdown / CSV / RST
- DOCX (via python-docx)
- HTML files
- JSON files
- URLs (via aiohttp + BeautifulSoup)
- Images (JPEG, PNG, WEBP, GIF)
"""

from __future__ import annotations

import io
import json as json_lib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class ExtractedContent:
    text: str = ""
    images: list[bytes] = field(default_factory=list)
    image_captions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: str = ""


def extract_pdf(data: bytes, filename: str = "document.pdf") -> ExtractedContent:
    """Extract text and images from a PDF, page-by-page for large files."""
    if len(data) > MAX_PDF_BYTES:
        return ExtractedContent(
            error=f"PDF too large ({len(data) / 1024 / 1024:.1f}MB > 50MB limit)",
            metadata={"filename": filename},
        )

    text_parts: list[str] = []
    images: list[bytes] = []
    captions: list[str] = []
    meta: dict = {"filename": filename, "pages": 0, "size_bytes": len(data)}

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            meta["pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"[Page {i+1}]\n{page_text}")

                try:
                    for img_obj in page.images:
                        if hasattr(img_obj, "data"):
                            images.append(img_obj.data)
                            captions.append(f"{filename} page {i+1}")
                except Exception:
                    pass

        if text_parts:
            return ExtractedContent(
                text="\n\n".join(text_parts),
                images=images,
                image_captions=captions,
                metadata=meta,
            )
    except ImportError:
        log.debug("pdfplumber not available, trying pypdf")
    except Exception as e:
        log.warning("pdfplumber failed: %s", e)

    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        meta["pages"] = len(reader.pages)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"[Page {i+1}]\n{page_text}")

        for page in reader.pages:
            if hasattr(page, "images"):
                for img_obj in page.images:
                    try:
                        images.append(img_obj.data)
                        captions.append(filename)
                    except Exception:
                        pass

        return ExtractedContent(
            text="\n\n".join(text_parts),
            images=images,
            image_captions=captions,
            metadata=meta,
        )
    except ImportError:
        log.warning("Neither pdfplumber nor pypdf installed.")
    except Exception as e:
        log.error("PDF extraction failed: %s", e)

    return ExtractedContent(
        error="No PDF library available. Install pdfplumber.",
        metadata=meta,
    )


def extract_text(data: bytes, filename: str = "document.txt") -> ExtractedContent:
    """Extract text from plain text / markdown / CSV / RST files."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    if ext in ("html", "htm"):
        return extract_html(data, filename)
    if ext == "json":
        return extract_json(data, filename)
    if ext in ("docx", "doc"):
        return extract_docx(data, filename)

    try:
        text = data.decode("utf-8", errors="replace")
        text = re.sub(r"#{1,6}\s+", "", text)
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        return ExtractedContent(
            text=text,
            metadata={"filename": filename, "chars": len(text)},
        )
    except Exception as e:
        return ExtractedContent(error=str(e))


def extract_html(data: bytes, filename: str = "document.html") -> ExtractedContent:
    """Extract text from an HTML file."""
    try:
        from bs4 import BeautifulSoup
        html_text = data.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html_text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else filename

        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup
        )

        text_parts = []
        for tag in main.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                                   "li", "blockquote", "pre", "td", "th"]):
            t = tag.get_text(separator=" ", strip=True)
            if len(t) > 20:
                text_parts.append(t)

        text = "\n\n".join(text_parts)
        return ExtractedContent(
            text=text,
            metadata={"filename": filename, "title": title, "chars": len(text)},
        )
    except Exception as e:
        return ExtractedContent(error=f"HTML extraction failed: {e}")


def extract_json(data: bytes, filename: str = "document.json") -> ExtractedContent:
    """Extract text content from a JSON file."""
    try:
        text = data.decode("utf-8", errors="replace")
        parsed = json_lib.loads(text)

        def flatten(obj, prefix=""):
            parts = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    parts.extend(flatten(v, f"{prefix}{k}: "))
            elif isinstance(obj, list):
                for item in obj:
                    parts.extend(flatten(item, prefix))
            elif isinstance(obj, str) and len(obj) > 10:
                parts.append(f"{prefix}{obj}")
            return parts

        text_parts = flatten(parsed)
        combined = "\n".join(text_parts)

        return ExtractedContent(
            text=combined,
            metadata={"filename": filename, "chars": len(combined),
                       "type": "json"},
        )
    except Exception as e:
        return ExtractedContent(error=f"JSON extraction failed: {e}")


def extract_docx(data: bytes, filename: str = "document.docx") -> ExtractedContent:
    """Extract text from a DOCX file."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(data))
        text_parts = []
        for para in doc.paragraphs:
            t = para.text.strip()
            if t:
                text_parts.append(t)

        images = []
        captions = []
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    img_data = rel.target_part.blob
                    if len(img_data) > 1000:
                        images.append(img_data)
                        captions.append(filename)
                except Exception:
                    pass

        text = "\n\n".join(text_parts)
        return ExtractedContent(
            text=text,
            images=images,
            image_captions=captions,
            metadata={"filename": filename, "chars": len(text),
                       "paragraphs": len(text_parts)},
        )
    except ImportError:
        return extract_text(data, filename)
    except Exception as e:
        return ExtractedContent(error=f"DOCX extraction failed: {e}")


def extract_image(data: bytes, filename: str = "image") -> ExtractedContent:
    """Validate and return image bytes for CLIP encoding."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if w < 50 or h < 50:
            return ExtractedContent(error="Image too small (< 50x50)")
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return ExtractedContent(
            images=[buf.getvalue()],
            image_captions=[Path(filename).stem],
            metadata={"filename": filename, "width": w, "height": h},
        )
    except Exception as e:
        return ExtractedContent(error=str(e))


async def extract_url(url: str) -> ExtractedContent:
    """Fetch and extract text + images from a URL."""
    try:
        import aiohttp
        from bs4 import BeautifulSoup

        timeout = aiohttp.ClientTimeout(total=20)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            )
        }
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return ExtractedContent(error=f"HTTP {resp.status}")
                html = await resp.text(errors="replace")

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup
        )

        text_parts = []
        for tag in main.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote"]):
            t = tag.get_text(separator=" ", strip=True)
            if len(t) > 30:
                text_parts.append(t)

        text = "\n\n".join(text_parts)

        return ExtractedContent(
            text=text,
            metadata={"url": url, "title": title, "chars": len(text)},
        )
    except Exception as e:
        return ExtractedContent(error=str(e))
