"""Download the sources, split them into citable chunks and build the vector index.

Laws (HTML, consolidated text) are split by article, so every chunk can be cited as
"art. 31". Annexes (PDF) are split by page, so every chunk can be cited as "p. 9".
Text that the consolidated version marks as revoked (<del>) is dropped.

    coe-ingest            # download if needed, chunk, embed
    coe-ingest --no-embed # chunk only (BM25 still works, no API key needed)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field

import httpx

from .config import CHUNKS_FILE, RAW_DIR, SOURCES, Source

MAX_CHARS = 1800  # long articles are split into parts of about this size

ART_RE = re.compile(r"^Art\.\s*(\d+)\s*(?:º|°|o)?\s*(-\s*[A-Z])?\b\.?")
HEADING_RE = re.compile(r"^(CAPÍTULO|Seção|SEÇÃO|Subseção|SUBSEÇÃO|TÍTULO)\b")
ANNEX_RE = re.compile(r"Anexo\s+([IVX]+)\s+(?:integrante|Integrante)", re.IGNORECASE)


@dataclass
class Chunk:
    id: str  # e.g. "L16642:art-31", "D57776-anexos:p15"
    source: str  # Source.key
    locator: str  # human readable: "art. 31", "p. 15 (Anexo I)"
    heading: str  # chapter / section path, or annex name
    text: str
    url: str
    article: str | None = None
    page: int | None = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- download


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "coe-agent/0.1 (portfolio project)"}
    with httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client:
        for src in SOURCES:
            path = RAW_DIR / src.file
            if path.exists() and not force:
                continue
            print(f"downloading {src.file} ...")
            resp = client.get(src.url)
            resp.raise_for_status()
            path.write_bytes(resp.content)


# --------------------------------------------------------------------------- laws (HTML)


def _law_lines(src: Source) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup((RAW_DIR / src.file).read_text(encoding="utf-8"), "html.parser")
    first_article = soup.find(string=re.compile(r"Art\.\s*1\s*º"))
    if first_article is None:
        raise RuntimeError(f"{src.file}: could not find Art. 1º")
    body = first_article.find_parent("div")
    for revoked in body.find_all("del"):
        revoked.decompose()
    lines = []
    for block in body.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        text = re.sub(r"\s+", " ", block.get_text(" ")).strip()
        if text:
            lines.append(text)
    return lines


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + len(line) + 1 > limit:
            parts.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


def chunk_law(src: Source) -> list[Chunk]:
    lines = _law_lines(src)
    chunks: list[Chunk] = []
    chapter = section = ""
    pending_heading: str | None = None
    article: str | None = None
    heading_at_article = ""
    buf: list[str] = []

    def flush() -> None:
        if article is None or not buf:
            return
        text = "\n".join(buf)
        parts = _split_long(text)
        for i, part in enumerate(parts, 1):
            suffix = f"-{i}" if len(parts) > 1 else ""
            chunks.append(
                Chunk(
                    id=f"{src.key}:art-{article}{suffix}",
                    source=src.key,
                    locator=f"art. {article}" + (f" (parte {i}/{len(parts)})" if suffix else ""),
                    heading=heading_at_article,
                    text=part if i == 1 else f"[art. {article}, continuação]\n{part}",
                    url=src.url,
                    article=article,
                )
            )

    for line in lines:
        if pending_heading is not None:
            # the line after "CAPÍTULO II" / "Seção I" is its title
            label = f"{pending_heading} - {line}"
            if pending_heading.upper().startswith("CAP") or pending_heading.upper().startswith("T"):
                chapter, section = label, ""
            else:
                section = label
            pending_heading = None
            continue
        if HEADING_RE.match(line) and len(line) < 40:
            pending_heading = line
            continue
        m = ART_RE.match(line)
        if m:
            flush()
            article = m.group(1) + (m.group(2) or "").replace(" ", "").replace("-", "-")
            heading_at_article = " > ".join(x for x in (chapter, section) if x)
            buf = [line]
            continue
        if article is not None:
            if line.startswith(("Voltar", "Este texto não substitui")):
                continue
            buf.append(line)
    flush()
    return chunks


# --------------------------------------------------------------------------- annexes (PDF)


def chunk_annex(src: Source) -> list[Chunk]:
    import pymupdf

    chunks: list[Chunk] = []
    current_annex = ""
    with pymupdf.open(RAW_DIR / src.file) as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text()
            text = re.sub(r"[ \t]+\n", "\n", text)
            text = re.sub(r"\n{2,}", "\n", text).strip()
            m = ANNEX_RE.search(text[:300])
            if m:
                current_annex = f"Anexo {m.group(1)}"
            if len(text) < 200:  # drawings without useful text
                continue
            # Tables come out of get_text() one cell per line. Add a Markdown version so the
            # LLM can read rows and columns (e.g. ceiling height x area x room type).
            tables = [t.to_markdown().strip() for t in page.find_tables().tables]
            tables = [t for t in tables if t.count("\n") >= 2]
            if tables:
                text += "\n\n" + "\n\n".join(f"[tabela da página, em Markdown]\n{t}" for t in tables)
            chunks.append(
                Chunk(
                    id=f"{src.key}:p{i}",
                    source=src.key,
                    locator=f"p. {i}" + (f" ({current_annex})" if current_annex else ""),
                    heading=current_annex,
                    text=text,
                    url=f"{src.url}#page={i}",
                    page=i,
                )
            )
    return chunks


# --------------------------------------------------------------------------- main


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for src in SOURCES:
        part = chunk_law(src) if src.kind == "law" else chunk_annex(src)
        print(f"{src.key}: {len(part)} chunks")
        chunks.extend(part)
    ids = [c.id for c in chunks]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise RuntimeError(f"duplicate chunk ids: {sorted(dupes)[:10]}")
    return chunks


def save_chunks(chunks: list[Chunk]) -> None:
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} chunks to {CHUNKS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-embed", action="store_true", help="skip the embedding step")
    args = parser.parse_args()

    download(force=args.force_download)
    chunks = build_chunks()
    save_chunks(chunks)
    if not args.no_embed:
        from .retrieval import build_vector_index

        build_vector_index()


if __name__ == "__main__":
    main()
