"""Minimal PMC Open Access XML reader.

Only the elements needed by the claim extractor are parsed; the original file is
never copied into the claim object (evidence spans are offsets + sentence hash).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import BinaryIO, Iterable
import xml.etree.ElementTree as ET

from cbi.m3.asof import parse_date


@dataclass
class PMCArticle:
    pmcid: str | None
    pmid: str | None
    doi: str | None
    title: str
    journal: str | None
    pub_date: date | None
    abstract: str
    body: str
    full_text: str
    path: str | None = None
    tables_text: str = ""
    body_sections: dict[str, str] = field(default_factory=dict)

    @property
    def sections(self) -> dict[str, str]:
        return {"abstract": self.abstract, "body": self.body, "tables": self.tables_text}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _find(el: ET.Element, name: str) -> ET.Element | None:
    for child in el.iter():
        if _local(child.tag) == name:
            return child
    return None


def _find_all(el: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in el.iter() if _local(child.tag) == name]


def _abstract_text(el: ET.Element | None) -> str:
    """Join abstract children, keeping section titles separated from text."""
    if el is None:
        return ""
    parts = []
    for child in el:
        if not isinstance(child.tag, str):
            continue
        title = ""
        for grandchild in child:
            if isinstance(grandchild.tag, str) and _local(grandchild.tag) == "title":
                title = _text(grandchild)
                break
        text = _text(child)
        body = text[len(title):].lstrip() if title and text.startswith(title) else text
        if body:
            parts.append(f"{title}: {body}" if title else body)
    return " ".join(parts) if parts else _text(el)


def _section_body(sec: ET.Element) -> str:
    """Paragraph text with nested subsection titles kept as sentence leads."""
    pieces: list[str] = []
    for child in sec:
        if not isinstance(child.tag, str):
            continue
        tag = _local(child.tag)
        if tag == "title":
            continue
        if tag == "p":
            text = _text(child)
            if text:
                pieces.append(text)
        elif tag == "sec":
            title = ""
            for grandchild in child:
                if isinstance(grandchild.tag, str) and _local(grandchild.tag) == "title":
                    title = _text(grandchild)
                    break
            body = _section_body(child)
            if body:
                pieces.append(f"{title}. {body}" if title else body)
    return " ".join(pieces)


def _body_sections(root: ET.Element) -> dict[str, str]:
    """Top-level titled body sections (introduction, methods, results, ...)."""
    body = _find(root, "body")
    out: dict[str, str] = {}
    if body is None:
        return out
    for sec in body:
        if not isinstance(sec.tag, str) or _local(sec.tag) != "sec":
            continue
        title = ""
        for ch in sec:
            if isinstance(ch.tag, str) and _local(ch.tag) == "title":
                title = _text(ch)
                break
        text = _section_body(sec)
        if title and text:
            out[title] = text
    return out


def _table_text(root: ET.Element) -> str:
    """Render JATS tables as header:value rows so HR/CI columns stay readable."""
    lines: list[str] = []
    for wrap in _find_all(root, "table-wrap"):
        label = _text(_find(wrap, "label"))
        caption = _text(_find(wrap, "caption"))
        title = " ".join(part for part in (label, caption) if part)
        if title:
            lines.append(title)
        table = _find(wrap, "table")
        if table is None:
            continue
        headers: list[str] = []
        rows = 0
        for tr in _find_all(table, "tr"):
            cells = [c for c in list(tr) if _local(c.tag) in {"td", "th"}]
            if not cells:
                continue
            values = [_text(c) for c in cells]
            if not headers:
                headers = values
                continue
            if len(values) == len(headers):
                rendered = "; ".join(f"{h}: {v}" for h, v in zip(headers, values))
            else:
                rendered = " | ".join(values)
            lines.append(rendered)
            rows += 1
            if rows >= 500:
                break
    return "\n".join(lines)


def _pub_date(root: ET.Element) -> date | None:
    candidates: list[date] = []
    for pd in _find_all(root, "pub-date"):
        year = _text(_find(pd, "year"))
        month = _text(_find(pd, "month")) or "1"
        day = _text(_find(pd, "day")) or "1"
        try:
            candidates.append(date(int(year), int(month), int(day)))
        except (ValueError, TypeError):
            continue
    return min(candidates) if candidates else None


def _zip_xml_member(names: list[str]) -> str | None:
    preferred = [n for n in names if n.lower().endswith((".xml", ".nxml"))]
    return (preferred or names or [None])[0]


def read_pmc_source(source: str | Path | bytes | bytearray | BinaryIO) -> tuple[bytes, str | None]:
    """Return (xml_bytes, original_path).

    Accepts a path, a .zip/.xml payload, raw XML bytes, or a readable file-like
    object.  PMC OA ships both single .nxml files and zip archives; the archive
    path is the common case when downloading a bulk OA package.
    """
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        if raw[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                member = _zip_xml_member(zf.namelist())
                if member is None:
                    raise ValueError("zip archive contains no XML members")
                return zf.read(member), None
        return raw, None
    if hasattr(source, "read"):
        return read_pmc_source(source.read())
    if isinstance(source, str) and source.lstrip().startswith("<"):
        return source.encode("utf-8"), None
    path = Path(source)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = _zip_xml_member(zf.namelist())
            if member is None:
                raise ValueError("zip archive contains no XML members")
            return zf.read(member), str(path)
    return path.read_bytes(), str(path)


def parse_pmc_xml(source: str | Path | bytes | bytearray | BinaryIO) -> PMCArticle:
    raw, path = read_pmc_source(source)
    root = ET.fromstring(raw)
    article_meta = _find(root, "article-meta")
    journal_meta = _find(root, "journal-meta")
    title = _text(_find(article_meta, "article-title")) or _text(_find(root, "article-title"))
    journal = _text(_find(journal_meta, "journal-title")) if journal_meta is not None else None
    ids: dict[str, str] = {}
    if article_meta is not None:
        for node in _find_all(article_meta, "article-id"):
            kind = node.attrib.get("pub-id-type", "")
            ids[kind] = _text(node)
    abstract_el = _find(article_meta, "abstract") if article_meta is not None else None
    abstract = _abstract_text(abstract_el)
    body_el = _find(root, "body")
    paragraphs = []
    if body_el is not None:
        for p in _find_all(body_el, "p"):
            t = _text(p)
            if t:
                paragraphs.append(t)
    body = "\n\n".join(paragraphs)
    tables_text = _table_text(root)
    full_text = "\n\n".join(part for part in (abstract, body, tables_text) if part).strip()
    return PMCArticle(
        pmcid=ids.get("pmc") or ids.get("pmcid"),
        pmid=ids.get("pmid"),
        doi=ids.get("doi"),
        title=title,
        journal=journal,
        pub_date=_pub_date(root),
        abstract=abstract,
        body=body,
        full_text=full_text,
        path=path,
        tables_text=tables_text,
        body_sections=_body_sections(root),
    )


def parse_pmc_dir(path: str | Path) -> list[PMCArticle]:
    out = []
    for p in sorted([*Path(path).glob("*.xml"), *Path(path).glob("*.nxml")]):
        try:
            out.append(parse_pmc_xml(p))
        except (ET.ParseError, ValueError, zipfile.BadZipFile):
            continue
    return out
