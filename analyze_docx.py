import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
}


def qn(prefix, name):
    return f"{{{NS[prefix]}}}{name}"


def read_xml(zf, name):
    try:
        return ET.fromstring(zf.read(name))
    except KeyError:
        return None


def text_of(el, include_deleted=False):
    parts = []
    for node in el.iter():
        tag = node.tag
        if tag == qn("w", "del") and not include_deleted:
            continue
        if tag in (qn("w", "t"), qn("w", "delText")):
            in_del = any_ancestor_is(node, el, qn("w", "del"))
            if include_deleted or not in_del:
                parts.append(node.text or "")
        elif tag == qn("w", "tab"):
            parts.append("\t")
        elif tag == qn("w", "br"):
            parts.append("\n")
    return normalize_text("".join(parts))


def any_ancestor_is(node, root, target_tag):
    # ElementTree has no parent pointers, so walk from root to find the node path.
    path = []

    def walk(cur, ancestors):
        if cur is node:
            path.extend(ancestors)
            return True
        for child in list(cur):
            if walk(child, ancestors + [cur]):
                return True
        return False

    walk(root, [])
    return any(a.tag == target_tag for a in path)


def normalize_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text):
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)?", text))


def style_maps(styles_root):
    styles = {}
    if styles_root is None:
        return styles
    for style in styles_root.findall("w:style", NS):
        style_id = style.get(qn("w", "styleId"))
        style_type = style.get(qn("w", "type"))
        name_el = style.find("w:name", NS)
        name = name_el.get(qn("w", "val")) if name_el is not None else style_id
        styles[style_id] = {"name": name, "type": style_type}
    return styles


def paragraph_style(p, styles):
    p_style = p.find("w:pPr/w:pStyle", NS)
    if p_style is None:
        return {"id": None, "name": None, "type": None}
    style_id = p_style.get(qn("w", "val"))
    meta = styles.get(style_id, {})
    return {"id": style_id, "name": meta.get("name", style_id), "type": meta.get("type")}


def infer_heading_level(text, style):
    name = (style.get("name") or "").lower()
    style_id = (style.get("id") or "").lower()
    m = re.search(r"(heading|заголовок)\s*([1-9])", name)
    if m:
        return int(m.group(2))
    m = re.search(r"heading([1-9])", style_id)
    if m:
        return int(m.group(1))
    if re.match(r"^(введение|заключение|список|приложени[ея]|оглавление)\b", text.lower()):
        return 1
    if re.match(r"^глава\s+\d+", text.lower()):
        return 1
    if re.match(r"^\d+\.\s+\S", text):
        return 1
    if re.match(r"^\d+\.\d+\.?\s+\S", text):
        return 2
    if re.match(r"^\d+\.\d+\.\d+\.?\s+\S", text):
        return 3
    return None


def is_probable_heading(text, style):
    if not text or len(text) > 240:
        return False
    if infer_heading_level(text, style):
        return True
    letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", text)
    if len(letters) >= 8 and letters.upper() == letters and not re.search(r"[.!?]$", text):
        return True
    return False


def table_rows(tbl):
    rows = []
    for tr in tbl.findall("w:tr", NS):
        row = []
        for tc in tr.findall("w:tc", NS):
            cell_parts = []
            for p in tc.findall(".//w:p", NS):
                t = text_of(p)
                if t:
                    cell_parts.append(t)
            row.append(" | ".join(cell_parts))
        rows.append(row)
    return rows


def extract_props(root, ns_name):
    props = {}
    if root is None:
        return props
    for child in list(root):
        tag = child.tag.split("}", 1)[-1]
        props[tag] = normalize_text(child.text or "")
    return props


def extract_comments(zf):
    comments_root = read_xml(zf, "word/comments.xml")
    comments = []
    if comments_root is None:
        return comments
    for comment in comments_root.findall("w:comment", NS):
        comments.append(
            {
                "id": comment.get(qn("w", "id")),
                "author": comment.get(qn("w", "author")),
                "date": comment.get(qn("w", "date")),
                "text": text_of(comment, include_deleted=True),
            }
        )
    return comments


def xml_count(root, tag_name):
    if root is None:
        return 0
    return sum(1 for _ in root.iter(qn("w", tag_name)))


def make_section_chunks(elements):
    sections = []
    current = {"heading": "Front matter / before first heading", "level": 0, "items": []}
    for item in elements:
        if item["type"] == "paragraph" and item.get("heading_level") and item.get("heading_level") <= 2:
            if current["items"] or current["heading"] != "Front matter / before first heading":
                sections.append(current)
            current = {
                "heading": item["text"],
                "level": item["heading_level"],
                "items": [],
            }
        elif item["type"] == "paragraph" and item.get("text"):
            current["items"].append(item["text"])
        elif item["type"] == "table":
            current["items"].append(f"[Table {item['index']}: {item['rows']} rows x {item['cols']} cols]")
    if current["items"] or current["heading"] != "Front matter / before first heading":
        sections.append(current)
    for sec in sections:
        joined = "\n".join(sec["items"])
        sec["word_count"] = word_count(joined)
        sec["preview"] = normalize_text(joined[:1400])
    return sections


def analyze(input_path, out_dir):
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path) as zf:
        names = zf.namelist()
        doc_root = read_xml(zf, "word/document.xml")
        styles = style_maps(read_xml(zf, "word/styles.xml"))
        core_props = extract_props(read_xml(zf, "docProps/core.xml"), "core")
        app_props = extract_props(read_xml(zf, "docProps/app.xml"), "app")
        comments = extract_comments(zf)
        media = [n for n in names if n.startswith("word/media/")]
        footnotes_root = read_xml(zf, "word/footnotes.xml")
        endnotes_root = read_xml(zf, "word/endnotes.xml")

        elements = []
        headings = []
        tables = []
        body = doc_root.find("w:body", NS) if doc_root is not None else None
        if body is not None:
            table_index = 0
            para_index = 0
            for child in list(body):
                if child.tag == qn("w", "p"):
                    para_index += 1
                    txt = text_of(child)
                    style = paragraph_style(child, styles)
                    h_level = infer_heading_level(txt, style) if is_probable_heading(txt, style) else None
                    item = {
                        "type": "paragraph",
                        "index": para_index,
                        "style": style,
                        "text": txt,
                        "word_count": word_count(txt),
                        "heading_level": h_level,
                    }
                    elements.append(item)
                    if h_level:
                        headings.append(
                            {
                                "level": h_level,
                                "text": txt,
                                "style": style.get("name"),
                                "paragraph_index": para_index,
                            }
                        )
                elif child.tag == qn("w", "tbl"):
                    table_index += 1
                    rows = table_rows(child)
                    max_cols = max((len(r) for r in rows), default=0)
                    tables.append(
                        {
                            "index": table_index,
                            "rows": len(rows),
                            "cols": max_cols,
                            "preview": rows[:4],
                        }
                    )
                    elements.append(
                        {
                            "type": "table",
                            "index": table_index,
                            "rows": len(rows),
                            "cols": max_cols,
                            "preview": rows[:2],
                        }
                    )

        full_text_parts = []
        for item in elements:
            if item["type"] == "paragraph" and item.get("text"):
                full_text_parts.append(item["text"])
            elif item["type"] == "table":
                full_text_parts.append(f"[Table {item['index']}: {item['rows']} rows x {item['cols']} cols]")
        full_text = "\n\n".join(full_text_parts)

        sections = make_section_chunks(elements)
        style_counts = Counter(
            (item.get("style", {}).get("name") or "(no style)")
            for item in elements
            if item["type"] == "paragraph" and item.get("text")
        )

        summary = {
            "input": str(input_path),
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "core_properties": core_props,
            "app_properties": app_props,
            "counts": {
                "paragraphs": sum(1 for i in elements if i["type"] == "paragraph"),
                "nonempty_paragraphs": sum(
                    1 for i in elements if i["type"] == "paragraph" and i.get("text")
                ),
                "words_estimate": word_count(full_text),
                "tables": len(tables),
                "images_or_media_files": len(media),
                "comments": len(comments),
                "footnotes": max(0, xml_count(footnotes_root, "footnote") - 2),
                "endnotes": max(0, xml_count(endnotes_root, "endnote") - 2),
                "insertions_tracked": xml_count(doc_root, "ins"),
                "deletions_tracked": xml_count(doc_root, "del"),
            },
            "style_counts_top": style_counts.most_common(30),
            "headings": headings,
            "tables": tables,
            "comments": comments,
            "sections": sections,
            "media_files_sample": media[:20],
        }

        (out_dir / "analysis.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "extracted_text.txt").write_text(full_text, encoding="utf-8")

        md_lines = []
        md_lines.append(f"# DOCX analysis: {input_path.name}")
        md_lines.append("")
        md_lines.append("## Counts")
        for key, value in summary["counts"].items():
            md_lines.append(f"- {key}: {value}")
        md_lines.append("")
        md_lines.append("## Core properties")
        for key, value in core_props.items():
            if value:
                md_lines.append(f"- {key}: {value}")
        md_lines.append("")
        md_lines.append("## Extended properties")
        for key, value in app_props.items():
            if value:
                md_lines.append(f"- {key}: {value}")
        md_lines.append("")
        md_lines.append("## Heading outline")
        for h in headings:
            indent = "  " * max(0, h["level"] - 1)
            md_lines.append(f"{indent}- L{h['level']}: {h['text']} ({h['style']})")
        md_lines.append("")
        md_lines.append("## Section previews")
        for sec in sections:
            md_lines.append(f"### {sec['heading']}")
            md_lines.append(f"Words: {sec['word_count']}")
            if sec["preview"]:
                md_lines.append(sec["preview"])
            md_lines.append("")
        md_lines.append("## Tables")
        for table in tables[:50]:
            md_lines.append(f"- Table {table['index']}: {table['rows']} x {table['cols']}")
            if table["preview"]:
                md_lines.append(f"  preview: {table['preview']}")
        if len(tables) > 50:
            md_lines.append(f"- ... {len(tables) - 50} more tables")
        md_lines.append("")
        if comments:
            md_lines.append("## Comments")
            for c in comments:
                md_lines.append(
                    f"- #{c.get('id')} by {c.get('author')}: {c.get('text')[:300]}"
                )
        (out_dir / "analysis.md").write_text("\n".join(md_lines), encoding="utf-8")

        return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="analysis_outputs")
    args = parser.parse_args()
    summary = analyze(args.input, args.out)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
