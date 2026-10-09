#!/usr/bin/env python3
"""
Extract PDF annotations (highlights, sticky notes, strike-outs) from an annotated
review PDF, and report each one with its page, the text it covers, and the CVPR
review line number -- so feedback can be acted on without hunting for locations.

Uses only `qpdf` and `pdftotext`, both already present on this cluster. No pip
install, no Slurm job: an earlier assumption that a parser had to be installed
was wrong.

  qpdf --json      -> annotation objects: page, subtype, /Contents, /Rect
  pdftotext -bbox  -> word bounding boxes, to recover the text under each /Rect
                      (and the ruler numbers the CVPR review style prints, which
                      is what gives us a line number for free)

Usage:
  python extract_annotations.py annotated.pdf            # human-readable
  python extract_annotations.py annotated.pdf --md       # markdown checklist
"""
from __future__ import annotations

import argparse
import html.parser
import json
import re
import subprocess
import sys
from pathlib import Path

# CVPR's review ruler prints one number per line in the outer margin. A token that
# is purely 3-4 digits and sits near the page edge is a ruler number, not content.
RULER_MAX_X_FRAC = 0.085   # left ruler: within this fraction of page width
RULER_MIN_X_FRAC = 0.915   # right ruler: beyond this fraction


class _BBoxParser(html.parser.HTMLParser):
    """pdftotext -bbox emits XHTML with <page>/<word> elements carrying coordinates."""

    def __init__(self):
        super().__init__()
        self.pages: list[dict] = []
        self._w = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "page":
            self.pages.append({"width": float(a["width"]), "height": float(a["height"]),
                               "words": []})
        elif tag == "word":
            self._w = {k: float(a[k]) for k in ("xmin", "ymin", "xmax", "ymax")}
            self._w["text"] = ""

    def handle_data(self, data):
        if self._w is not None:
            self._w["text"] += data

    def handle_endtag(self, tag):
        if tag == "word" and self._w is not None:
            self._w["text"] = self._w["text"].strip()
            if self._w["text"]:
                self.pages[-1]["words"].append(self._w)
            self._w = None


def page_words(pdf: Path) -> list[dict]:
    xml = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"],
                         capture_output=True, text=True, check=True).stdout
    p = _BBoxParser()
    p.feed(xml)
    return p.pages


def _s(v):
    """qpdf JSON v2 prefixes strings: 'u:' for unicode, 'b:' for base64 binary."""
    if isinstance(v, str):
        if v.startswith("u:"):
            return v[2:]
        if v.startswith("b:"):
            return ""
    return v or ""


def _deref(objs: dict, ref):
    """Objects are keyed 'obj:N 0 R', but references inside the model are 'N 0 R'."""
    if not isinstance(ref, str):
        return ref
    o = objs.get(ref) or objs.get(f"obj:{ref}")
    if isinstance(o, dict) and "value" in o:
        return o["value"]
    return o


def annotations(pdf: Path) -> list[dict]:
    """Pull annotation dicts out of the qpdf object model, tagged with their page."""
    j = json.loads(subprocess.run(["qpdf", "--json", str(pdf)],
                                  capture_output=True, text=True, check=True).stdout)
    q = j.get("qpdf")
    objs = q[1] if isinstance(q, list) and len(q) > 1 else j.get("objects", {})
    pages = j.get("pages", [])
    out = []
    for pageno, page in enumerate(pages, start=1):
        pobj = _deref(objs, page.get("object")) or {}
        annots = pobj.get("/Annots")
        if isinstance(annots, str):                       # indirect array
            annots = _deref(objs, annots)
        if not isinstance(annots, list):
            continue
        for ref in annots:
            a = _deref(objs, ref)
            if not isinstance(a, dict):
                continue
            sub = str(a.get("/Subtype", "")).lstrip("/")
            if sub in ("Link", "Popup"):
                continue
            rect = a.get("/Rect")
            if isinstance(rect, str):
                rect = _deref(objs, rect)
            out.append({
                "page": pageno,
                "type": sub,
                "contents": _s(a.get("/Contents")).strip(),
                "author": (_s(a.get("/T")) or _s(a.get("/Name")).lstrip("/")).strip(),
                "rect": [float(x) for x in rect] if isinstance(rect, list) else None,
            })
    return out


def covered(page: dict, rect: list[float]) -> tuple[str, list[int]]:
    """Words under `rect`, split into content text and CVPR ruler line numbers.

    PDF y grows upward; pdftotext y grows downward -- hence the flip."""
    x0, y0, x1, y1 = min(rect[0], rect[2]), min(rect[1], rect[3]), max(rect[0], rect[2]), max(rect[1], rect[3])
    H, W = page["height"], page["width"]
    top, bot = H - y1, H - y0
    text, lines = [], []
    for w in page["words"]:
        if w["xmax"] < x0 or w["xmin"] > x1 or w["ymax"] < top or w["ymin"] > bot:
            continue
        near_edge = w["xmax"] < RULER_MAX_X_FRAC * W or w["xmin"] > RULER_MIN_X_FRAC * W
        if near_edge and re.fullmatch(r"\d{1,4}", w["text"]):
            lines.append(int(w["text"]))
        else:
            text.append(w["text"])
    return " ".join(text), sorted(set(lines))


def nearest_lines(page: dict, rect: list[float]) -> list[int]:
    """Ruler numbers on the same horizontal band as the annotation.

    CVPR prints a ruler in BOTH margins: the left one numbers the left column,
    the right one the right column. A horizontal band therefore crosses two
    unrelated rulers, so we keep only the one on the annotation's own side --
    otherwise a left-column note reports both L295-298 and L346-349, which is
    worse than useless."""
    H, W = page["height"], page["width"]
    top, bot = H - max(rect[1], rect[3]), H - min(rect[1], rect[3])
    left_col = (min(rect[0], rect[2]) + max(rect[0], rect[2])) / 2 < W / 2
    band = []
    for w in page["words"]:
        if not re.fullmatch(r"\d{1,4}", w["text"]):
            continue
        on_left, on_right = w["xmax"] < RULER_MAX_X_FRAC * W, w["xmin"] > RULER_MIN_X_FRAC * W
        if not (on_left if left_col else on_right):
            continue
        if w["ymax"] > top - 6 and w["ymin"] < bot + 6:
            band.append(w)
    return sorted({int(w["text"]) for w in band})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--md", action="store_true", help="emit a markdown checklist")
    a = ap.parse_args()

    pages = page_words(a.pdf)
    anns = annotations(a.pdf)
    if not anns:
        print(f"No annotations found in {a.pdf}.")
        print("(If you have annotated it, make sure the editor SAVED/flattened the file.)")
        return 0

    anns.sort(key=lambda x: (x["page"], -(x["rect"][3] if x["rect"] else 0)))
    if a.md:
        print(f"# Review feedback extracted from `{a.pdf.name}`\n")
        print(f"{len(anns)} annotations.\n")
    for i, an in enumerate(anns, 1):
        pg = pages[an["page"] - 1] if an["page"] <= len(pages) else None
        txt, lines = ("", [])
        if pg and an["rect"]:
            txt, lines = covered(pg, an["rect"])
            if not lines:
                lines = nearest_lines(pg, an["rect"])
        loc = f"p{an['page']}" + (f" L{lines[0]}" if len(lines) == 1 else
                                  f" L{lines[0]}-{lines[-1]}" if lines else "")
        if a.md:
            print(f"- [ ] **{loc}** ({an['type']}){' — ' + an['author'] if an['author'] else ''}")
            if txt:
                print(f"      > {txt[:400]}")
            if an["contents"]:
                print(f"      **{an['contents']}**")
        else:
            print(f"\n[{i}] {loc}  ({an['type']})")
            if txt:
                print(f"    text: {txt[:300]}")
            if an["contents"]:
                print(f"    note: {an['contents']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
