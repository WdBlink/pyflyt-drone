#!/usr/bin/env python3
"""Venue calibration measurements for the deli-autoresearch paper workflow.

Deterministic only: no model calls, no network access, no scheduling.
Measures accepted papers (page count, figure/table counts, caption structure),
aggregates a venue profile, validates this paper's figure/table plan against
that profile, and checks the compiled draft against the plan.

Requires Python 3.9+ standard library. PDF text extraction needs poppler
(`pdftotext`, `pdfinfo`) on PATH, or pass pre-extracted text files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile


SKILL = Path(__file__).resolve().parents[1]
VERSION = 1
LIMIT = 16 * 1024 * 1024

FIGURE_TYPES = {
    "method_overview", "system_architecture", "teaser", "qualitative_comparison",
    "quantitative_results", "ablation_visual", "failure_case", "dataset_example",
    "pipeline_detail", "analysis_plot",
}
TABLE_TYPES = {
    "main_results", "ablation", "comparison_related_work", "dataset_statistics",
    "hyperparameter", "cost_breakdown", "taxonomy", "survey_summary", "notation",
}
PRIORITIES = ("must", "optional")
PLACEMENTS = ("body", "appendix", "supplementary")

FIGURE_LABEL = re.compile(
    r"^[ \t]*(?:figures?|figs?)\.?(?=[ \t])[ \t]*"
    r"(?P<num>\d+(?:[-–]\d+)?|[ivxlcdm]+)(?=[ \t.:\-–—])[ \t]*"
    r"(?P<sep>[:.．\-–—])[ \t]*(?P<rest>\S.*)?$",
    re.IGNORECASE,
)
TABLE_LABEL = re.compile(
    r"^[ \t]*(?:tables?|tabs?)\.?(?=[ \t])[ \t]*"
    r"(?P<num>\d+(?:[-–]\d+)?|[ivxlcdm]+)(?=[ \t.:\-–—])[ \t]*"
    r"(?P<sep>[:.．\-–—])[ \t]*(?P<rest>\S.*)?$",
    re.IGNORECASE,
)
FIGURE_MENTION = re.compile(
    r"\b(?:figures?|figs?)\.?\s*(?P<num>\d+(?:[-–]\d+)?|[ivxlcdm]+)\b", re.IGNORECASE)
TABLE_MENTION = re.compile(
    r"\b(?:tables?|tabs?)\.?\s*(?P<num>\d+(?:[-–]\d+)?|[ivxlcdm]+)\b", re.IGNORECASE)
PANEL_TOKEN = re.compile(r"\((?P<label>[a-z])\)")
SECTION_HEAD = re.compile(r"^[ \t]*(?:\d+(?:\.\d+)*[ \t]+)?[A-Z][A-Za-z0-9 ,:'’\-]{0,60}$")
REFERENCE_HEAD = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+)*[ \t]+)?(?:references|reference|bibliography|参考文献)[ \t]*:?[ \t]*$",
    re.IGNORECASE)
RESULT_HINT = re.compile(
    r"\b(compare|comparison|ablation|results?|performance|accuracy|success rate|"
    r"baseline|speed-?up|latency|runtime|error rate|overall|quantitative|"
    r"main result|setting|hyperparameter|statistics|summary)\b",
    re.IGNORECASE,
)


class VenueError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise VenueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    data = path.read_bytes()
    require(len(data) <= LIMIT, f"{path}: exceeds {LIMIT} bytes")
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise VenueError(f"{path}: invalid UTF-8 JSON: {exc}") from exc


def roman_to_int(text):
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    lowered = text.lower()
    if not lowered or any(ch not in values for ch in lowered):
        return None
    total, previous = 0, 0
    for ch in reversed(lowered):
        value = values[ch]
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total


def label_number(text):
    if text.isdigit():
        return int(text)
    if "-" in text or "–" in text:
        return None
    value = roman_to_int(text)
    if value is None:
        return None
    if to_roman(value).lower() != text.lower():
        return None
    return value


def to_roman(value):
    table = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
             (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
             (5, "v"), (4, "iv"), (1, "i")]
    parts = []
    for amount, symbol in table:
        while value >= amount:
            parts.append(symbol)
            value -= amount
    return "".join(parts)


def tool(name):
    found = shutil.which(name)
    require(found, f"Required tool '{name}' not found on PATH. Install poppler "
                    f"(pdftotext/pdfinfo) or pass pre-extracted text instead.")
    return found


def pdf_to_text(path):
    text = subprocess.run([tool("pdftotext"), "-q", "-enc", "UTF-8", str(path), "-"],
                          capture_output=True, timeout=300)
    require(text.returncode == 0,
            f"pdftotext failed for {path}: {text.stderr.decode('utf-8', 'replace').strip()}")
    return text.stdout.decode("utf-8", "replace")


def pdf_page_count(path):
    info = subprocess.run([tool("pdfinfo"), str(path)], capture_output=True, timeout=120)
    require(info.returncode == 0,
            f"pdfinfo failed for {path}: {info.stderr.decode('utf-8', 'replace').strip()}")
    for line in info.stdout.decode("utf-8", "replace").splitlines():
        if line.lower().startswith("pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                break
    return None


def split_pages(text):
    return [page for page in text.split("\f")]


def caption_body(lines, index):
    """Collect the caption text that follows a label line.

    Stops at a blank line, another caption label, or a section heading, so a
    caption never swallows the following paragraph.
    """
    parts = [lines[index]]
    for line in lines[index + 1:index + 4]:
        stripped = line.strip()
        if not stripped:
            break
        if FIGURE_LABEL.match(line) or TABLE_LABEL.match(line):
            break
        if REFERENCE_HEAD.match(line):
            break
        if SECTION_HEAD.match(stripped):
            break
        parts.append(stripped)
    return " ".join(part.strip() for part in parts if part.strip())


def panel_markers(lines, index):
    """Collect subfigure panel labels such as (a), (b).

    LaTeX subfigures emit their own '(a) caption' lines *before* the parent
    caption, and some layouts put the markers on the line after it. Scan both
    directions, but only near the caption, so panel labels from unrelated text
    do not get attributed to this figure.

    Backward scan (the common LaTeX case) walks up to the nearest blank line,
    because subfigure captions are separated from the parent caption by blank
    lines. Forward scan stops at the first blank line.
    """
    markers = set()
    for offset in (1, 2, 3):
        position = index + offset
        if position >= len(lines):
            break
        line = lines[position].strip()
        if not line:
            break
        if FIGURE_LABEL.match(line) or TABLE_LABEL.match(line):
            break
        if REFERENCE_HEAD.match(line) or SECTION_HEAD.match(line):
            break
        for token in PANEL_TOKEN.finditer(line):
            markers.add(token.group("label"))
    for offset in (-1, -2, -3, -4, -5):
        position = index + offset
        if position < 0:
            break
        line = lines[position].strip()
        if not line:
            continue
        if FIGURE_LABEL.match(line) or TABLE_LABEL.match(line):
            break
        if REFERENCE_HEAD.match(line) or SECTION_HEAD.match(line):
            break
        for token in PANEL_TOKEN.finditer(line):
            markers.add(token.group("label"))
    return sorted(markers)


APPENDIX_REF = re.compile(
    r"\b(?:in|from|see|of|and)\s+(?:the\s+)?"
    r"(?:appendix|supplement(?:ary|al material)?|app\.|sec\.|section)\b",
    re.IGNORECASE)


def body_cited_numbers(text, pattern, limit=None):
    """Numbers cited in the running text, excluding pointers into the appendix.

    A phrase such as "Figure 5 of Appendix E" names an appendix figure, not a
    body figure whose caption was missed. Counting it as a missing body
    caption would report a false undercount.
    """
    found = set()
    for page_index, page in enumerate(split_pages(text)):
        if limit is not None and page_index >= limit:
            continue
        for match in pattern.finditer(page):
            number = label_number(match.group("num"))
            if number is None:
                continue
            tail = page[match.end():match.end() + 60]
            if APPENDIX_REF.search(tail):
                continue
            found.add(number)
    return found


def measure_text(text, source=None, digest=None, reported_pages=None):
    """Measure one paper from extracted text. Pure function of `text`."""
    pages = split_pages(text)
    total_pages = reported_pages if reported_pages is not None else len(pages)
    figures, tables = [], []
    seen_figures, seen_tables = set(), set()
    reference_page = None
    for page_index, page in enumerate(pages):
        lines = page.splitlines()
        for line_index, line in enumerate(lines):
            if reference_page is None and REFERENCE_HEAD.match(line):
                reference_page = page_index
            for kind, pattern in (("figure", FIGURE_LABEL), ("table", TABLE_LABEL)):
                match = pattern.match(line)
                if not match:
                    continue
                number = label_number(match.group("num"))
                if number is None:
                    continue
                key = (kind, number)
                if key in (seen_figures if kind == "figure" else seen_tables):
                    continue
                if kind == "figure":
                    seen_figures.add(key)
                else:
                    seen_tables.add(key)
                body = caption_body(lines, line_index)
                if kind == "figure":
                    panels = panel_markers(lines, line_index)
                    record = {
                        "number": number,
                        "label_text": match.group("num"),
                        "caption": body[:600],
                        "page": page_index + 1,
                        "panels": panels,
                        "panel_count": len(panels),
                    }
                    figures.append(record)
                else:
                    record = {
                        "number": number,
                        "label_text": match.group("num"),
                        "caption": body[:600],
                        "page": page_index + 1,
                        "result_hint": bool(RESULT_HINT.search(body)),
                    }
                    tables.append(record)
    figures.sort(key=lambda item: item["number"])
    tables.sort(key=lambda item: item["number"])
    # References usually starts mid-page, so the page carrying the heading still
    # holds body content: count it as body, not as references.
    if reference_page is not None:
        main_body_pages = reference_page + 1
        pages_basis = "references_heading"
    else:
        # No References heading: every page is body for this paper, but that
        # is a different measurement basis than papers that do have one. Keep
        # the basis visible so the pages figure is not compared as if it were
        # measured the same way.
        main_body_pages = total_pages
        pages_basis = "total_pages_fallback"
    warnings = []
    if figures and max(item["number"] for item in figures) != len(figures):
        warnings.append("figure numbering is not contiguous; some captions were not detected")
    if tables and max(item["number"] for item in tables) != len(tables):
        warnings.append("table numbering is not contiguous; some captions were not detected")
    if reference_page is None:
        warnings.append("no References heading found; main-body page count falls back to total pages")
    # A figure or table cited in the running text but never captured as a
    # caption means the caption block was not recognised: the count is a floor,
    # not a total. Report it instead of letting a low count pass unnoticed.
    #
    # Only citations on body pages count. A reference such as "Figure 5 of
    # Appendix E" points at an appendix figure, not at a missing body caption,
    # and must not raise a false undercount warning.
    body_limit = reference_page if reference_page is not None else total_pages
    cited_figures = body_cited_numbers(text, FIGURE_MENTION, limit=body_limit)
    cited_tables = body_cited_numbers(text, TABLE_MENTION, limit=body_limit)
    missed_figures = sorted(cited_figures - {item["number"] for item in figures})
    missed_tables = sorted(cited_tables - {item["number"] for item in tables})
    if missed_figures:
        warnings.append("figures cited in the body but not captured as captions "
                        f"(likely undercount): {missed_figures}")
    if missed_tables:
        warnings.append("tables cited in the body but not captured as captions "
                        f"(likely undercount): {missed_tables}")
    return {
        "source": source,
        "sha256": digest,
        "pages_total": total_pages,
        "pages_main_body": main_body_pages,
        "pages_basis": pages_basis,
        "figures": figures,
        "tables": tables,
        "counts": {
            "figures": len(figures),
            "tables": len(tables),
            "figure_panels": sum(item["panel_count"] for item in figures),
            "result_tables": sum(1 for item in tables if item["result_hint"]),
        },
        "warnings": warnings,
    }


def measure_pdf(path):
    path = Path(path)
    require(path.is_file(), f"No such file: {path}")
    return measure_text(pdf_to_text(path), source=str(path),
                        digest=sha256_file(path), reported_pages=pdf_page_count(path))


def measure_text_file(path):
    path = Path(path)
    require(path.is_file(), f"No such file: {path}")
    raw = path.read_bytes()
    require(len(raw) <= LIMIT, f"{path}: exceeds {LIMIT} bytes")
    return measure_text(raw.decode("utf-8", "replace"), source=str(path),
                        digest=hashlib.sha256(raw).hexdigest())


def percentile(values, quantile):
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    position = (len(data) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(data[low])
    return data[low] + (data[high] - data[low]) * (position - low)


def summarize(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return {
        "n": len(clean),
        "min": min(clean),
        "p25": round(percentile(clean, 0.25), 2),
        "median": round(statistics.median(clean), 2),
        "p75": round(percentile(clean, 0.75), 2),
        "max": max(clean),
        "mean": round(statistics.fmean(clean), 2),
    }


def budget_band(values):
    """Integer band around the sample median.

    target = rounded median; min = floor(p25); max = ceil(p75), with the
    percentiles linearly interpolated. Rounding outward keeps the band from
    excluding counts that sit inside the sample's interquartile range: it
    constrains outliers, not typical values.
    """
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    target = int(math.floor(statistics.median(clean) + 0.5))
    low = int(math.floor(percentile(clean, 0.25)))
    high = int(math.ceil(percentile(clean, 0.75)))
    low = max(low, 0)
    high = max(high, low)
    target = min(max(target, low), high)
    return {"target": target, "min": low, "max": high}


def build_profile(records, venue, direction, design=None):
    require(records, "No sample records: measure at least one accepted paper first")
    pages = [record.get("pages_main_body") for record in records]
    figures = [record.get("counts", {}).get("figures") for record in records]
    tables = [record.get("counts", {}).get("tables") for record in records]
    panels = [record.get("counts", {}).get("figure_panels") for record in records]
    result_tables = [record.get("counts", {}).get("result_tables") for record in records]
    # Page counts measured on different bases are not the same quantity. Report
    # the mixed basis instead of averaging it away.
    bases = sorted({record.get("pages_basis") or "unknown" for record in records})
    stats = {
        "pages_main_body": summarize(pages),
        "pages_total": summarize([record.get("pages_total") for record in records]),
        "figures": summarize(figures),
        "tables": summarize(tables),
        "figure_panels": summarize(panels),
        "result_tables": summarize(result_tables),
    }
    profile = {
        "version": VERSION,
        "generated_at": now(),
        "venue": venue,
        "direction": direction,
        "sample_size": len(records),
        "stats": stats,
        "pages_basis": bases if len(bases) > 1 else bases[0],
        "pages_basis_mixed": len(bases) > 1,
        "budget": {
            "figures": budget_band(figures),
            "tables": budget_band(tables),
            "pages_main_body": budget_band(pages),
        },
        "samples": [{
            "source": record.get("source"),
            "sha256": record.get("sha256"),
            "venue_label": record.get("venue_label"),
            "year": record.get("year"),
            "acceptance_evidence": record.get("acceptance_evidence"),
            "direction_match": record.get("direction_match"),
            "pages_main_body": record.get("pages_main_body"),
            "pages_total": record.get("pages_total"),
            "pages_basis": record.get("pages_basis"),
            "counts": record.get("counts"),
            "warnings": record.get("warnings", []),
        } for record in records],
        "unverified_samples": [record.get("source") for record in records
                               if not record.get("acceptance_evidence")],
        "design_patterns": aggregate_design(design, records),
    }
    if profile["pages_basis_mixed"]:
        profile.setdefault("warnings", []).append(
            "sample mixes page-count bases (" + ", ".join(bases) + "); the pages median "
            "compares different measurement bases and must be read with care")
    return profile


def aggregate_design(design, records):
    """Aggregate host-authored per-figure/table design notes into patterns."""
    if not design:
        return None
    require(isinstance(design, dict) and "papers" in design,
            "design file must be an object with a 'papers' member")
    papers = design["papers"]
    require(isinstance(papers, dict), "design.papers must be an object keyed by sample id")
    figure_types, table_types, placements, paper_total = {}, {}, {}, 0
    for sample_id, entry in papers.items():
        require(isinstance(entry, dict), f"design.papers[{sample_id}] must be an object")
        paper_total += 1
        for kind, counter in (("figures", figure_types), ("tables", table_types)):
            for item in entry.get(kind, []):
                require(isinstance(item, dict), f"design item in {sample_id}.{kind} must be an object")
                label = item.get("type")
                require(label, f"design item in {sample_id}.{kind} is missing 'type'")
                counter.setdefault(label, {"count": 0, "papers": 0, "examples": []})
                counter[label]["count"] += 1
                if sample_id not in counter[label]["examples"]:
                    counter[label]["examples"].append(sample_id)
                placement = item.get("placement")
                if placement:
                    require(placement in PLACEMENTS,
                            f"design placement must be one of {PLACEMENTS}, got {placement}")
                    placements[placement] = placements.get(placement, 0) + 1
    for counter in (figure_types, table_types):
        for label, entry in counter.items():
            entry["papers"] = len(entry["examples"])
            entry["paper_share"] = round(entry["papers"] / paper_total, 3) if paper_total else 0.0
    return {
        "sample_papers": paper_total,
        "figure_types": figure_types,
        "table_types": table_types,
        "placements": placements,
    }


def plan_items(plan):
    require(isinstance(plan, dict), "plan must be an object")
    items = plan.get("items")
    require(isinstance(items, list) and items, "plan.items must be a nonempty array")
    return items


def check_plan(profile, plan):
    items = plan_items(plan)
    errors, warnings = [], []
    identifiers = set()
    kinds = {"figure": [], "table": []}
    for index, item in enumerate(items):
        where = f"plan.items[{index}]"
        require(isinstance(item, dict), f"{where} must be an object")
        for field in ("id", "kind", "type", "question", "claim", "data", "section", "priority"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}: missing required field '{field}'")
        identifier = item.get("id")
        if identifier:
            if identifier in identifiers:
                errors.append(f"{where}: duplicate id '{identifier}'")
            identifiers.add(identifier)
        kind = item.get("kind")
        if kind in ("figure", "table"):
            kinds[kind].append(item)
        elif kind:
            errors.append(f"{where}: kind must be 'figure' or 'table', got '{kind}'")
        item_type = item.get("type")
        if kind == "figure" and item_type and item_type not in FIGURE_TYPES:
            errors.append(f"{where}: unknown figure type '{item_type}'; allowed: "
                          f"{', '.join(sorted(FIGURE_TYPES))}")
        if kind == "table" and item_type and item_type not in TABLE_TYPES:
            errors.append(f"{where}: unknown table type '{item_type}'; allowed: "
                          f"{', '.join(sorted(TABLE_TYPES))}")
        priority = item.get("priority")
        if priority and priority not in PRIORITIES:
            errors.append(f"{where}: priority must be one of {PRIORITIES}, got '{priority}'")
        question = (item.get("question") or "").strip()
        claim = (item.get("claim") or "").strip()
        if question and claim and question == claim:
            errors.append(f"{where}: 'question' and 'claim' must differ; a figure states a "
                          f"conclusion, not a restated question")
        data = (item.get("data") or "")
        if item.get("priority", "must") == "must" and ("TODO" in data or "待补" in data):
            warnings.append(f"{where}: must-have item still points at pending data")
    counts = {kind: len(values) for kind, values in kinds.items()}
    deviation_reason = (plan.get("deviation_reason") or "").strip()
    for kind, band in (("figure", profile.get("budget", {}).get("figures")),
                       ("table", profile.get("budget", {}).get("tables"))):
        if not band:
            continue
        count = counts[kind]
        if count < band["min"] or count > band["max"]:
            message = (f"planned {kind} count {count} is outside the venue band "
                       f"[{band['min']}, {band['max']}] (target {band['target']})")
            if deviation_reason:
                warnings.append(message + "; deviation_reason was supplied")
            else:
                errors.append(message + "; add plan.deviation_reason to justify it")
    patterns = profile.get("design_patterns") or {}
    for kind, key in (("figure", "figure_types"), ("table", "table_types")):
        for label, entry in (patterns.get(key) or {}).items():
            if entry.get("paper_share", 0) >= 0.5:
                if not any(item.get("type") == label for item in kinds[kind]):
                    message = (f"venue pattern '{label}' appears in {entry['paper_share']:.0%} of "
                               f"sampled papers but no planned {kind} covers it")
                    if deviation_reason:
                        warnings.append(message + "; deviation_reason was supplied")
                    else:
                        errors.append(message)
    return {
        "status": "fail" if errors else ("warn" if warnings else "pass"),
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
    }


def check_realize(plan, measured):
    items = plan_items(plan)
    plan_figures = [item for item in items if item.get("kind") == "figure"]
    plan_tables = [item for item in items if item.get("kind") == "table"]
    measured_figures = measured.get("counts", {}).get("figures", 0)
    measured_tables = measured.get("counts", {}).get("tables", 0)
    errors, warnings = [], []
    if measured_figures != len(plan_figures):
        errors.append(f"draft has {measured_figures} figures but the plan lists {len(plan_figures)}")
    if measured_tables != len(plan_tables):
        errors.append(f"draft has {measured_tables} tables but the plan lists {len(plan_tables)}")
    for item in plan_figures + plan_tables:
        if item.get("priority", "must") != "must":
            continue
        data = item.get("data") or ""
        if "TODO" in data or "待补" in data:
            errors.append(f"plan item {item.get('id')} still points at pending data: {data}")
    for warning in measured.get("warnings", []):
        warnings.append(warning)
    return {
        "status": "fail" if errors else ("warn" if warnings else "pass"),
        "measured": {"figures": measured_figures, "tables": measured_tables,
                     "pages_main_body": measured.get("pages_main_body"),
                     "pages_total": measured.get("pages_total")},
        "planned": {"figures": len(plan_figures), "tables": len(plan_tables)},
        "errors": errors,
        "warnings": warnings,
    }


def design_template(records):
    papers = {}
    for index, record in enumerate(records, 1):
        key = record.get("sample_id") or f"S{index:02d}"
        papers[key] = {
            "source": record.get("source"),
            "figures": [{"number": item["number"], "type": "", "claim": "",
                         "placement": "", "notes": ""} for item in record.get("figures", [])],
            "tables": [{"number": item["number"], "type": "", "claim": "",
                        "placement": "", "notes": ""} for item in record.get("tables", [])],
        }
    return {"version": VERSION, "papers": papers}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Python 3.9+. Measurements only; venue relevance and design judgement stay with the host.")
    commands = parser.add_subparsers(dest="command", required=True)

    measure = commands.add_parser("measure", help="Measure one accepted paper")
    measure.add_argument("--pdf", type=Path)
    measure.add_argument("--text", type=Path, help="Pre-extracted text file")
    measure.add_argument("--out", required=True, help="Output JSON path")

    profile = commands.add_parser("profile", help="Aggregate sample measurements into a venue profile")
    profile.add_argument("--measurements", required=True, type=Path)
    profile.add_argument("--design", type=Path, help="Host-authored per-figure/table design notes")
    profile.add_argument("--venue", required=True)
    profile.add_argument("--direction", required=True)
    profile.add_argument("--out", required=True)

    template = commands.add_parser("design-template", help="Emit a blank design-record template for the host")
    template.add_argument("--measurements", required=True, type=Path)
    template.add_argument("--out", required=True)

    check = commands.add_parser("check-plan", help="Validate this paper's figure/table plan against the venue profile")
    check.add_argument("--profile", required=True, type=Path)
    check.add_argument("--plan", required=True, type=Path)
    check.add_argument("--out")

    realize = commands.add_parser("check-realize", help="Check the compiled draft against the plan")
    realize.add_argument("--plan", required=True, type=Path)
    realize.add_argument("--pdf", type=Path)
    realize.add_argument("--text", type=Path)
    realize.add_argument("--out")

    args = parser.parse_args()
    try:
        if args.command == "measure":
            require(bool(args.pdf) != bool(args.text), "Pass exactly one of --pdf or --text")
            record = measure_pdf(args.pdf) if args.pdf else measure_text_file(args.text)
            record["measured_at"] = now()
            result = record
        elif args.command == "profile":
            measurements = read_json(args.measurements)
            records = measurements.get("samples") if isinstance(measurements, dict) else measurements
            require(isinstance(records, list) and records, "measurements file has no samples")
            design = read_json(args.design) if args.design else None
            result = build_profile(records, args.venue, args.direction, design)
        elif args.command == "design-template":
            measurements = read_json(args.measurements)
            records = measurements.get("samples") if isinstance(measurements, dict) else measurements
            require(isinstance(records, list) and records, "measurements file has no samples")
            result = design_template(records)
        elif args.command == "check-plan":
            profile_data = read_json(args.profile)
            plan = read_json(args.plan)
            result = check_plan(profile_data, plan)
        else:
            require(bool(args.pdf) != bool(args.text), "Pass exactly one of --pdf or --text")
            measured = measure_pdf(args.pdf) if args.pdf else measure_text_file(args.text)
            result = check_realize(read_json(args.plan), measured)
        payload = encoded(result)
        if getattr(args, "out", None):
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(payload)
        else:
            sys.stdout.write(payload.decode("utf-8"))
    except (VenueError, OSError, UnicodeError, subprocess.SubprocessError) as exc:
        print(f"venue_profile: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
