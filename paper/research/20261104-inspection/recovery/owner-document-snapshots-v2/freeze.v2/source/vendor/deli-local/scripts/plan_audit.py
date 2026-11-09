#!/usr/bin/env python3
"""Audit whether a plan's commitments were actually delivered.

Mechanical part of the R6 plan-compliance review (references/plan-compliance.md).
This script does what can be decided mechanically and refuses to guess the rest:

  * every planned figure/table: is it present in the compiled draft, under the
    same number, and referenced in the running text;
  * every plan item whose data field is still pending: flagged;
  * the plan file against a stored snapshot: a plan edited after the fact is
    reported as plan_rewritten rather than silently compared against the
    version that already matches the draft.

It does NOT judge whether a deviation is reasonable, and it does NOT decide
whether a disclosure is adequate. Those are the reviewer's calls; the script
only supplies the evidence and the mechanical verdicts.

Requires Python 3.9+ standard library. PDF measurement needs poppler
(`pdftotext`, `pdfinfo`) on PATH, or a pre-extracted text file.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


FIGURE_RE = re.compile(r"Figure\s+(\d+)\s*\.\s*[A-Z]")
TABLE_RE = re.compile(r"Table\s+(\d+)\s*\.\s*[A-Z]")
# A caption in a two-column layout does not start its own line: the caption
# block is set at the end of a text line, after the preceding sentence.
CAPTION_CONTEXT = 240
REF_RE = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_tool(name):
    found = shutil.which(name)
    if not found:
        raise SystemExit(
            f"plan_audit: '{name}' not found on PATH. Install poppler "
            f"(pdftotext/pdfinfo), or pass --draft-text with pre-extracted text.")
    return found


def draft_text(pdf, text_path):
    if text_path is not None:
        return Path(text_path).read_text(encoding="utf-8", errors="replace")
    require_tool("pdftotext")
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"plan_audit: pdftotext failed for {pdf}")
    return out.stdout


def draft_labels(text):
    """Figure and table numbers that appear as captions in the draft.

    In a two-column layout a caption is set at the end of a text line, so the
    pattern is matched anywhere in the line. A caption begins with the label,
    a period, and a capitalised sentence; a body reference such as
    "Figure 2(b) is the operative result" does not match that shape and is
    therefore not counted as a caption.
    """
    figures = set()
    tables = set()
    for line in text.splitlines():
        for match in FIGURE_RE.finditer(line):
            figures.add(int(match.group(1)))
        for match in TABLE_RE.finditer(line):
            tables.add(int(match.group(1)))
    return figures, tables


def plan_items(plan):
    items = plan.get("items")
    if not isinstance(items, list):
        raise SystemExit("plan_audit: plan has no 'items' array")
    return items


def contract_items(contract):
    """Frozen contract items in the plan-item shape the audit already understands.

    The frozen contract is the approved baseline. Scoring against it means a
    later rewrite of the plan file cannot quietly become the thing that is
    audited, and an approved item that vanished from the plan is reported
    instead of disappearing from the comparison.

    `draft_label` carries the mapping from a promised item to the caption that
    actually delivers it. A promise whose ID was rotated after approval still
    matches the right figure or table instead of the one that merely shares
    its number.
    """
    items = []
    for item in contract.get("items", []):
        entry = {
            "id": item.get("id"),
            "kind": item.get("kind"),
            "type": item.get("kind"),
            "question": item.get("purpose", ""),
            "claim": item.get("claim", ""),
            "data": "; ".join(item.get("evidence_requirements", []) or item.get("required_content", [])),
            "section": item.get("section", ""),
            "priority": item.get("priority", "must"),
        }
        # Three states, which must not be conflated:
        #   key absent      -> match the caption by ID
        #   present, empty  -> no caption carries this promise
        #   present, filled -> that caption carries it
        if "draft_label" in item:
            entry["delivered_as"] = item["draft_label"] or ""
        items.append(entry)
    return items


def item_kind(item):
    kind = item.get("kind")
    if kind in ("figure", "table"):
        return kind
    return "content"


def label_of(item):
    """The number a planned figure/table should carry in the draft."""
    identifier = str(item.get("id", ""))
    match = re.search(r"(\d+)$", identifier)
    return int(match.group(1)) if match else None


def audit(plan, plan_path, snapshot, draft_text_value, figures, tables,
          draft_path, plan_hash):
    results = []
    delivered_figures = set()
    delivered_tables = set()

    for item in plan_items(plan):
        kind = item_kind(item)
        identifier = str(item.get("id", ""))
        entry = {
            "id": identifier,
            "kind": kind,
            "promised": item.get("claim") or item.get("question") or "",
            "section": item.get("section", ""),
            "priority": item.get("priority", "must"),
            "data": item.get("data", ""),
        }
        if kind in ("figure", "table"):
            if "delivered_as" in item:
                # The contract records which caption carries this promise. An
                # empty value is meaningful: it says no caption carries it, so
                # matching by ID would credit the wrong figure or table.
                delivered_as = item.get("delivered_as")
                match = re.search(r"(\d+)", str(delivered_as)) if delivered_as else None
                number = int(match.group(1)) if match else None
                explicit_none = not delivered_as
            else:
                delivered_as = None
                explicit_none = False
                number = label_of(item)
            if explicit_none:
                entry["delivered"] = "no draft caption carries this promise"
                entry["verdict"] = "undisclosed_deviation"
            elif number is None:
                # The plan item carries no number, so it cannot be matched to a
                # caption mechanically. Deciding whether it was delivered is a
                # reading task, not a counting task.
                entry["delivered"] = "plan item carries no figure/table number"
                entry["verdict"] = "requires_reviewer"
                entry["reviewer_question"] = ("Which figure or table in the draft "
                                              "delivers this item, if any?")
            else:
                present = number in (figures if kind == "figure" else tables)
                entry["delivered"] = (f"{kind} {number} present in the draft"
                                      if present else
                                      f"{kind} {number} not found among the draft captions")
                entry["verdict"] = "fulfilled" if present else "undisclosed_deviation"
                if present:
                    (delivered_figures if kind == "figure"
                     else delivered_tables).add(number)
        else:
            entry["delivered"] = "not mechanically checkable"
            entry["verdict"] = "requires_reviewer"
        data = item.get("data") or ""
        if "TODO" in data or "待补" in data:
            entry["pending_data"] = data
            if entry["verdict"] == "fulfilled":
                entry["verdict"] = "disclosed_deviation"
        results.append(entry)

    # Plans may declare visuals that are deliberately deferred. Deferring in the
    # plan is not the same as disclosing it in the draft, so these are handed
    # to the reviewer: only they can tell whether the draft says so too.
    for item in plan.get("pending_visuals", []) or []:
        kind = item_kind(item)
        number = label_of(item)
        present = (number in (figures if kind == "figure" else tables)
                   if number is not None else None)
        entry = {
            "id": str(item.get("id", "")),
            "kind": kind,
            "promised": item.get("claim") or item.get("question") or "",
            "section": item.get("section", ""),
            "priority": "deferred",
            "data": item.get("data", ""),
            "blocked_by": item.get("blocked_by", ""),
            "delivered": (f"{kind} {number} was in fact delivered" if present
                          else f"{kind} {number} deferred in the plan"
                          if number is not None
                          else "deferred in the plan, no number to match"),
            "verdict": "requires_reviewer",
            "reviewer_question": ("Is this deferral also disclosed in the draft? "
                                  "If not it is an undisclosed_deviation."),
        }
        results.append(entry)

    # Rewrites: compare the plan against the stored snapshot.
    rewrites = []
    if snapshot:
        snap = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        snap_hash = snap.get("sha256")
        snap_items = {str(i.get("id")): i for i in snap.get("items", [])}
        now_items = {str(i.get("id")): i for i in plan_items(plan)}
        if snap_hash and snap_hash != plan_hash:
            for identifier in sorted(set(snap_items) | set(now_items)):
                before = snap_items.get(identifier)
                after = now_items.get(identifier)
                if before is None:
                    rewrites.append({"id": identifier, "change": "added",
                                     "before": None, "after": after})
                elif after is None:
                    rewrites.append({"id": identifier, "change": "removed",
                                     "before": before, "after": None})
                elif before != after:
                    rewrites.append({"id": identifier, "change": "modified",
                                     "before": before, "after": after})
    return results, rewrites, delivered_figures, delivered_tables


# Numbers a plan states, in the forms that appear in prose. Scientific
# notation is rendered several ways, so each value is checked against a set of
# plausible renderings rather than one exact string. The mantissa of a value in
# scientific notation ("1.37e-05") is part of the value, so the pattern allows
# an exponent to follow.
NUMBER_RE = re.compile(r"(?<![\w.])(\d+\.\d+(?:[eE][-+]?\d+)?|\d+)(?![\w])")


def number_forms(value):
    """Renderings of one numeric value that a draft might use."""
    forms = {value}
    try:
        number = float(value)
    except ValueError:
        return forms
    if number == int(number):
        forms.add(str(int(number)))
    # 1.37e-5, 1.37 x 10^-5, 1.37*10^-5, and the bare mantissa, which is how a
    # draft often writes the same quantity in prose.
    mantissa, _, exponent = f"{number:.10g}".partition("e")
    if exponent:
        forms.add(f"{mantissa}e{exponent}")
        forms.add(f"{mantissa} x 10^{exponent}")
        forms.add(f"{mantissa} * 10^{exponent}")
        forms.add(mantissa)
        forms.add(f"{float(mantissa):g}")
    else:
        forms.add(f"{number:g}")
    return forms


def numeric_promises(plan):
    """Numeric values stated in the plan, with the item that states them.

    The same value may be stated by several items or in several fields; it is
    reported once per item and value so the reviewer sees which commitment
    carries it.
    """
    found = []
    seen = set()
    for item in plan_items(plan):
        identifier = str(item.get("id", ""))
        for field in ("claim", "data", "question"):
            text = str(item.get(field, ""))
            for match in NUMBER_RE.finditer(text):
                value = match.group(1)
                if len(value) < 2 and "." not in value:
                    # Bare small integers are far too common to be evidence of
                    # a specific commitment; skip them.
                    continue
                key = (identifier, value)
                if key in seen:
                    continue
                seen.add(key)
                found.append({"id": identifier, "value": value, "field": field})
    return found


def audit_numbers(plan, text):
    """Check that values promised numerically in the plan appear in the draft."""
    normalized = text.replace("\u2212", "-").replace("\u00d7", "x")
    results = []
    for promise in numeric_promises(plan):
        forms = number_forms(promise["value"])
        found = any(form in normalized for form in forms)
        results.append({
            "id": promise["id"],
            "value": promise["value"],
            "field": promise["field"],
            "found_in_draft": found,
            "verdict": "fulfilled" if found else "undisclosed_deviation",
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Python 3.9+. Mechanical evidence for the R6 plan-compliance review; "
               "the reviewer makes the final judgement.")
    sub = parser.add_subparsers(dest="command", required=True)

    audit_cmd = sub.add_parser("audit", help="Audit a plan against a compiled draft")
    audit_cmd.add_argument("--plan", required=True, type=Path)
    audit_cmd.add_argument("--plan-snapshot", type=Path,
                           help="Plan snapshot taken before the round started")
    audit_cmd.add_argument("--contract", type=Path,
                           help="Frozen deliverable contract; the audit is then scored against "
                                "the approved items rather than the current plan file")
    audit_cmd.add_argument("--draft", required=True, type=Path)
    audit_cmd.add_argument("--draft-text", type=Path,
                           help="Pre-extracted draft text, instead of reading the PDF")
    audit_cmd.add_argument("--out", type=Path)

    snap_cmd = sub.add_parser("snapshot", help="Store a plan snapshot for later comparison")
    snap_cmd.add_argument("--plan", required=True, type=Path)
    snap_cmd.add_argument("--out", required=True, type=Path)

    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            record = {
                "plan_path": str(args.plan),
                "sha256": sha256_file(args.plan),
                "captured_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
                "items": plan.get("items", []),
                "pending_visuals": plan.get("pending_visuals", []),
                "original_promise_set": plan.get("original_promise_set"),
                "plan_change_log": plan.get("plan_change_log", []),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            print(json.dumps({"status": "snapshot_written", "out": str(args.out),
                              "sha256": record["sha256"]}, ensure_ascii=False, indent=1))
            return 0

        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        plan_hash = sha256_file(args.plan)
        text = draft_text(args.draft, args.draft_text)
        figures, tables = draft_labels(text)
        baseline = plan
        contract_hash = None
        dropped_from_plan = []
        if args.contract:
            contract = json.loads(args.contract.read_text(encoding="utf-8"))
            contract_items_list = contract_items(contract)
            if not contract_items_list:
                raise SystemExit("plan_audit: the frozen contract has no items")
            contract_hash = contract.get("contract_hash")
            baseline = dict(plan)
            baseline["items"] = contract_items_list
            baseline["pending_visuals"] = []
            planned = {str(item.get("id")) for item in plan_items(plan)}
            dropped_from_plan = sorted(item["id"] for item in contract_items_list
                                       if item["id"] not in planned)
        results, rewrites, delivered_f, delivered_t = audit(
            baseline, args.plan, args.plan_snapshot, text, figures, tables,
            args.draft, plan_hash)
        numbers = audit_numbers(baseline, text)

        counts = {"promised": 0, "fulfilled": 0, "disclosed_deviation": 0,
                  "undisclosed_deviation": 0, "scope_expansion": 0,
                  "plan_rewritten": len(rewrites), "requires_reviewer": 0}
        for entry in results:
            counts["promised"] += 1
            counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1

        mechanically = counts["fulfilled"] + counts["disclosed_deviation"] \
            + counts["undisclosed_deviation"]
        reviewer_pending = counts.get("requires_reviewer", 0)
        # A pending item is neither kept nor broken, so the ratio is reported as
        # provisional: 1.0 over four settled items must not read as full
        # compliance when five more are still undecided.
        provisional = reviewer_pending > 0
        kept = round(counts["fulfilled"] / mechanically, 4) if mechanically else None

        # A plan that was narrowed after the fact to match the draft would score
        # a perfect ratio against its own items. If the plan records the promise
        # set it was written against, that set is the honest baseline, so it is
        # scored separately and reported alongside.
        baseline = None
        original = plan.get("original_promise_set")
        if isinstance(original, dict) and isinstance(original.get("items"), list):
            base_items = original["items"]
            base_delivered = 0
            base_settled = 0
            base_pending = 0
            base_rows = []
            for item in base_items:
                kind = item_kind(item)
                # A narrowed plan may have rotated ids, so an entry may record
                # which draft label actually carries the promise. Prefer that over
                # the id, which would otherwise match the wrong table.
                # An explicit delivered_as of null means the promise is NOT
                # carried by any draft label: it must not fall back to id
                # matching, or a rotated id matches the wrong table.
                if "delivered_as" in item:
                    delivered_as = item.get("delivered_as")
                    number = None
                    if delivered_as:
                        match = re.search(r"(\d+)", str(delivered_as))
                        number = int(match.group(1)) if match else None
                else:
                    delivered_as = None
                    number = label_of(item)
                present = (number in (figures if kind == "figure" else tables)
                           if number is not None else None)
                if present:
                    verdict = "fulfilled"
                    base_delivered += 1
                else:
                    verdict = "not_delivered"
                base_settled += 1
                base_rows.append({
                    "id": str(item.get("id", "")),
                    "kind": kind,
                    "promised": item.get("claim") or item.get("question") or "",
                    "section": item.get("section", ""),
                    "delivered_as": delivered_as,
                    "delivered": (f"{kind} {number} was in fact delivered" if present
                                  else f"{kind} {number} not delivered in the draft"
                                  if number is not None
                                  else "no draft label carries this promise"),
                    "verdict": verdict,
                })
            baseline = {
                "note": original.get("note", ""),
                "promised": base_settled,
                "delivered": base_delivered,
                "not_delivered": base_settled - base_delivered,
                "commitment_kept": round(base_delivered / base_settled, 4)
                if base_settled else None,
                "items": base_rows,
            }

        report = {
            "status": "audited",
            "plan": str(args.plan),
            "plan_sha256": plan_hash,
            "plan_snapshot": str(args.plan_snapshot) if args.plan_snapshot else None,
            "contract": str(args.contract) if args.contract else None,
            "contract_hash": contract_hash,
            "audited_against": "frozen_contract" if args.contract else "plan_file",
            "dropped_from_plan": dropped_from_plan,
            "draft": str(args.draft),
            "draft_figures": sorted(figures),
            "draft_tables": sorted(tables),
            "counts": counts,
            "commitment_kept": kept,
            "commitment_kept_is_provisional": provisional,
            "reviewer_pending": reviewer_pending,
            "original_promise_set": baseline,
            "plan_change_log": plan.get("plan_change_log", []),
            "items": results,
            "numeric_promises": numbers,
            "plan_rewrites": rewrites,
            "note": ("Mechanical evidence only. Whether a deviation is reasonable and "
                     "whether a disclosure is adequate are reviewer decisions, and the "
                     "plan may have been updated legitimately since the snapshot."),
        }
        payload = json.dumps(report, ensure_ascii=False, indent=1)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
            print(payload)
        else:
            print(payload)
        return 0
    except (OSError, ValueError, SystemExit) as exc:
        print(f"plan_audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
