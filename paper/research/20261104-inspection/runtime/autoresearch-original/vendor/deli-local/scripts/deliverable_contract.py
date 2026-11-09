#!/usr/bin/env python3
"""Frozen deliverable contract: Brief -> task decomposition -> execution -> acceptance.

This closes the loop that the plan-compliance review could only observe after the
fact. The approved Brief is turned into one frozen, itemized contract; every later
step is checked against that contract instead of against prose, counts, or whatever
the plan file happens to say this round.

Subcommands:

  freeze     Freeze the approved Brief into an itemized deliverable contract, and
             render the human-facing Brief section from that same structure.
  render     Re-render the human-facing Brief section from a frozen contract.
  coverage   Check a proposed task decomposition: every required item covered or
             explicitly blocked, tasks bound to the current contract, dispatchable
             work identified.
  bind       Bind a produced artifact to a contract item, with content hashes.
  review     Prepare the per-item semantic review request, then apply the review
             agent's per-item verdicts.
  branch     Record the explicit disposition of a failed or limited experiment.
  accept     Aggregate the mechanical checks and the review verdicts into one
             final acceptance report that names each blocking item.

The split of labour is deliberate. The program owns identity, binding, coverage,
hashes and arithmetic. The review agent owns the scientific question: does this
artifact answer the question the contract froze, are the comparison and the
statistics sound, does the conclusion overreach. A file called F5 existing on
disk is not completion.

Requires Python 3.9+ standard library.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VERSION = 1
KINDS = ("figure", "table", "section", "manuscript")
PRIORITIES = ("must", "should", "optional")
TASK_STATUS = ("planned", "running", "done", "failed", "blocked", "cancelled")
CHECK_VERDICTS = ("satisfied", "unmet", "blocked")
BRANCH_DECISIONS = ("rerun", "negative_result", "revise_brief")

CONTRACT_FILE = "state/deliverable_contract.json"
BRIEF_FILE = "state/deliverable_brief.md"
TASK_FILE = "state/task_plan.json"
BINDING_FILE = "state/deliverable_bindings.json"
REVIEW_FILE = "state/deliverable_reviews.json"
BRANCH_FILE = "state/deliverable_branches.json"
ACCEPT_FILE = "state/deliverable_acceptance.json"

# Severity order used when one item carries several problems.
SEVERITY = ("missing", "unmet", "blocked", "pending_review")


class ContractError(Exception):
    """A contract, binding or review input violates the documented contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


# ---------------------------------------------------------------- primitives


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractError(f"missing file: {path}")
    except ValueError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}")


def local_path(root, name):
    """A project-relative path that cannot escape the project."""
    text = str(name)
    require(text and not os.path.isabs(text), f"use a project-relative path: {name}")
    require(".." not in Path(text).parts, f"path escapes the project: {name}")
    return Path(root) / text


def write_text(path, text, force=False):
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    require(force or not path.exists(), f"output exists with different content: {path}; use --force")
    require(not path.is_symlink(), f"refusing to write through a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=".deli-contract-", delete=False)
    try:
        with handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)
    return True


def write_json(path, value, force=False):
    return write_text(path, json.dumps(value, ensure_ascii=False, indent=1) + "\n", force)


def project_file(root, name, label):
    path = local_path(root, name)
    require(path.is_file(), f"{label}: missing file {name}")
    return path


def items_hash(items):
    """Hash of the obligation set only, so metadata edits never invalidate bindings."""
    return sha_text(canonical(items))


def string_list(value, label, required=False):
    require(isinstance(value, list), f"{label}: expected an array")
    if required:
        require(bool(value), f"{label}: expected a nonempty array")
    for entry in value:
        require(isinstance(entry, str) and entry.strip(), f"{label}: entries must be nonempty strings")
    return value


# ---------------------------------------------------------------- contract


def validate_item(item, index):
    label = f"item[{index}]"
    require(isinstance(item, dict), f"{label}: expected an object")
    identifier = item.get("id")
    require(isinstance(identifier, str) and identifier.strip(), f"{label}: id must be a nonempty string")
    require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", identifier), f"{label}: id has unsupported characters")
    kind = item.get("kind")
    require(kind in KINDS, f"{label}: kind must be one of {', '.join(KINDS)}")
    title = item.get("title")
    require(isinstance(title, str) and title.strip(), f"{label}: title must be a nonempty string")
    purpose = item.get("purpose")
    require(isinstance(purpose, str) and len(purpose.strip()) >= 10,
            f"{label}: purpose must state the research question in at least 10 characters")
    claim = item.get("claim")
    require(isinstance(claim, str) and claim.strip(), f"{label}: claim must state the corresponding claim")
    criteria = string_list(item.get("acceptance_criteria"), f"{label}.acceptance_criteria", required=True)
    for field in ("required_content", "comparison", "measurement_requirements",
                  "evidence_requirements", "dependencies"):
        if field in item:
            string_list(item[field], f"{label}.{field}")
    priority = item.get("priority", "must")
    require(priority in PRIORITIES, f"{label}: priority must be one of {', '.join(PRIORITIES)}")
    plan_ref = item.get("plan_ref", "")
    require(not plan_ref or (isinstance(plan_ref, str) and plan_ref.strip()),
            f"{label}: plan_ref must be a nonempty string when present")
    normalized = {
        "id": identifier,
        "kind": kind,
        "title": title.strip(),
        "purpose": purpose.strip(),
        "claim": claim.strip(),
        "required_content": list(item.get("required_content", [])),
        "comparison": list(item.get("comparison", [])),
        "measurement_requirements": list(item.get("measurement_requirements", [])),
        "evidence_requirements": list(item.get("evidence_requirements", [])),
        "dependencies": list(item.get("dependencies", [])),
        "acceptance_criteria": list(criteria),
        "priority": priority,
        "section": item.get("section", ""),
        "plan_ref": plan_ref.strip(),
    }
    if "draft_label" in item:
        # Present-but-empty is meaningful: it records that no caption carries
        # this promise. Absent means the host has not decided yet, and the
        # caption is matched by ID. The two must not be conflated, or a
        # rotated promise matches the wrong figure or table.
        draft_label = item["draft_label"]
        require(draft_label is None or isinstance(draft_label, str),
                f"{label}: draft_label must be a string or null")
        label_text = (draft_label or "").strip()
        require(not label_text or re.fullmatch(r"(?:Figure|Table)\s+\d+", label_text),
                f"{label}: draft_label must look like 'Figure 2' or 'Table 1'")
        normalized["draft_label"] = label_text
    return normalized


def validate_items(raw):
    require(isinstance(raw, list) and raw, "the deliverable list must be a nonempty array")
    items = [validate_item(item, index) for index, item in enumerate(raw)]
    identifiers = [item["id"] for item in items]
    require(len(identifiers) == len(set(identifiers)), "deliverable IDs must be unique")
    return items


def load_contract(root):
    path = local_path(root, CONTRACT_FILE)
    require(path.is_file(), f"no frozen contract: {CONTRACT_FILE}; run freeze first")
    contract = read_json(path)
    require(contract.get("version") == VERSION, f"unsupported contract version: {contract.get('version')}")
    contract["items"] = validate_items(contract.get("items"))
    contract["contract_hash"] = items_hash(contract["items"])
    return contract


def render_brief(contract):
    """Render the human-facing Brief section from the frozen contract.

    The same structure that the user approves is the structure that execution is
    checked against, so the two cannot drift apart.
    """
    items = contract["items"]
    lines = [
        "## 交付物清单（逐项冻结）",
        "",
        f"合同版本 {contract.get('version')}，合同哈希 `{contract.get('contract_hash', '')[:16]}`。",
        "下表即执行契约：每一项都有稳定 ID、要回答的问题、对应主张、必要内容、比较对象、",
        "测量与证据要求、依赖和完成条件。后续任何一项未满足，验收都会指名它并阻止交付。",
        "",
        "| ID | 类型 | 标题 | 要回答的问题 | 对应主张 | 依赖 | 完成条件 | 优先级 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in items:
        criteria = "；".join(item["acceptance_criteria"])
        lines.append(
            f"| {item['id']} | {item['kind']} | {item['title']} | {item['purpose']} | {item['claim']} | "
            f"{'、'.join(item['dependencies']) or '—'} | {criteria} | {item['priority']} |")
    lines.append("")
    for item in items:
        lines.append(f"### {item['id']} — {item['title']}")
        lines.append("")
        lines.append(f"- 要回答的问题：{item['purpose']}")
        lines.append(f"- 对应主张：{item['claim']}")
        if item["required_content"]:
            lines.append("- 必要内容：" + "；".join(item["required_content"]))
        if item["comparison"]:
            lines.append("- 比较对象：" + "；".join(item["comparison"]))
        if item["measurement_requirements"]:
            lines.append("- 测量与证据要求：" + "；".join(item["measurement_requirements"]))
        if item["evidence_requirements"]:
            lines.append("- 证据要求：" + "；".join(item["evidence_requirements"]))
        if item["dependencies"]:
            lines.append("- 依赖：" + "、".join(item["dependencies"]))
        lines.append("- 完成条件：" + "；".join(item["acceptance_criteria"]))
        if item["section"]:
            lines.append(f"- 所在章节：{item['section']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- freeze


def cmd_freeze(args):
    root = Path(args.project).resolve()
    require(root.is_dir(), f"project directory not found: {root}")
    source = project_file(root, args.items, "the proposed deliverable list")
    raw = read_json(source)
    items = raw.get("items") if isinstance(raw, dict) else raw
    items = validate_items(items)

    brief_hash = ""
    brief_source = ""
    if args.brief:
        brief_path = project_file(root, args.brief, "the approved Brief")
        brief_hash = sha_file(brief_path)
        brief_source = args.brief

    previous = None
    previous_path = local_path(root, CONTRACT_FILE)
    if previous_path.is_file():
        previous = read_json(previous_path)
        require(args.revision_of is not None or previous.get("contract_hash") == items_hash(items),
                "the frozen contract already exists with different items; pass --revision-of to change it explicitly")

    revision = None
    if args.revision_of:
        require(previous is not None, "--revision-of needs an existing frozen contract")
        require(args.approved_by and args.approved_at and args.reason,
                "a contract revision needs --approved-by, --approved-at and --reason: reducing an "
                "approved deliverable or changing the research scope is an explicit, re-approved change")
        before = {item["id"]: item for item in previous.get("items", [])}
        after = {item["id"]: item for item in items}
        removed = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        changed = sorted(rid for rid in set(before) & set(after) if before[rid] != after[rid])
        revision = {
            "of_contract_hash": previous.get("contract_hash"),
            "approved_by": args.approved_by,
            "approved_at": args.approved_at,
            "reason": args.reason,
            "removed": removed,
            "added": added,
            "changed": changed,
        }
        require(not removed or args.reason.strip(),
                "removing an approved deliverable needs a stated reason")

    contract = {
        "version": VERSION,
        "contract_id": args.contract_id or (previous or {}).get("contract_id") or f"contract-{now()}",
        "created_at": (previous or {}).get("created_at") or now(),
        "frozen_at": now(),
        "brief_source": brief_source,
        "brief_sha256": brief_hash,
        "items_source": args.items,
        "items_source_sha256": sha_file(source),
        "venue": args.venue or (previous or {}).get("venue", ""),
        "paper_type": args.paper_type or (previous or {}).get("paper_type", ""),
        "items": items,
        "revisions": list((previous or {}).get("revisions", [])) + ([revision] if revision else []),
    }
    contract["contract_hash"] = items_hash(items)
    write_json(local_path(root, CONTRACT_FILE), contract, force=args.force)
    rendered = render_brief(contract)
    write_text(local_path(root, BRIEF_FILE), rendered, force=args.force)
    print(json.dumps({
        "status": "frozen",
        "contract": CONTRACT_FILE,
        "brief": BRIEF_FILE,
        "contract_hash": contract["contract_hash"],
        "items": len(items),
        "must_items": sum(1 for item in items if item["priority"] == "must"),
        "revision": revision,
    }, ensure_ascii=False, indent=1))
    return 0


def cmd_render(args):
    root = Path(args.project).resolve()
    contract = load_contract(root)
    rendered = render_brief(contract)
    if args.out:
        write_text(local_path(root, args.out), rendered, force=args.force)
        print(json.dumps({"status": "rendered", "out": args.out}, ensure_ascii=False))
    else:
        sys.stdout.write(rendered)
    return 0


# ---------------------------------------------------------------- coverage


def load_tasks(root):
    path = local_path(root, TASK_FILE)
    require(path.is_file(), f"no task decomposition: {TASK_FILE}")
    value = read_json(path)
    tasks = value.get("tasks") if isinstance(value, dict) else value
    require(isinstance(tasks, list) and tasks, "the task decomposition must be a nonempty array")
    known = set()
    for index, task in enumerate(tasks):
        label = f"task[{index}]"
        require(isinstance(task, dict), f"{label}: expected an object")
        identifier = task.get("id")
        require(isinstance(identifier, str) and identifier.strip(), f"{label}: id must be a nonempty string")
        require(identifier not in known, f"{label}: duplicate task ID {identifier}")
        known.add(identifier)
        string_list(task.get("requirement_ids", []), f"{label}.requirement_ids")
        string_list(task.get("depends_on", []), f"{label}.depends_on")
        string_list(task.get("blocked_by", []), f"{label}.blocked_by")
        status = task.get("status", "planned")
        require(status in TASK_STATUS, f"{label}: status must be one of {', '.join(TASK_STATUS)}")
    for task in tasks:
        for dependency in task.get("depends_on", []):
            require(dependency in known, f"task {task['id']} depends on unknown task {dependency}")
    return value, tasks


def cmd_coverage(args):
    root = Path(args.project).resolve()
    contract = load_contract(root)
    value, tasks = load_tasks(root)
    bound = value.get("contract_hash") if isinstance(value, dict) else None
    problems = []
    if bound and bound != contract["contract_hash"]:
        problems.append({
            "kind": "stale_decomposition",
            "message": "任务分解绑定的合同版本已过期，必须按当前合同重新分解",
            "expected": contract["contract_hash"], "found": bound})
    identifiers = {item["id"] for item in contract["items"]}
    for task in tasks:
        for requirement in task.get("requirement_ids", []):
            require(requirement in identifiers,
                    f"task {task['id']} refers to unknown contract item {requirement}")
    for item in contract["items"]:
        serving = [task for task in tasks if item["id"] in task.get("requirement_ids", [])]
        if not serving:
            problems.append({"kind": "uncovered", "id": item["id"],
                             "message": f"{item['id']} 缺失：没有任何任务覆盖该项"})
            continue
        blocked = [task for task in serving if task.get("status") in ("blocked", "failed")]
        if blocked and all(task.get("status") in ("blocked", "failed") for task in serving):
            blockers = sorted({value for task in blocked for value in task.get("blocked_by", [])})
            named = [value for value in blockers if value in item["dependencies"]] or blockers
            problems.append({
                "kind": "blocked", "id": item["id"],
                "message": f"{item['id']} 被 {'、'.join(named) or '未指明的依赖'} 阻塞"})
    dispatchable = []
    for task in tasks:
        if task.get("status") != "planned":
            continue
        done = {other["id"] for other in tasks if other.get("status") == "done"}
        if all(dependency in done for dependency in task.get("depends_on", [])):
            dispatchable.append(task["id"])
    report = {
        "status": "blocked" if problems else "ok",
        "contract_hash": contract["contract_hash"],
        "tasks": len(tasks),
        "items": len(contract["items"]),
        "coverage": [
            {"id": item["id"], "priority": item["priority"],
             "tasks": [task["id"] for task in tasks if item["id"] in task.get("requirement_ids", [])]}
            for item in contract["items"]],
        "dispatchable": dispatchable,
        "problems": problems,
    }
    if args.out:
        write_json(local_path(root, args.out), report, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if problems else 0


# ---------------------------------------------------------------- bindings


def load_bindings(root, contract):
    path = local_path(root, BINDING_FILE)
    if not path.is_file():
        return {}
    value = read_json(path)
    stored = value.get("contract_hash")
    if stored and stored != contract["contract_hash"]:
        return {}
    items = value.get("items", {})
    require(isinstance(items, dict), "invalid binding registry")
    return items


def cmd_bind(args):
    root = Path(args.project).resolve()
    contract = load_contract(root)
    item = next((entry for entry in contract["items"] if entry["id"] == args.requirement), None)
    require(item is not None, f"unknown contract item: {args.requirement}")
    artifact = project_file(root, args.artifact, "the artifact")
    evidence = []
    for name in args.evidence or []:
        evidence.append({"path": name, "sha256": sha_file(project_file(root, name, "evidence"))})
    entry = {
        "artifact": args.artifact,
        "artifact_sha256": sha_file(artifact),
        "manuscript_ref": args.manuscript_ref or "",
        "evidence": evidence,
        "bound_at": now(),
    }
    bindings = load_bindings(root, contract)
    bindings[args.requirement] = entry
    write_json(local_path(root, BINDING_FILE),
               {"contract_hash": contract["contract_hash"], "items": bindings}, force=True)
    print(json.dumps({"status": "bound", "requirement": args.requirement, **entry},
                     ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- review


def manuscript_files(root, manuscript_root):
    base = local_path(root, manuscript_root)
    require(base.is_dir(), f"manuscript root not found: {manuscript_root}")
    return sorted(base.rglob("*.tex"))


def find_reference(paths, label):
    """Return the .tex file that carries the label, or None.

    A label counts as present when the file either defines it (\\label) or cites
    it (\\ref and friends). A produced artifact that never reaches the manuscript
    is not a delivered artifact.
    """
    if not label:
        return None
    pattern = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref|label)\{" + re.escape(label) + r"\}")
    for path in paths:
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                return str(path)
        except OSError:
            continue
    return None


def load_reviews(root, contract):
    path = local_path(root, REVIEW_FILE)
    if not path.is_file():
        return {}
    value = read_json(path)
    stored = value.get("contract_hash")
    if stored and stored != contract["contract_hash"]:
        return {}
    reviews = value.get("reviews", {})
    require(isinstance(reviews, dict), "invalid review registry")
    return reviews


def cmd_review(args):
    root = Path(args.project).resolve()
    contract = load_contract(root)
    bindings = load_bindings(root, contract)
    reviews = load_reviews(root, contract)
    if args.apply:
        return apply_review(root, contract, args)
    items = []
    reusable = []
    not_ready = []
    for item in contract["items"]:
        binding = bindings.get(item["id"])
        if not binding:
            # An item with no bound artifact has nothing to review yet. It is
            # reported as not ready rather than pushed into the review request,
            # where a reviewer would have nothing to judge.
            not_ready.append({"requirement_id": item["id"], "status": "not_ready",
                              "reason": "该要求尚未绑定产物，无法评审"})
            continue
        existing = reviews.get(item["id"])
        current = binding.get("artifact_sha256")
        if existing and existing.get("artifact_sha256") == current and existing.get("verdict") == "satisfied":
            reusable.append(item["id"])
            continue
        items.append({
            "requirement_id": item["id"],
            "kind": item["kind"],
            "title": item["title"],
            "purpose": item["purpose"],
            "claim": item["claim"],
            "required_content": item["required_content"],
            "comparison": item["comparison"],
            "measurement_requirements": item["measurement_requirements"],
            "evidence_requirements": item["evidence_requirements"],
            "dependencies": item["dependencies"],
            "acceptance_criteria": item["acceptance_criteria"],
            "artifact": binding.get("artifact"),
            "artifact_sha256": current,
            "manuscript_ref": binding.get("manuscript_ref", ""),
            "evidence": binding.get("evidence", []),
            "previous_verdict": (existing or {}).get("verdict", ""),
        })
    request = {
        "version": VERSION,
        "contract_hash": contract["contract_hash"],
        "scope": "final" if args.final else "incremental",
        "items_to_review": items,
        "not_ready": not_ready,
        "reusable": reusable,
        "status": "needs_judgment",
        "instruction": (
            "逐项判断每个 requirement_id 是否真正满足它冻结下来的要求：产物是否回答了 purpose，"
            "是否包含 required_content，比较对象与统计是否成立，结论是否超出证据。"
            "证据不足时判 unmet 或 blocked，不要因为文件存在就判 satisfied。"
            "只返回 checks 与 cross_check，不复述材料原文。"),
    }
    request["request_id"] = sha_text(canonical(request))
    if args.out:
        write_json(local_path(root, args.out), request, force=args.force)
    print(json.dumps({"status": "request_written" if args.out else "request",
                      "request_id": request["request_id"], "out": args.out,
                      "to_review": len(items), "reusable": len(reusable),
                      "not_ready": len(not_ready)}, ensure_ascii=False, indent=1))
    return 0


def apply_review(root, contract, args):
    request = read_json(local_path(root, args.request) if args.request else args.request)
    decision = read_json(local_path(root, args.decision) if args.decision else args.decision)
    require(request.get("request_id") == decision.get("request_id"),
            "decision request_id does not match the request")
    require(request.get("contract_hash") == contract["contract_hash"],
            "the request was prepared against another contract version; prepare again")
    checks = decision.get("checks")
    require(isinstance(checks, list) and checks, "checks must be a nonempty array")
    requested = {entry["requirement_id"] for entry in request.get("items_to_review", [])}
    seen = set()
    for check in checks:
        require(isinstance(check, dict), "each check must be an object")
        identifier = check.get("requirement_id")
        require(identifier in requested, f"check refers to an item not requested: {identifier}")
        require(identifier not in seen, f"duplicate check for {identifier}")
        seen.add(identifier)
        verdict = check.get("verdict")
        require(verdict in CHECK_VERDICTS, f"invalid verdict for {identifier}: {verdict}")
        require(isinstance(check.get("explanation"), str) and check["explanation"].strip(),
                f"{identifier}: explanation must be a nonempty string")
        locations = string_list(check.get("evidence_locations"), f"{identifier}.evidence_locations", required=True)
        require(len(locations) >= 1, f"{identifier}: evidence_locations must be nonempty")
        artifact_hash = check.get("artifact_sha256", "")
        expected = next((entry.get("artifact_sha256") for entry in request.get("items_to_review", [])
                         if entry["requirement_id"] == identifier), "")
        require(not artifact_hash or not expected or artifact_hash == expected,
                f"{identifier}: the review was made against a different artifact revision")
    require(seen == requested,
            "the review must address every requested item exactly once: missing "
            + ", ".join(sorted(requested - seen)))
    if request.get("scope") == "final":
        cross = decision.get("cross_check")
        require(isinstance(cross, dict), "a final review needs a cross_check object")
        require(cross.get("verdict") in CHECK_VERDICTS, "cross_check.verdict is invalid")
        require(isinstance(cross.get("explanation"), str) and cross["explanation"].strip(),
                "cross_check.explanation must be a nonempty string")
    reviews = load_reviews(root, contract)
    for check in checks:
        identifier = check["requirement_id"]
        entry = next(item for item in request["items_to_review"] if item["requirement_id"] == identifier)
        reviews[identifier] = {
            "verdict": check["verdict"],
            "explanation": check["explanation"].strip(),
            "evidence_locations": check["evidence_locations"],
            "artifact_sha256": check.get("artifact_sha256", entry.get("artifact_sha256", "")),
            "reviewed_at": now(),
        }
    write_json(local_path(root, REVIEW_FILE),
               {"contract_hash": contract["contract_hash"], "reviews": reviews}, force=True)
    print(json.dumps({"status": "applied", "checks": len(checks), "reviews": len(reviews)},
                     ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- branches


def load_branches(root):
    path = local_path(root, BRANCH_FILE)
    if not path.is_file():
        return []
    value = read_json(path)
    branches = value.get("branches", value)
    require(isinstance(branches, list), "invalid branch registry")
    return branches


def bundle_digest(root):
    """Digest of everything an acceptance report depends on.

    An acceptance report is only valid for the contract, artifacts, evidence and
    review verdicts it was computed against. Recomputing this digest later tells
    a caller whether that report is still current, without trusting a flag.

    The artifact and evidence hashes are recomputed from the files on disk, not
    copied from the binding, so editing a file after the report was written makes
    the report stale.
    """
    root = Path(root)
    contract = load_contract(root)
    bindings = load_bindings(root, contract)
    reviews = load_reviews(root, contract)
    branches = load_branches(root)
    parts = {"contract": contract["contract_hash"], "items": []}
    for item in contract["items"]:
        binding = bindings.get(item["id"], {})
        artifact = local_path(root, binding["artifact"]) if binding.get("artifact") else None
        current_artifact = (sha_file(artifact) if artifact is not None and artifact.is_file()
                            else binding.get("artifact_sha256", ""))
        evidence = []
        for entry in binding.get("evidence", []):
            path = local_path(root, entry["path"])
            evidence.append(sha_file(path) if path.is_file() else entry.get("sha256", ""))
        review = reviews.get(item["id"], {})
        parts["items"].append({
            "id": item["id"],
            "artifact_sha256": current_artifact,
            "evidence": evidence,
            "manuscript_ref": binding.get("manuscript_ref", ""),
            "verdict": review.get("verdict", ""),
            "review_artifact_sha256": review.get("artifact_sha256", ""),
        })
    parts["branches"] = sorted(
        (branch.get("experiment", ""), branch.get("decision", ""),
         branch.get("followup_task", ""), branch.get("approved_at", ""))
        for branch in branches)
    return sha_text(canonical(parts))


def cmd_branch(args):
    root = Path(args.project).resolve()
    contract = load_contract(root)
    branches = load_branches(root)
    branch = {
        "experiment": args.experiment,
        "outcome": args.outcome,
        "decision": args.decision,
        "reason": args.reason,
        "followup_task": args.followup_task or "",
        "approved_by": args.approved_by or "",
        "approved_at": args.approved_at or "",
        "recorded_at": now(),
        "contract_hash": contract["contract_hash"],
    }
    require(args.decision in BRANCH_DECISIONS, f"decision must be one of {', '.join(BRANCH_DECISIONS)}")
    if args.decision == "rerun":
        require(args.followup_task, "a rerun decision needs --followup-task: the experiment is not done")
    if args.decision == "revise_brief":
        require(args.approved_by and args.approved_at and args.reason,
                "a brief revision needs --approved-by, --approved-at and --reason")
    branches = [entry for entry in branches if entry.get("experiment") != args.experiment]
    branches.append(branch)
    write_json(local_path(root, BRANCH_FILE), {"branches": branches}, force=True)
    print(json.dumps({"status": "recorded", **branch}, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- accept


def draft_text(path, text_path):
    if text_path is not None:
        return Path(text_path).read_text(encoding="utf-8", errors="replace")
    require(shutil.which("pdftotext"), "pdftotext not found on PATH; pass --draft-text instead")
    out = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True)
    require(out.returncode == 0, f"pdftotext failed for {path}")
    return out.stdout


def cmd_accept(args):
    root = Path(args.project).resolve()
    report = compute_report(root, args)
    if args.out:
        write_json(local_path(root, args.out), report, force=args.force)
    print(json.dumps({key: report[key] for key in
                      ("status", "deliverable", "counts", "must_total", "must_satisfied",
                       "commitment_kept", "blocking")}, ensure_ascii=False, indent=1))
    return 0 if report["deliverable"] else 1


def cmd_banner(args):
    """Render the draft banner that an unfinished manuscript must carry.

    An unfinished run may still produce a PDF for review, but the PDF has to say
    on its face that it is a draft and which promised items are still missing.
    The banner is derived from the same computation as the acceptance report, so
    the two cannot disagree.

    The banner is plain ASCII LaTeX so it compiles in an ordinary paper without
    extra packages; the item IDs are the payload, since the detail lives in the
    acceptance report.
    """
    root = Path(args.project).resolve()
    report = compute_report(root, args)
    require(not report["deliverable"], "the contract is satisfied: no draft banner is required")
    blocking = report["blocking"]
    item_ids = sorted({entry["id"] for entry in report["items"] if entry["verdict"] != "satisfied"})
    banner = (f"DELIVERABLE CONTRACT DRAFT - NOT FOR SUBMISSION. "
              f"Contract items satisfied: {report['must_satisfied']}/{report['must_total']}. "
              f"Outstanding items: {', '.join(item_ids) or 'none'}.")
    tex = ("\\begin{center}\\fbox{\\parbox{0.94\\linewidth}{\\small "
           "\\textbf{DRAFT --- NOT FOR SUBMISSION.}\\par "
           + banner.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
           + " This manuscript is incomplete against its approved deliverable contract; "
           "the outstanding items are listed above.}}\n\\end{center}\n")
    if args.out:
        write_text(local_path(root, args.out), tex, force=args.force)
    print(json.dumps({"status": "banner_written" if args.out else "banner",
                      "out": args.out, "banner": banner,
                      "outstanding_items": item_ids}, ensure_ascii=False, indent=1))
    return 0


def compute_report(root, args):
    contract = load_contract(root)
    bindings = load_bindings(root, contract)
    reviews = load_reviews(root, contract)
    branches = load_branches(root)
    problems = []

    plan_items = None
    if getattr(args, "plan", None):
        plan = read_json(project_file(root, args.plan, "the current plan file"))
        plan_items = plan.get("items", [])
        # A plan may carry deferred visuals outside 'items'. They are still
        # promises, so they count as present; only an ID that appears nowhere
        # has silently disappeared.
        planned = {str(entry.get("id")) for entry in plan_items}
        planned |= {str(entry.get("id")) for entry in plan.get("pending_visuals", []) or []}
        for item in contract["items"]:
            # A contract item may name the plan ID that carries it, which is how
            # a rotated or renamed ID stays traceable to its promise.
            candidates = {item["id"]}
            if item.get("plan_ref"):
                candidates.add(str(item["plan_ref"]))
            if not candidates & planned:
                problems.append({
                    "kind": "dropped_from_plan", "id": item["id"],
                    "message": f"{item['id']} 已在冻结合同中但不在当前计划文件中，"
                               "已批准目标不得在后续规划中无声消失"})

    manuscript_paths = []
    check_manuscript = bool(getattr(args, "manuscript_root", None))
    if check_manuscript:
        manuscript_paths = manuscript_files(root, args.manuscript_root)

    text = None
    if getattr(args, "draft_pdf", None) or getattr(args, "draft_text", None):
        source = args.draft_pdf or args.draft_text
        text = draft_text(source, args.draft_text)

    items = []
    counts = {"total": len(contract["items"]), "satisfied": 0, "missing": 0,
              "unmet": 0, "blocked": 0, "pending_review": 0}
    must_total = 0
    must_satisfied = 0

    for item in contract["items"]:
        item_problems = []
        binding = bindings.get(item["id"])
        if not binding:
            item_problems.append({"kind": "missing",
                                  "message": f"{item['id']} 缺失：没有绑定产物"})
        else:
            artifact = local_path(root, binding["artifact"])
            if not artifact.is_file():
                item_problems.append({"kind": "missing",
                                      "message": f"{item['id']} 缺失：绑定的产物文件不存在"})
            elif sha_file(artifact) != binding.get("artifact_sha256"):
                item_problems.append({"kind": "missing",
                                      "message": f"{item['id']} 缺失：绑定的产物内容已变更，需重新绑定"})
            for evidence in binding.get("evidence", []):
                evidence_path = local_path(root, evidence["path"])
                if not evidence_path.is_file():
                    item_problems.append({"kind": "missing",
                                          "message": f"{item['id']} 缺失：证据 {evidence['path']} 不存在"})
                elif sha_file(evidence_path) != evidence.get("sha256"):
                    item_problems.append({"kind": "missing",
                                          "message": f"{item['id']} 缺失：证据 {evidence['path']} 已变更"})
            reference = binding.get("manuscript_ref", "")
            # When a manuscript root was given, the reference must be found there.
            # An empty or unmatched manuscript is a missing artifact, not a
            # silently skipped check.
            if reference and check_manuscript:
                if find_reference(manuscript_paths, reference) is None:
                    item_problems.append({
                        "kind": "missing",
                        "message": f"{item['id']} 缺失：产物已生成但未插入稿件（找不到引用 {reference}）"})
        review = reviews.get(item["id"])
        if not review and binding:
            # Only an item that has an artifact can be waiting for a review.
            # An item with no artifact is already reported as missing.
            item_problems.append({"kind": "pending_review",
                                  "message": f"{item['id']} 待评审：缺少针对该要求的语义评审"})
        elif review:
            if binding and review.get("artifact_sha256") and review["artifact_sha256"] != binding.get("artifact_sha256"):
                item_problems.append({
                    "kind": "pending_review",
                    "message": f"{item['id']} 待评审：评审针对的是旧版本产物"})
            elif review.get("verdict") == "unmet":
                item_problems.append({"kind": "unmet",
                                      "message": f"{item['id']} 未满足要求：{review.get('explanation', '')}"})
            elif review.get("verdict") == "blocked":
                item_problems.append({"kind": "blocked",
                                      "message": f"{item['id']} 被阻塞：{review.get('explanation', '')}"})
        for branch in branches:
            if branch.get("decision") == "rerun" and branch.get("experiment") in item["dependencies"]:
                item_problems.append({
                    "kind": "blocked",
                    "message": f"{item['id']} 被 {branch['experiment']} 阻塞：该实验失败，已决定补做"
                               f"（后续任务 {branch.get('followup_task', '未指定')}）"})
            elif branch.get("decision") == "negative_result" and branch.get("experiment") in item["dependencies"]:
                item_problems.append({
                    "kind": "pending_review",
                    "message": f"{item['id']} 待评审：{branch['experiment']} 的负结果需按原完成条件评审"})
        kinds = [problem["kind"] for problem in item_problems]
        verdict = "satisfied"
        for kind in SEVERITY:
            if kind in kinds:
                verdict = kind
                break
        counts[verdict] = counts.get(verdict, 0) + 1
        if item["priority"] == "must":
            must_total += 1
            if verdict == "satisfied":
                must_satisfied += 1
        items.append({"id": item["id"], "kind": item["kind"], "priority": item["priority"],
                      "title": item["title"], "verdict": verdict, "problems": item_problems})
        problems.extend(item_problems)

    blocking = [problem["message"] for problem in problems]
    deliverable = not blocking
    kept = round(must_satisfied / must_total, 4) if must_total else None
    banner = ""
    if not deliverable:
        outstanding = sorted({entry["id"] for entry in items if entry["verdict"] != "satisfied"})
        banner = (f"DELIVERABLE CONTRACT DRAFT - NOT FOR SUBMISSION. "
                  f"Contract items satisfied: {must_satisfied}/{must_total}. "
                  f"Outstanding items: {', '.join(outstanding) or 'none'}.")
        if text is not None:
            require("DELIVERABLE CONTRACT DRAFT" in text,
                    "the draft PDF does not carry the draft marker and the blocking list")
    report = {
        "status": "passed" if deliverable else "blocked",
        "deliverable": deliverable,
        "contract_hash": contract["contract_hash"],
        "bundle_hash": bundle_digest(root),
        "generated_at": now(),
        "counts": counts,
        "must_total": must_total,
        "must_satisfied": must_satisfied,
        "commitment_kept": kept,
        "items": items,
        "blocking": blocking,
        "draft_banner": banner,
        "note": ("Mechanical checks and recorded review verdicts only. Whether the science is "
                 "correct remains the review agent's and the host's responsibility."),
    }
    return report


# ---------------------------------------------------------------- cli


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Python 3.9+. The program owns identity, binding, coverage and hashes; "
               "the review agent owns the scientific judgement.")
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="Freeze the approved Brief into an itemized contract")
    freeze.add_argument("--project", required=True)
    freeze.add_argument("--items", required=True, help="Project-relative JSON list of deliverable items")
    freeze.add_argument("--brief", help="Project-relative approved Brief, hashed into the contract")
    freeze.add_argument("--contract-id")
    freeze.add_argument("--venue")
    freeze.add_argument("--paper-type")
    freeze.add_argument("--revision-of", help="Explicitly revise the existing frozen contract")
    freeze.add_argument("--approved-by")
    freeze.add_argument("--approved-at")
    freeze.add_argument("--reason")
    freeze.add_argument("--force", action="store_true")
    freeze.set_defaults(run=cmd_freeze)

    render = commands.add_parser("render", help="Render the human-facing Brief section from the contract")
    render.add_argument("--project", required=True)
    render.add_argument("--out")
    render.add_argument("--force", action="store_true")
    render.set_defaults(run=cmd_render)

    coverage = commands.add_parser("coverage", help="Check a proposed task decomposition against the contract")
    coverage.add_argument("--project", required=True)
    coverage.add_argument("--out")
    coverage.add_argument("--force", action="store_true")
    coverage.set_defaults(run=cmd_coverage)

    bind = commands.add_parser("bind", help="Bind a produced artifact to a contract item")
    bind.add_argument("--project", required=True)
    bind.add_argument("--requirement", required=True)
    bind.add_argument("--artifact", required=True, help="Project-relative artifact path")
    bind.add_argument("--manuscript-ref", default="", help="LaTeX label of the manuscript reference")
    bind.add_argument("--evidence", action="append", help="Project-relative evidence path; repeatable")
    bind.add_argument("--force", action="store_true")
    bind.set_defaults(run=cmd_bind)

    review = commands.add_parser("review", help="Prepare the per-item semantic review, or apply its verdicts")
    review.add_argument("--project", required=True)
    review.add_argument("--out", help="Where to write the review request")
    review.add_argument("--final", action="store_true", help="Whole-draft cross-check scope")
    review.add_argument("--apply", action="store_true", help="Apply a review decision instead of preparing one")
    review.add_argument("--request")
    review.add_argument("--decision")
    review.add_argument("--force", action="store_true")
    review.set_defaults(run=cmd_review)

    branch = commands.add_parser("branch", help="Record the disposition of a failed or limited experiment")
    branch.add_argument("--project", required=True)
    branch.add_argument("--experiment", required=True)
    branch.add_argument("--outcome", required=True)
    branch.add_argument("--decision", required=True, choices=BRANCH_DECISIONS)
    branch.add_argument("--reason", default="")
    branch.add_argument("--followup-task", default="")
    branch.add_argument("--approved-by", default="")
    branch.add_argument("--approved-at", default="")
    branch.add_argument("--force", action="store_true")
    branch.set_defaults(run=cmd_branch)

    accept = commands.add_parser("accept", help="Aggregate mechanical checks and review verdicts")
    accept.add_argument("--project", required=True)
    accept.add_argument("--draft-pdf")
    accept.add_argument("--draft-text")
    accept.add_argument("--manuscript-root", help="Project-relative directory searched for manuscript references")
    accept.add_argument("--plan", help="Project-relative current plan file, checked against the contract")
    accept.add_argument("--out")
    accept.add_argument("--force", action="store_true")
    accept.set_defaults(run=cmd_accept)

    banner = commands.add_parser("banner", help="Render the draft banner an unfinished manuscript must carry")
    banner.add_argument("--project", required=True)
    banner.add_argument("--draft-pdf")
    banner.add_argument("--draft-text")
    banner.add_argument("--manuscript-root")
    banner.add_argument("--plan")
    banner.add_argument("--out")
    banner.add_argument("--force", action="store_true")
    banner.set_defaults(run=cmd_banner)

    args = parser.parse_args()
    try:
        return args.run(args)
    except (ContractError, OSError, UnicodeError, ValueError) as exc:
        print(f"deliverable_contract: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
