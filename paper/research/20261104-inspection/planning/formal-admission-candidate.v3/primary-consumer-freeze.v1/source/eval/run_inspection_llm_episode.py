"""One development LLM slot using the sole owner and immutable attempt receipts.

Known infrastructure failure permits one symmetric retry. Missing terminal or
model completion blocks replay; policy outcomes remain the original scored Y.
This module does not dispatch formal instances or approve G4.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from eval.inspection_episode_receipts import UnresolvedEpisode, canonical_digest, digest, durable_once, validate_flight
from eval.inspection_llm_episode_receipts import read, validate_llm_episode
from eval.inspection_model_receipts import validate_model_operation
from eval.inspection_model_service import operation_directory
from eval.inspection_process_cleanup import SubprocessTree, stop_worker_tree

ROOT = Path(__file__).resolve().parents[1]


def owner_action(run, action, packet, operation_id, log, reservation=None):
    """Use the existing native CLI, never write its .paper state ourselves."""
    cli = Path(run)/"runtime/autoresearch-state-reader-v1/scripts/research.py"
    argv = [sys.executable, str(cli), "--run", str(Path(run).resolve()), action,
        "--packet", str(Path(packet).resolve()), "--operation-id", operation_id]
    if reservation is not None:
        argv += ["--reservation", reservation]
    with Path(log).open("xb") as stream:
        process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    if process.returncode != 0:
        raise UnresolvedEpisode("native owner refused or failed; preserve its transport: "+str(log))
    return json.loads(Path(log).read_text())


def _stop(process):
    # Reuse the birth-checked descendant inventory, including a service which
    # owns a separate process group. Do not invent guarantees for host loss.
    class Child:
        pid = process.pid
        def is_alive(self): return process.poll() is None
        def terminate(self): process.terminate()
        def kill(self): process.kill()
        def join(self, timeout):
            try: process.wait(timeout=timeout)
            except subprocess.TimeoutExpired: pass
    return stop_worker_tree(Child())


def _expected_meter(plan, reservation):
    return {**({"formal_dispatch_binding": _formal(plan).binding()} if "formal_dispatch" in plan else {}),
        "reservation": reservation, **{k: plan[k] for k in
        ("owner_frozen_plan", "model_receipt_sha256", "generation_profile", "service_sha256")},
        "reserved": plan["reservation_usage"]}


def _formal(plan):
    from eval.inspection_formal_dispatch import admit_episode_plan
    return admit_episode_plan(plan, ROOT)


def _bindings(attempt, operation):
    paths = [p for p in attempt.rglob("*") if p.is_file() and p != attempt/"terminal.json"]
    return {str(p.relative_to(attempt)): digest(p) for p in paths}, {
        str(p.relative_to(operation)): digest(p) for p in operation.rglob("*") if p.is_file()}


def execute_development_llm_episode(directory, plan, command_factory, *, run, reservation_prefix, timeout_seconds=900,
                                    call_owner=owner_action):
    """Run/reconstruct one slot. An unresolved attempt never starts another."""
    if (plan["condition"] not in ("B3-M1", "B3-M2", "B3-M3", "B4-M1", "B4-M2")
            or type(plan["repeat_ordinal"]) is not int or plan["repeat_ordinal"] not in (0, 1, 2)
            or plan["max_attempts"] != 2 or type(plan["max_attempts"]) is not int
            or "formal_dispatch" not in plan and not plan["case_id"].startswith(("dev-", "devext-"))
            or plan["max_clock_lag_seconds"] != .1):
        raise ValueError("only fixed development slots and the two-attempt contract are admitted")
    if "formal_dispatch" in plan:
        _formal(plan)
        if Path(run).resolve() != Path(plan["formal_dispatch"]["run"]).resolve():
            raise ValueError("formal reservation owner differs from its dispatch")
    directory, run = Path(directory), Path(run).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan_hash = canonical_digest(plan)
    with (directory/".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise UnresolvedEpisode("another coordinator owns this LLM slot") from error
        if (directory/"plan.json").exists():
            if canonical_digest(read(directory/"plan.json")) != plan_hash:
                raise UnresolvedEpisode("LLM slot identity changed")
        else:
            durable_once(directory/"plan.json", plan)
        for ordinal in range(2):
            formal = _formal(plan) if "formal_dispatch" in plan else None
            attempt = directory/f"attempt-{ordinal:02d}"
            reservation = reservation_prefix+f"-a{ordinal}"
            operation = operation_directory(reservation)
            resource = run/"qualification/local-model-episodes"/reservation
            settled_paths = (resource/"complete-attempt.zip", resource/"usage-receipt.json",
                run/"resource-requests"/(reservation+".settle.json"))
            if attempt.exists():
                if not (attempt/"terminal.json").exists():
                    raise UnresolvedEpisode("unknown attempt requires source-bound reconciliation")
                terminal = read(attempt/"terminal.json")
                try:
                    if set(terminal.get("settlement_file_hashes", {})) != {str(p.resolve()) for p in settled_paths} or any(
                            not Path(p).resolve().is_relative_to(run) or digest(p) != h
                            for p, h in terminal["settlement_file_hashes"].items()):
                        raise ValueError("settled evidence binding differs")
                except (OSError, ValueError) as error:
                    raise UnresolvedEpisode("settled archive/usage/packet changed or missing") from error
                if (terminal.get("status") not in ("completed", "known_operational_failure")
                        or terminal.get("status") == "known_operational_failure" and terminal.get("retry_allowed") is not True):
                    raise UnresolvedEpisode("unknown or nonretryable LLM attempt blocks replay")
                try:
                    intent, started, returned, tree_proof = [read(attempt/name) for name in
                        ("intent.json", "started.json", "process-return.json", "process-tree.json")]
                    if (intent["plan_sha256"] != plan_hash or intent["reservation"] != reservation
                            or intent["attempt_ordinal"] != ordinal or not isinstance(intent["argv"], list)
                            or started["plan_sha256"] != plan_hash or type(started["pid"]) is not int or started["pid"] <= 0
                            or returned["plan_sha256"] != plan_hash or returned["returncode"] != terminal["returncode"]
                            or returned["stdout_sha256"] != digest(attempt/"stdout.log")
                            or tree_proof["status"] != "closed" or tree_proof["root_pid"] != started["pid"]
                            or not (attempt/"reserve-owner.json").is_file() or not (attempt/"settle-owner.json").is_file()):
                        raise ValueError("attempt start/return/cleanup proof differs")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    raise UnresolvedEpisode("LLM attempt dispatch/return proof is missing or changed") from error
                payload, model_payload = _bindings(attempt, operation)
                if (terminal["plan_sha256"] != plan_hash or terminal["payload_hashes"] != payload
                        or terminal["model_payload_hashes"] != model_payload):
                    raise UnresolvedEpisode("LLM terminal/input payload changed")
                if terminal["status"] == "completed":
                    y, evidence = validate_llm_episode(attempt/"flight", operation, plan, reservation, terminal["returncode"])
                    if terminal["Y"] != y or terminal["evidence"] != evidence:
                        raise UnresolvedEpisode("LLM cached episode differs")
                    if formal is not None:
                        from eval.inspection_formal_dispatch import admit_model_settlement
                        admit_model_settlement(plan, ROOT, reservation, ordinal, evidence["model_meter"])
                    return y, {"episode_receipt": str((attempt/"terminal.json").resolve()),
                        "episode_receipt_sha256": digest(attempt/"terminal.json"), "cache_reused": True,
                        "actual_attempts": ordinal+1, "plan_sha256": plan_hash, "reservation": reservation}
                if terminal["status"] != "known_operational_failure" or terminal["retry_allowed"] is not True:
                    raise UnresolvedEpisode("unknown or nonretryable LLM attempt blocks replay")
                _, failed_llm = validate_llm_episode(attempt/"flight", operation, plan, reservation, terminal["returncode"],
                    audit_operational_failure=True)
                if formal is not None:
                    from eval.inspection_formal_dispatch import admit_model_settlement
                    admit_model_settlement(plan, ROOT, reservation, ordinal, failed_llm["model_meter"])
                failed_y, failed_evidence = validate_flight(attempt/"flight", plan, terminal["returncode"])
                if (failed_y is not None or failed_evidence != terminal["physical_validation"]
                        or not set(failed_evidence["failure_reasons"]) <= {
                            "worker_operational_failure", "incomplete_or_wrong_clock", "clock_lag_exceeded"}):
                    raise UnresolvedEpisode("cached retry is not a known infrastructure failure")
                continue
            # No reservation or child process is created after a source fault.
            if any(digest(p) != h for group in ("source_hashes", "dependency_hashes") for p, h in plan[group].items()):
                raise UnresolvedEpisode("pre-dispatch frozen source/dependency differs")
            if digest(plan["case_path"]) != plan["case_file_sha256"]:
                raise UnresolvedEpisode("pre-dispatch private input differs")
            attempt.mkdir()
            reserve_packet = run/"resource-requests"/(reservation+".reserve.json")
            purpose = {"kind": "inspection_local_generation", "phase": "development" if formal is None else formal.binding()["phase"],
                "model_registry_sha256": plan["model_registry_sha256"], "model_receipt_sha256": plan["model_receipt_sha256"],
                "condition": plan["condition"], "case_id": plan["case_id"], "repeat_ordinal": plan["repeat_ordinal"],
                "episode_plan_sha256": plan_hash, "attempt_ordinal": ordinal}
            if formal is not None:
                purpose["formal_dispatch"] = plan["formal_dispatch"]["packet"]
                purpose["formal_dispatch_binding"] = formal.binding()
            durable_once(reserve_packet, {"usage": plan["reservation_usage"], "purpose": json.dumps(purpose, sort_keys=True)})
            call_owner(run, "budget-reserve", reserve_packet, reservation, attempt/"reserve-owner.json")
            command = command_factory(attempt/"flight", reservation)
            durable_once(attempt/"intent.json", {"role": "intent, not proof of dispatch/completion", "argv": command,
                "reservation": reservation, "plan_sha256": plan_hash, "attempt_ordinal": ordinal})
            tree, process, cleanup_evidence = SubprocessTree(), None, None
            with (attempt/"stdout.log").open("xb") as stream:
                # Spawn uncertainty is kept unresolved; it is never a scored0.
                try:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    tree.attach(process)
                    durable_once(attempt/"started.json", {"pid": process.pid, "plan_sha256": plan_hash,
                        "at": datetime.now(timezone.utc).isoformat()})
                    code = tree.wait(timeout_seconds)
                finally:
                    if process is not None:
                        try:
                            if process.pid not in tree.owned and process.poll() is None:
                                _stop(process)
                            cleanup_evidence = tree.close()
                        except BaseException as error:
                            cleanup_evidence = {"status": "unknown", "error": repr(error)}
                            if process.poll() is None:
                                _stop(process)
                        durable_once(attempt/"process-tree.json", cleanup_evidence)
            durable_once(attempt/"process-return.json", {"returncode": code, "plan_sha256": plan_hash,
                "stdout_sha256": digest(attempt/"stdout.log")})
            meter = None
            meter_error = None
            try:
                meter = validate_model_operation(operation, _expected_meter(plan, reservation))
            except (OSError, ValueError, KeyError, TypeError) as error:
                meter_error = repr(error)
            # Preserve even a failed/unknown operation before settling. No data
            # file referenced by a settled usage receipt is overwritten later.
            resource.mkdir(parents=True, exist_ok=False)
            archive = resource/"complete-attempt.zip"
            archive_files = {"episode-plan.json": directory/"plan.json"}
            for label, base in (("attempt", attempt), ("model-operation", operation)):
                archive_files.update({label+"/"+str(p.relative_to(base)): p for p in base.rglob("*") if p.is_file()})
            if any(p.is_symlink() for p in archive_files.values()):
                raise UnresolvedEpisode("linked episode archive entry")
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
                for name, path in archive_files.items(): z.write(path, name)
                z.writestr("manifest.json", json.dumps({n: digest(p) for n, p in archive_files.items()}, sort_keys=True))
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise UnresolvedEpisode("episode archive failed integrity check")
            receipt = resource/"usage-receipt.json"
            durable_once(receipt, {"role": "local attempt resource receipt; not a scored admission",
                "reservation": reservation, "plan_sha256": plan_hash, "resource_recount": meter,
                "usage": meter["usage"] if meter is not None else {}, "unknown_reason": meter_error,
                "archive": str(archive.relative_to(run)), "archive_sha256": digest(archive),
                "capital_energy_cost_unmetered": True, "paid_API_used": False})
            settle_packet = run/"resource-requests"/(reservation+".settle.json")
            durable_once(settle_packet, {"receipt": str(receipt.relative_to(run)),
                "status": "known" if meter is not None else "unknown", "usage": meter["usage"] if meter is not None else {}})
            call_owner(run, "budget-settle", settle_packet, "settle-"+reservation, attempt/"settle-owner.json", reservation)
            if formal is not None and meter is not None:
                from eval.inspection_formal_dispatch import admit_model_settlement
                admit_model_settlement(plan, ROOT, reservation, ordinal, meter)
            y, evidence = None, None
            value, physical = validate_flight(attempt/"flight", plan, code)
            retry = False
            admission_error = meter_error
            if meter is not None:
                try:
                    y, evidence = validate_llm_episode(attempt/"flight", operation, plan, reservation, code,
                        audit_operational_failure=True)
                    retry = y is None
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                    admission_error = repr(error)
            if cleanup_evidence["status"] != "closed":
                y, retry, admission_error = None, False, "process cleanup requires source-bound reconciliation"
            payload, model_payload = _bindings(attempt, operation)
            durable_once(attempt/"terminal.json", {"status": "completed" if y is not None else
                "unknown_process_cleanup" if cleanup_evidence["status"] != "closed" else
                "known_operational_failure" if meter is not None else "unknown_model_usage",
                "plan_sha256": plan_hash, "returncode": code, "Y": y, "evidence": evidence,
                "physical_validation": physical, "admission_error": admission_error, "retry_allowed": retry,
                "payload_hashes": payload, "model_payload_hashes": model_payload,
                "settlement_file_hashes": {str(p.resolve()): digest(p) for p in (archive, receipt, settle_packet)}})
            if y is not None:
                return y, {"episode_receipt": str((attempt/"terminal.json").resolve()),
                    "episode_receipt_sha256": digest(attempt/"terminal.json"), "cache_reused": False,
                    "actual_attempts": ordinal+1, "plan_sha256": plan_hash, "reservation": reservation}
            if not retry:
                raise UnresolvedEpisode("unknown or source/input/score/causal fault blocks this LLM slot")
        raise UnresolvedEpisode("finite symmetric infrastructure retry budget exhausted")
