"""Read-only facts for exact original slot030; no recovery or redispatch."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

RUN = Path(__file__).resolve().parents[3]
PROJECT = RUN.parents[2]
BUNDLE = Path(__file__).resolve().parents[1]
PREVIOUS = RUN / "recovery/pilot-v4-cleanup-024.v1/source/continue_pilot.py"


def main():
    import hashlib
    assert hashlib.sha256(PREVIOUS.read_bytes()).hexdigest() == "bb196f6c6939d9c050eb56f1b435eee68264347093b1cf85f0556cae12101c54"
    spec = importlib.util.spec_from_file_location("_inspection_old_cleanup_reader", PREVIOUS)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    episode = PROJECT / "logs/inspection-eight-condition-dev-flights.v4/dev-L3-01-B4-M1-1"
    attempt = episode / "attempt-00"
    reservation = "dev-all8-pilot-v4-030-a0"
    plan, terminal, tree = [helper.read(p) for p in (
        episode / "plan.json", attempt / "terminal.json", attempt / "process-tree.json")]
    assert plan["case_id"] == "dev-L3-01" and plan["condition"] == "B4-M1" and plan["repeat_ordinal"] == 1
    assert terminal["plan_sha256"] == helper.canonical_digest(plan)
    assert terminal["status"] == "unknown_process_cleanup" and terminal["Y"] is None
    assert terminal["retry_allowed"] is False and terminal["returncode"] == 0
    assert tree["status"] == "unknown" and tree["root_pid"] == 81848 and tree["remaining_owned"] == []
    assert tree["unclaimed_new_python_orphans"] == [86715]
    assert set(tree["owned_births"]) == {"81848", "81850", "81851", "81968", "81979"}
    assert not (episode / "attempt-01").exists()
    refs = helper.live_frozen_sources()
    refs.extend(helper.checked(p) for p in (PREVIOUS, Path(__file__), episode / "plan.json", attempt / "terminal.json", attempt / "process-tree.json"))
    operation = helper.operation_directory(reservation)
    payload, model_payload = helper._bindings(attempt, operation)
    assert payload == terminal["payload_hashes"] and model_payload == terminal["model_payload_hashes"]
    refs.extend(helper.checked(attempt / name, sha) for name, sha in payload.items())
    refs.extend(helper.checked(operation / name, sha) for name, sha in model_payload.items())
    refs.extend(helper.checked(name, sha) for name, sha in terminal["settlement_file_hashes"].items())
    resource = RUN / "qualification/local-model-episodes" / reservation
    with zipfile.ZipFile(resource / "complete-attempt.zip") as archive:
        assert archive.testzip() is None
    y, evidence = helper.validate_llm_episode(attempt / "flight", operation, plan, reservation, 0)
    assert y == 0.3436623024638006 and evidence == terminal["evidence"]
    assert evidence["model_meter"]["usage"] == {"calls": 24, "tokens": 173756}
    owner = helper.owner()
    entry = owner["usage_ledger"][reservation]
    assert entry["status"] == "known" and entry["reported_usage"] == {"calls": 24, "tokens": 173756}
    assert entry["receipt_sha256"] == helper.digest(resource / "usage-receipt.json")
    assert owner["frozen_plan"] == plan["owner_frozen_plan"] and owner["task_stop"] is None
    purpose = json.loads(entry["purpose"])
    assert purpose["episode_plan_sha256"] == helper.canonical_digest(plan) and purpose["attempt_ordinal"] == 0
    assert (purpose["case_id"], purpose["condition"], purpose["repeat_ordinal"]) == ("dev-L3-01", "B4-M1", 1)
    processes = helper._inventory()
    still_live = {pid: processes[int(pid)] for pid, old in tree["owned_births"].items()
                  if int(pid) in processes and processes[int(pid)]["born"] == old["born"]
                  and not processes[int(pid)]["status"].startswith("Z")}
    orphans = {str(pid): row for pid, row in processes.items()
               if row["uid"] == helper.os.getuid() and row["parent"] == 1
               and "python" in row["command"].lower() and not row["status"].startswith("Z")}
    observation = {"at": datetime.now(timezone.utc).isoformat(), "remaining_original_live": still_live,
                   "current_same_user_python_orphans": orphans, "original_unclaimed_pid_present": 86715 in processes,
                   "scope": "later observation only; original86715birth/ownership unavailable; not original closed"}
    assert helper._bindings(attempt, operation) == (payload, model_payload)
    for ref in refs:
        helper.checked(PROJECT / ref["path"], ref["sha256"])
    out = BUNDLE / "evidence.v1"
    out.mkdir()
    captures = []
    for source, name in [(attempt / "terminal.json", "original-terminal.json"), (attempt / "process-tree.json", "original-process-tree.json"),
                         (Path("/tmp/inspection-v4-second-cleanup-native-status.v1.json"), "native-status.json"),
                         (Path("/tmp/inspection-v4-second-cleanup-process-observation.v1.txt"), "process-observation.txt"),
                         (Path("/tmp/inspection-pilot-v4-continuation.v1.stdout"), "aborted-continuation.stdout")]:
        target = out / name
        shutil.copyfile(source, target)
        captures.append(helper.checked(target))
    result = {"role": "facts only; no late-cleanup acceptance or continuation", "target_order_index": 30,
              "plan_canonical_sha256": helper.canonical_digest(plan), "Y_before_cleanup_override": y,
              "original_status": terminal["status"], "model_usage": evidence["model_meter"]["usage"],
              "causal_joins": len(evidence["causal_request_joins"]), "evidence_canonical_sha256": helper.canonical_digest(evidence),
              "later_observation": observation, "new_resource_operations": 0, "files": refs, "captures": captures}
    (out / "binding.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("role", "target_order_index", "plan_canonical_sha256", "Y_before_cleanup_override", "model_usage", "causal_joins", "later_observation")}, ensure_ascii=False))
    print("binding_sha256", helper.digest(out / "binding.json"))


if __name__ == "__main__":
    main()
