"""Recount the actual installed-development 20-second intervention."""
import json
from pathlib import Path
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
Q = RUN / "qualification/final-development-delay.v1"
OUT = PROJECT / "logs/inspection-final-development-delay.v1"
RUNTIME = RUN / "runtime/autoresearch-document-snapshots-v2"
sys.path.insert(0, str(PROJECT))
from eval.inspection_episode_receipts import canonical_digest, digest, durable_once
from eval.inspection_formal_admission import native_owner_reader
from eval.inspection_model_service import operation_directory
from eval.run_inspection_llm_episode import execute_development_llm_episode


def read(path):
    return json.loads(path.read_bytes())


def main(status):
    intent, result, plan = read(Q/"intent.json"), read(Q/"result.json"), read(OUT/"plan.json")
    assert intent["plan"] == plan and plan["injected_delay_seconds"] == 20 and plan["ablation"] is None
    assert plan["condition"] == "B4-M1" and "formal_dispatch" not in plan
    assert result["cache_reused"] is False and 1 <= result["actual_attempts"] <= 2

    def forbidden(*_):
        raise ValueError("offline read-only recount cannot dispatch")
    y, replay = execute_development_llm_episode(OUT, plan, forbidden, run=RUN,
        reservation_prefix=intent["reservation_prefix"], read_only=True)
    assert y == result["Y"] and replay["cache_reused"] is True
    assert all(result[k] == replay[k] for k in ("episode_receipt","episode_receipt_sha256","actual_attempts","reservation"))
    terminal = Path(replay["episode_receipt"])
    doc = read(terminal)
    assert doc["status"]=="completed" and doc["returncode"]==0 and doc["plan_sha256"]==canonical_digest(plan)
    attempt = terminal.parent
    world = read(attempt/"flight/world.json")
    assert world["simulated_seconds"] == 600 and world["physics_steps_executed"] == 144000
    assert world["clock_lag_over_0p1_seconds_ticks"] == 0 and world["max_clock_lag_seconds"] <= .1
    assert world["physics_runtime"]["errors"] == [] and doc["evidence"]["frozen_score_recount"]["score"] == world["score"]
    actor = [json.loads(line) for line in (attempt/"flight/actor.jsonl").read_text().splitlines()]
    injected = [r for r in actor if r["kind"] == "actual_model_response" and r["injected_delay_wall_seconds"] > 0]
    assert injected
    details = []
    for row in injected:
        assert row["injected_delay_requested_seconds"] == 20 and not row["final_after_flight"]
        before, after = row["worker_at_model_reply"][0], row["worker_after_response"][0]
        wait = row["injected_delay_wall_seconds"]
        assert wait >= 20 and after-before >= min(20,600-before)-.2
        details.append({"transport_ordinal":row["response"]["submitted_ordinal"],"reply_sim_time":before,
            "post_wait_sim_time":after,"sim_advance":after-before,"injected_wall":wait,
            "service_wall":row["response"].get("wall_seconds"),
            "client_wall":row["response"]["parent_end_to_end_wall_seconds"]})
    end = [r for r in actor if r["kind"]=="decoded_reply" and r["worker_progress_at_enqueue"][0]>=600]
    assert all(r["commands_enqueued_not_accepted"]==[] for r in end)
    owner = native_owner_reader(RUNTIME)(RUN)
    budget = read(status)["budget"]
    assert read(status)["stale"]==[] and budget["unknown_usage"]==budget["unresolved_usage"]==[]
    assert budget["reserved"]=={"calls":0,"tokens":0,"cost":{}}
    refs = [Q/"intent.json",Q/"result.json",Path(__file__).resolve(),OUT/"plan.json"]
    usage = {"calls":0,"tokens":0}
    for a in sorted(OUT.glob("attempt-*")):
        refs.extend(p for p in a.rglob("*") if p.is_file())
        binding=read(a/"intent.json")
        key=binding["reservation"]
        entry=owner["usage_ledger"][key]
        assert entry["status"]=="known" and entry["unreserved"] is False
        receipt=RUN/entry["receipt"]
        archive=receipt.parent/"complete-attempt.zip"
        assert digest(receipt)==entry["receipt_sha256"] and read(receipt)["usage"]==entry["reported_usage"]
        assert digest(archive)==read(receipt)["archive_sha256"]
        with zipfile.ZipFile(archive) as z: assert z.testzip() is None
        usage={k:usage[k]+entry["reported_usage"][k] for k in usage}
        refs.extend((receipt,archive,RUN/"resource-requests"/(key+".reserve.json"),RUN/"resource-requests"/(key+".settle.json")))
        refs.extend(p for p in operation_directory(key).rglob("*") if p.is_file())
    assert all(budget["known_usage"][k]-prior==usage[k] for k,prior in (("calls",2649),("tokens",17143094)))
    with (Q/"native-status.json").open("xb") as f:f.write(status.read_bytes())
    refs.append(Q/"native-status.json")
    value={"role":"one installed development wait arm; no formal dispatch, effect size or hardware claim",
        "Y":y,"score":{k:world["score"][k] for k in ("C","S","D","H","E","Y","SR")},
        "max_clock_seconds":world["max_clock_lag_seconds"],"steps":world["physics_steps_executed"],
        "injected":details,"after_finish_command_count":0,"actual_attempts":result["actual_attempts"],
        "known_usage":usage,"native_known_usage":budget["known_usage"],
        "files":{str(p.relative_to(PROJECT)):digest(p) for p in refs}}
    durable_once(Q/"binding.json",value)
    print(json.dumps({**{k:v for k,v in value.items() if k!="files"},
                      "binding_sha256":digest(Q/"binding.json")},allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
