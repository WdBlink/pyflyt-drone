"""Recount the installed clean flight and the previously measured event sibling."""
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
sys.path.insert(0, str(PROJECT))

from eval.inspection_common_calendar import rescore_common_calendar
from eval.inspection_episode_receipts import digest, durable_once, execute_local_episode
from eval.inspection_event_endpoints import analyze_event_trace
from eval.inspection_population import PopulationCase, event_clean_sibling, load_development_case

PAIR = RUN / "qualification/dev-same-level-event-pair.v2"
CLEAN = PROJECT / "logs/inspection-final-development-clean.v1/devext-OOD-L3-00-B0-0"
OUT = RUN / "qualification/final-development-clean.v1"


def read(path):
    return json.loads(path.read_bytes())


def main(stdout):
    def forbidden(_):
        raise ValueError("offline evidence recount cannot start a flight")

    out = OUT
    assert (out/"intent.json").is_file() and not (out/"binding.json").exists()
    pair = read(PAIR/"pair.json")
    assert digest(PAIR/"pair.json") == "6ff386e880379d9b88e3382bb6af9714e1365c0eb30e187f5bc7a6819253ac84"
    event_path = Path(pair["flights"]["event"]["root"]) / "attempt-00/flight/world.json"
    assert digest(event_path) == pair["flights"]["event"]["world_sha256"]
    plan = read(CLEAN/"plan.json")
    y, receipt = execute_local_episode(CLEAN, plan, forbidden)
    returned = json.loads(stdout.read_text().splitlines()[-1])
    assert returned["Y"] == y and returned["cache_reused"] is False
    assert receipt["episode_receipt_sha256"] == returned["episode_receipt_sha256"]
    terminal = Path(receipt["episode_receipt"])
    clean_path = terminal.parent/"flight/world.json"
    world = read(clean_path)
    assert world["simulated_seconds"] == 600 and world["physics_steps_executed"] == 144000
    assert world["max_clock_lag_seconds"] <= .1 and world["clock_lag_over_0p1_seconds_ticks"] == 0
    assert world["physics_runtime"]["errors"] == [] and world["public_events"] == []
    parent = PopulationCase.from_dict(read(PAIR/"event/devext-OOD-L3-00.json"))
    child = load_development_case(world["private_case"])
    assert child == event_clean_sibling(parent)
    assert child.digest() == pair["clean_case_sha256"]
    old_world = read(event_path)
    assert old_world["clock_mode"] == "wallclock_1x" and old_world["physics_steps_executed"] == 144000
    event_endpoints = analyze_event_trace(old_world, parent)
    assert event_endpoints["event_count"] > 0
    # Match the secondary collector: use common-calendar scoring for the clean
    # side, and call the original-event endpoint reader only for the event side.
    clean_endpoints = {"role":"event endpoints not applicable to the clean arm",
                       "observed_public_event_count":len(world["public_events"]),"events":None}
    common = rescore_common_calendar(world, parent)
    grader = RUN/"experiment/evaluator.v2.py"
    assert digest(grader) == "de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b"
    recount = json.loads(subprocess.check_output([sys.executable,str(grader),"--world",str(clean_path),
                         "--expected-world-sha256",digest(clean_path)],timeout=90))
    assert recount["score_equal"] and recount["score"] == world["score"]
    for label, value in (("event-endpoints",event_endpoints),("clean-endpoints",clean_endpoints),
                         ("clean-common-calendar",common),("clean-frozen-score-recount",recount)):
        durable_once(out/(label+".json"),value)
    with (out/"actual-clean.stdout").open("xb") as stream:
        stream.write(stdout.read_bytes())
    attempts = sorted(CLEAN.glob("attempt-*"))
    assert len(attempts) == receipt["actual_attempts"] and 1 <= len(attempts) <= 2
    refs = [*sorted(out.glob("*.json")),out/"actual-clean.stdout",PAIR/"pair.json",event_path,
            CLEAN/"plan.json",Path(__file__).resolve(),grader]
    for attempt in attempts:
        refs.extend(p for p in attempt.rglob("*") if p.is_file())
    value = {"role":"installed clean development mode; event sibling is prior actual flight, no causal estimate",
             "root":child.case_id,"same_geometry_noise_level":True,"clean_actual_attempts":len(attempts),
             "event_actual_public_events":len(old_world["public_events"]),"clean_actual_public_events":0,
             "clean_clock_seconds":world["max_clock_lag_seconds"],"clean_score":world["score"]["Y"],
             "clean_common_calendar_Y":common["score"]["Y"],"prior_event_Y":old_world["score"]["Y"],
             "prior_event_endpoint_rows":len(event_endpoints["rows"]),
             "files":{str(path.relative_to(PROJECT)):digest(path) for path in refs}}
    durable_once(out/"binding.json",value)
    print(json.dumps({**{k:v for k,v in value.items() if k!="files"},
          "binding_sha256":digest(out/"binding.json")},allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
