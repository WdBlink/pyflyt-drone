"""Run actual installed flight validators on compressed development copies."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT/"paper/research/20261104-inspection"
Q = RUN/"qualification/formal-capacity-final.v1"
sys.path.insert(0,str(PROJECT))
from eval.inspection_episode_receipts import validate_flight

source=RUN/"qualification/pack-closed-formal-phase.py"
spec=importlib.util.spec_from_file_location("tested_apfs_pack",source)
pack=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def main(tool, result_path):
    assert pack.digest(tool)==pack.TOOL_SHA
    paths=[
        PROJECT/"logs/inspection-final-development-clean.v1/devext-OOD-L3-00-B0-0",
        PROJECT/"logs/inspection-final-development-delay.v1",
    ]
    rows=[]
    # Match the owner/group and allocation device of the actual receipt store.
    with tempfile.TemporaryDirectory(prefix="inspection-apfs-readback-",dir=PROJECT/"logs") as tmp:
        tmp=Path(tmp).resolve()
        for i,path in enumerate(paths):
            terminal=path/"attempt-00/terminal.json"
            t=json.loads(terminal.read_bytes())
            plan=json.loads((path/"plan.json").read_bytes())
            flight=tmp/str(i)
            flight.mkdir()
            for name in ("run.json","world.json","private-case.json"):
                shutil.copy2(terminal.parent/"flight"/name,flight/name)
            world=flight/"world.json"
            expected=pack.digest(world)
            first=pack.pack_file(world,expected,tool,tmp)
            second=pack.pack_file(world,expected,tool,tmp)
            assert first["status"]=="compressed" and second["status"]=="unchanged"
            value, validation=validate_flight(flight,plan,0)
            assert value==t["Y"] and validation==t.get("physical_validation",t.get("evidence"))
            grader=RUN/"experiment/evaluator.v2.py"
            result=json.loads(subprocess.check_output([sys.executable,str(grader),"--world",str(world),
                "--expected-world-sha256",expected],timeout=90))
            assert result["score_equal"] and result["score"]==json.loads(world.read_bytes())["score"]
            assert pack.digest(terminal.parent/"flight/world.json")==expected
            rows.append({"case":plan["case_id"],"Y":value,"compressed":first,"replay":second,
                         "validate_flight_exact":True,"frozen_score_equal":True})
        before=tmp/"wrong-sha.json"
        before.write_bytes(b"example")
        try: pack.pack_file(before,"0"*64,tool,tmp)
        except ValueError: assert before.read_bytes()==b"example"
        else: raise AssertionError("wrong SHA must refuse")
        link=tmp/"alias.json"
        link.symlink_to(before)
        try: pack.pack_file(link,pack.digest(before),tool,tmp)
        except ValueError: assert link.is_symlink()
        else: raise AssertionError("linked path must refuse")
        no_closure=tmp/"no-recording"
        no_closure.mkdir()
        try: pack.closed_rows(no_closure)
        except ValueError: assert not list(no_closure.iterdir())
        else: raise AssertionError("no completion must refuse")
    out={"role":"two actual development evidence copies; no formal data or production file mutated",
         "source_sha256":pack.digest(source),"probe_sha256":pack.digest(Path(__file__).resolve()),
         "tool_sha256":pack.digest(tool),"rows":rows,"wrong_hash_refused":True,"symlink_refused":True,
         "missing_sdk_completion_refused":True}
    assert result_path.parent.resolve()==Q
    with result_path.open("x") as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]).absolute(),Path(sys.argv[2]).absolute())
