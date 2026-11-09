"""Losslessly lower allocation for exact already sealed G1/executor development traces."""
import fcntl
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
Q = RUN / "qualification"
OUT = Q / "formal-main-checkpoints.v1"
LOGS = PROJECT / "logs"
TOOL = LOGS / "inspection-g4-tools.v1/applesauce"
PACK = Q / "pack-closed-formal-phase.py"
sys.path.insert(0, str(PROJECT))

from eval.inspection_episode_receipts import durable_once
from eval.inspection_formal_admission import file_sha

PINS = {
    "executor-full.v1": "21c6ee9864e3ec59ffcb445bc00bbd583b7ef0af70e8096a009db028c6d89a6a",
    "executor-full.v2": "8abb0ab3af95b33037ce790e7a574dbab9ae25034dfcf08860bc63d351e60959",
    "executor-full.v3": "56a968cc6bf307f87edce3b0ead69ec29ffcc235aa434d3dc1d03628d5ce33d4",
    "executor-full.v4": "ef8a9efc045d661acffe1963b7c4eb25834ccb156c35b159eaf5e9c9df01a747",
    "g1-full.v1": "b7820ec90cf487122d92896bb336fd7fd086410eb9472261b3b815b926ca14ca",
    "g1-full.v2": "60ea1fb1b40787af80eb174cb3f8105bb6dbd55ea1075307f376c54f5c37236a",
    "g1-full.v3": "438028618dd18110a0bec82f0bba3950e1de82254d9e8eff399029c06c95d4da",
    "g1-full.v4": "b07166e6415a9f8cfb6f7d003f53dc356938807537f9ad4a7b451fce108d1b97",
}


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    return {"path": str(path.relative_to(RUN)), "sha256": file_sha(path)}


def main():
    main_logs = LOGS / "inspection-main-primary.v1"
    assert (main_logs / "checkpoint.request").is_file()
    assert json.loads((main_logs / "checkpoint-pauses.jsonl").read_text().splitlines()[-1])["order_index"] == 33
    assert (main_logs / "order-receipts/000032.json").is_file() and not (main_logs / "order-receipts/000033.json").exists()
    assert not (RUN / "datasets/holdout.v1").exists()
    assert file_sha(PACK) == "48913b00749a3cb44021edde3ac77e44f8f0a0b146945c5a9c01c0e5fb11c04d"
    assert file_sha(TOOL) == "930a613509fa1db336f3810a59c7c1747d2241630ebac8629bef651896ff0a8b"
    f = runpy.run_path(str(PACK))
    scope = []
    all_refs = {}
    manifests = []
    out = OUT / "closed-qualification-trace-pack.v1.json"
    journal = OUT / "closed-qualification-trace-pack.v1.jsonl"
    assert not out.exists() and not journal.exists()
    # Old producers are closed; serialize any maintenance against our
    # separate maintenance lock without touching native or formal locks.
    with (OUT / ".qualification-pack.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for label, pin in PINS.items():
            folder = Q / label
            path = f["file"](folder / "manifest.json", folder)
            assert file_sha(path) == pin
            manifests.append(ref(path))
            rows = read(path)["grid"]
            assert len(rows) == (100 if label.startswith("g1-") else 40)
            assert len({row["file"] for row in rows}) == len(rows)
            for row in rows:
                assert row["status"] == "completed" and Path(row["file"]).name == row["file"]
                file = f["file"](folder / row["file"], folder)
                assert file.suffix == ".json" and file.stat().st_dev == PROJECT.stat().st_dev
                assert file_sha(file) == row["sha256"]
                all_refs[file] = row["sha256"]
                if file.stat().st_size >= 256 * 1024:
                    scope.append((folder, file, row["sha256"]))
        assert len(all_refs) == 560 and len(scope) >= 140
        before = shutil.disk_usage(PROJECT).free
        assert before > 2 * max(file.stat().st_size for _,file,_ in scope)
        saved = 0
        count = 0
        with journal.open("x") as stream:
            for folder, file, expected in scope:
                result = f["pack_file"](file, expected, TOOL, folder)
                saved += result["before_allocated"] - result["after_allocated"]
                count += result["status"] == "compressed"
                stream.write(json.dumps({"path":str(file.relative_to(RUN)),**result})+"\n")
                stream.flush()
                os.fsync(stream.fileno())
        for path, pin in all_refs.items():
            assert file_sha(path) == pin
        assert all(ref(RUN/m["path"]) == m for m in manifests)
        assert (main_logs / "checkpoint.request").is_file()
        after = shutil.disk_usage(PROJECT).free
        value = {"role":"actual closed development trace storage only, no evidence changes",
                 "manifests":manifests,"world_or_trace_hashes_verified_before_and_after":len(all_refs),
                 "eligible_at_256KiB":len(scope),"compressed":count,"allocated_block_savings_bytes":saved,
                 "free_before_bytes":before,"free_after_bytes":after,"measured_free_change_bytes":after-before,
                 "tool_sha256":file_sha(TOOL),"packer_sha256":file_sha(PACK),"journal":ref(journal),
                 "formal_checkpoint_unchanged":True}
        durable_once(out,value)
        print(json.dumps({"artifact":ref(out),**{k:v for k,v in value.items() if k!="manifests"}}))


if __name__ == "__main__":
    main()
