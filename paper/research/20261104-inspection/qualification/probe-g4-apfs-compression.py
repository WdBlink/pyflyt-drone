"""Measure byte-exact APFS compression on temporary copies of closed evidence."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
OUT = RUN / "qualification/formal-capacity-final.v1"
TOOL_SHA = "930a613509fa1db336f3810a59c7c1747d2241630ebac8629bef651896ff0a8b"
WIRE = {"reserve-owner.json", "settle-owner.json"}


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main(tool):
    started = time.monotonic()
    assert tool.is_file() and not tool.is_symlink() and digest(tool) == TOOL_SHA
    source = OUT/"flight-storage.json"
    samples = json.loads(source.read_text())["samples"]
    assert len(samples) == 112
    records = []
    for item in samples:
        base = PROJECT/item["attempt"]
        assert digest(base/"terminal.json") == item["terminal_sha256"]
        scopes = {"episode_nonwire":base}
        if "reservation" in item:
            scopes["model_originals"] = PROJECT/"logs/inspection-model-resource-operations.v1"/item["reservation"]
        with tempfile.TemporaryDirectory(prefix="inspection-compression-copy-",dir="/tmp") as directory:
            temporary = Path(directory)
            assert temporary.stat().st_dev == base.stat().st_dev
            members = {}
            for group, root in scopes.items():
                for path in sorted(root.rglob("*")):
                    assert not path.is_symlink()
                    if not path.is_file() or group=="episode_nonwire" and path.parent==base and path.name in WIRE:
                        continue
                    assert path.stat().st_nlink==1
                    copy = temporary/group/path.relative_to(root)
                    copy.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,copy)
                    members[copy]=(group,path,digest(path),copy.stat())
            subprocess.run([str(tool),"compress","-c","zlib","-l","9",str(temporary)],
                           capture_output=True,text=True,check=True,timeout=40)
            totals = {group:dict(logical=0,before_allocated=0,after_allocated=0,compressed_files=0,files=0) for group in scopes}
            world = None
            for copy,(group,path,expected,before) in members.items():
                after = copy.stat()
                assert digest(copy)==expected and before.st_size==after.st_size and after.st_nlink==1
                assert stat.S_IMODE(before.st_mode)==stat.S_IMODE(after.st_mode)
                assert before.st_mtime_ns==after.st_mtime_ns
                totals[group]["logical"]+=before.st_size
                totals[group]["before_allocated"]+=before.st_blocks*512
                totals[group]["after_allocated"]+=after.st_blocks*512
                totals[group]["compressed_files"]+=bool(after.st_flags&32)
                totals[group]["files"]+=1
                if path==base/"flight/world.json":
                    world={"source_sha256":expected,"logical":before.st_size,"before_allocated":before.st_blocks*512,
                           "after_allocated":after.st_blocks*512,"flag":bool(after.st_flags&32)}
            assert world is not None and world["flag"] and world["before_allocated"]>world["after_allocated"]
            assert totals["episode_nonwire"]["logical"]==item["episode_nonwire_bytes"]
            if "model_originals" in totals:
                assert totals["model_originals"]["logical"]==item["model_originals_bytes"]
            records.append({"attempt":item["attempt"],"role":item["role"],"class":"llm" if len(scopes)==2 else "classic",
                            "groups":totals,"world":world})
    maxima = {group:{scope:max(r["groups"][scope]["after_allocated"] for r in records if r["class"]==group)
                      for scope in (("episode_nonwire","model_originals") if group=="llm" else ("episode_nonwire",))}
              for group in ("classic","llm")}
    result = {"role":"actual allocated bytes on same APFS with compressed temporary copies; originals untouched",
              "tool_version":"applesauce-cli 0.5.30","tool_sha256":TOOL_SHA,"codec":"zlib","level":9,
              "samples":len(records),"per_class_allocated_maxima":maxima,"records":records,
              "flight_inventory_sha256":digest(source),"probe_source_sha256":digest(Path(__file__).resolve()),
              "seconds":time.monotonic()-started}
    with (OUT/"apfs-temporary-compression.v1.json").open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
