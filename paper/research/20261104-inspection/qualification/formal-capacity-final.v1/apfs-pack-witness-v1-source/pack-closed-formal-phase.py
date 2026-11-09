"""Apply verified lossless APFS compression after a full formal phase closes."""
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
LOGS = PROJECT / "logs"
FORMAL = LOGS / "inspection-formal-episodes.v1"
TOOL_SHA = "930a613509fa1db336f3810a59c7c1747d2241630ebac8629bef651896ff0a8b"
COUNTS = {"main":1296,"holdout":216,"clean":216,"ablation":108,"latency":180}
WIRE = {"reserve-owner.json","settle-owner.json"}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def file(path, root):
    require(path.is_absolute() and path.is_relative_to(root) and ".." not in path.parts
            and not any(part.is_symlink() for part in (path,*path.parents))
            and path.is_file() and path.stat().st_nlink==1,"closed artifact must be a regular unlinked project file")
    return path


def fingerprint(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,
            info.st_mode,info.st_uid,info.st_gid,info.st_nlink)


def pack_file(path, expected, tool, root, *, staging=None):
    path = file(path,root)
    before = path.stat()
    require(digest(path)==expected,"original bytes changed before compression")
    if before.st_flags & 32 or before.st_size < 16384:
        return {"before_allocated":before.st_blocks*512,"after_allocated":before.st_blocks*512,
                "logical":before.st_size,"status":"unchanged","sha256":expected}
    with tempfile.TemporaryDirectory(prefix=".inspection-compress-",dir=staging or root) as td:
        candidate = Path(td)/path.name
        shutil.copy2(path,candidate)
        subprocess.run([str(tool),"compress","-c","zlib","-l","9",str(candidate)],
                       capture_output=True,check=True,timeout=40)
        after = candidate.stat()
        require(digest(candidate)==expected and after.st_dev==before.st_dev and after.st_nlink==1
                and after.st_size==before.st_size and after.st_mtime_ns==before.st_mtime_ns
                and stat.S_IMODE(after.st_mode)==stat.S_IMODE(before.st_mode)
                and (after.st_uid,after.st_gid)==(before.st_uid,before.st_gid),"compressed copy differs")
        require(fingerprint(path.stat())==fingerprint(before) and digest(path)==expected,"source changed during compression")
        if after.st_flags&32 and after.st_blocks < before.st_blocks:
            with candidate.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(candidate,path)
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            require(digest(file(path,root))==expected,"compressed installed bytes differ")
            return {"before_allocated":before.st_blocks*512,"after_allocated":path.stat().st_blocks*512,
                    "logical":before.st_size,"status":"compressed","sha256":expected}
    return {"before_allocated":before.st_blocks*512,"after_allocated":before.st_blocks*512,
            "logical":before.st_size,"status":"unchanged","sha256":expected}


def read(path):
    file(path,PROJECT)
    require(path.stat().st_size < 16*1024**2,"oversized receipt metadata")
    return json.loads(path.read_bytes())


def closed_rows(recording):
    primary = recording/"primary-callbacks-complete.json"
    secondary = recording/"secondary-callbacks-complete.json"
    require(primary.is_file()!=secondary.is_file() and not (recording/"checkpoint.request").exists(),
            "the SDK must have finished exactly one formal phase")
    completion = primary if primary.is_file() else secondary
    complete = read(completion)
    phase = complete["phase"] if primary.is_file() else complete["track"]
    require(phase in COUNTS and (phase in ("main","holdout"))==primary.is_file(),"unknown formal phase")
    request = read(recording/"run-request.json")
    require(Path(request["run"])==RUN,"wrong study run")
    execution = request["execution" if primary.is_file() else "secondary_execution"]
    require(execution==complete["execution_manifest"]
            and digest(file(RUN/execution["path"],RUN))==execution["sha256"],"formal execution manifest differs")
    order = read(recording/"episode-order.json")["slots"]
    count = COUNTS[phase]
    require(len(order)==count==complete.get("actual_episode_denominator",complete.get("additional_actual_episode_denominator")),
            "complete frozen phase denominator required")
    arms = phase in ("ablation","latency")
    groups = ("episodes",) if arms else ("classic","llm")
    index = {} if arms else read(recording/"recording-completion-index.json")["results_sha256"]
    collected, refs = {}, {completion:digest(completion)}
    for group in (*groups,"root-means"):
        result_path = recording/group/"results.json"
        result = read(result_path)
        refs[result_path]=digest(result_path)
        if not arms:
            require(refs[result_path]==index[group],"SDK results changed")
        require(result["execution_status"]=="completed" and result["attempted_count"]==result["reported_denominator"]
                and len(result["runs"])==result["reported_denominator"],"SDK phase is not complete")
        for row in result["runs"]:
            raw_path = file(recording/row["raw"],recording/group/"raw")
            raw = read(raw_path)
            require(raw["status"]==row["status"]=="completed" and raw["attempted"] is True
                    and all(raw[k]==row[k] for k in ("case_id","seed","condition","value")),
                    "SDK row differs from its raw")
            refs[raw_path]=digest(raw_path)
            if group=="root-means":
                require(raw["evidence"]["not_an_additional_episode"] is True,"root mean is not a physical attempt")
            else:
                key = row["case_id"],row["condition"],row["seed"]
                require(key not in collected,"duplicate SDK episode")
                collected[key]=raw
    require(len(collected)==count,"wrong physical SDK row count")
    rows = []
    for i,slot in enumerate(order):
        key = slot["case_id"],slot["arm"] if arms else slot["condition"],slot["repeat_ordinal"]
        require(slot["order_index"]==i and key in collected,"SDK episode order differs")
        rows.append((slot,collected.pop(key)))
    return phase,request,rows,refs


def phase_files(phase, request, rows, refs):
    sys.path.insert(0,str(PROJECT))
    from eval.inspection_episode_receipts import canonical_digest
    from eval.inspection_formal_admission import native_owner_reader
    runtime = RUN/"runtime/autoresearch-document-snapshots-v2"
    ledger = native_owner_reader(runtime)(RUN)["usage_ledger"]
    phase_root = FORMAL/(phase if phase in ("main","holdout") else "secondary-"+phase)
    model_root = LOGS/"inspection-model-resource-operations.v1"
    scope = {}

    def pin(path, expected=None, *, write=False):
        path = file(path,PROJECT)
        actual = digest(path)
        require(expected is None or actual==expected,"closed bytes differ from their receipt")
        require(refs.setdefault(path,actual)==actual,"inconsistent closed byte reference")
        if write: scope[path]=actual
        return actual

    seen = set()
    for slot,raw in rows:
        e = raw["evidence"]
        cond = slot["condition"]
        require(e["score_recount_exact"] is True and e["case_id"]==slot["case_id"] and e["condition"]==cond
                and e["repeat_ordinal"]==slot["repeat_ordinal"],"actual episode identity differs")
        llm = cond not in ("B0","B1","B2")
        terminal = file(Path(e["episode_receipt" if llm else "terminal_path"]),phase_root)
        require(terminal.name=="terminal.json","terminal receipt required")
        directory = terminal.parent.parent
        require(directory.parent==phase_root and directory not in seen,"duplicate or noncanonical episode directory")
        seen.add(directory)
        plan = read(directory/"plan.json")
        require(plan["case_id"]==slot["case_id"] and plan["condition"]==cond
                and plan["repeat_ordinal"]==slot["repeat_ordinal"]
                and Path(plan["formal_dispatch"]["run"])==RUN,"actual formal plan differs")
        pin(directory/"plan.json")
        digest_plan = canonical_digest(plan)
        pin(terminal,e["episode_receipt_sha256" if llm else "terminal_sha256"])
        pin(terminal.parent/"flight/world.json",e["world_sha256"])
        attempts = sorted(directory.glob("attempt-*"))
        require(len(attempts)==e["retained_attempts"] and 1<=len(attempts)<=2
                and terminal.parent==attempts[-1]
                and [a.name for a in attempts]==[f"attempt-{i:02d}" for i in range(len(attempts))],
                "closed finite attempts required")
        for i,attempt in enumerate(attempts):
            doc = read(attempt/"terminal.json")
            final = i==len(attempts)-1
            require(doc["plan_sha256"]==digest_plan and
                    ((final and doc["status"]=="completed") or
                    (not final and doc["status"]=="known_operational_failure" and doc["retry_allowed"] is True)),
                    "unknown, nonretryable or incomplete attempt")
            hashes = doc["payload_hashes" if llm else "record_hashes"]
            if llm:
                require(set(hashes)=={str(p.relative_to(attempt)) for p in attempt.rglob("*")
                                      if p.is_file() and p.name!="terminal.json"},"LLM attempt file set differs")
            for name,expected in hashes.items():
                if expected is None:
                    require(not (attempt/name).exists(),"missing-data receipt differs")
                else:
                    pin(file(attempt/name,attempt),expected)
            for path in attempt.rglob("*"):
                require(not path.is_symlink(),"linked attempt input")
                if path.is_file() and not (path.parent==attempt and path.name in WIRE):
                    pin(path,write=True)
                else:
                    require(path.is_dir() or path.is_file(),"unsupported attempt entry")
            if llm:
                reservation = f"formal-{phase}-{request['g4']['sha256'][:12]}-{slot['order_index']:04d}-a{i}"
                require(read(attempt/"intent.json")["reservation"]==reservation,"reservation identity differs")
                entry = ledger[reservation]
                require(entry["status"]=="known" and entry["unreserved"] is False,"attempt is not settled")
                resource = RUN/"qualification/local-model-episodes"/reservation
                require(RUN/entry["receipt"]==resource/"usage-receipt.json"
                        and set(doc["settlement_file_hashes"])=={str(path) for path in
                        (resource/"usage-receipt.json",resource/"complete-attempt.zip",
                         RUN/"resource-requests"/(reservation+".settle.json"))},"settlement archive identity differs")
                pin(resource/"usage-receipt.json",entry["receipt_sha256"])
                for name,expected in doc["settlement_file_hashes"].items():
                    pin(file(Path(name),RUN),expected)
                operation = model_root/reservation
                require(set(doc["model_payload_hashes"])=={str(p.relative_to(operation)) for p in operation.rglob("*")
                        if p.is_file()},"closed model operation file set differs")
                for name,expected in doc["model_payload_hashes"].items():
                    pin(file(operation/name,operation),expected,write=True)
    return scope


def main(recording,tool):
    require(recording.is_absolute() and recording.is_relative_to(LOGS) and recording.is_dir()
            and not any(part.is_symlink() for part in (recording,*recording.parents)),"actual formal SDK output required")
    require(tool.is_absolute() and tool.is_file() and not tool.is_symlink() and digest(tool)==TOOL_SHA,
            "wrong APFS compression executable")
    lock = file(FORMAL/".matrix.lock",FORMAL)
    with lock.open("rb") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        phase,request,rows,refs = closed_rows(recording)
        output = RUN/"qualification/formal-capacity-final.v1"/f"{phase}-pack-completed.json"
        require(not output.exists(),"this phase already has a completed compression receipt")
        scope = phase_files(phase,request,rows,refs)
        require(scope and shutil.disk_usage(LOGS).free>2*max(path.stat().st_size for path in scope),"temporary copy headroom")
        result = {}
        for path,expected in sorted(scope.items()):
            result[str(path.relative_to(PROJECT))]=pack_file(path,expected,tool,PROJECT,staging=LOGS)
        require(all(digest(path)==expected for path,expected in refs.items()),"post-compression original closure differs")
        summary = {"phase":phase,"rows":len(rows),"files":result,
                   "total_allocated_before":sum(r["before_allocated"] for r in result.values()),
                   "total_allocated_after":sum(r["after_allocated"] for r in result.values()),
                   "free_bytes_after":shutil.disk_usage(LOGS).free,
                   "role":"post-phase physical allocation only; exact logical receipt bytes retained"}
        with output.open("x") as target:json.dump(summary,target,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in summary.items() if k!="files"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]).absolute(),Path(sys.argv[2]).absolute())
