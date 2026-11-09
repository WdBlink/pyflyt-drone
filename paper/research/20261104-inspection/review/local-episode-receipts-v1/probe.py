"""Independent light IO/process receipt probes. All flight JSON is synthetic."""

from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import stat
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from eval import inspection_episode_receipts as receipt


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    with Path(p).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=False)
    sources = [REPO/"eval/inspection_episode_receipts.py", REPO/"tests/test_inspection_episode_receipts.py"]
    before = {str(p.relative_to(REPO)):sha(p) for p in sources}
    save(out/"source-snapshot.json", {str(p.relative_to(REPO)):p.read_text() for p in sources})
    tests = runpy.run_path(str(sources[1]))
    results = {}

    def setup(name):
        base = out/name
        base.mkdir()
        plan, command, counter = tests["inputs"](base)
        return base, plan, command, counter

    for change in ("stdout", "missing-start", "intent-argv", "terminal-Y"):
        base, plan, command, counter = setup("cache-"+change)
        episode = base/"episode"
        receipt.execute_local_episode(episode, plan, command("ok"))
        attempt = episode/"attempt-00"
        if change == "stdout":
            (attempt/"stdout.log").write_text("changed after terminal\n")
        elif change == "missing-start":
            (attempt/"started.json").unlink()
        elif change == "intent-argv":
            p = attempt/"intent.json"; data=json.loads(p.read_text());data["argv"]=["different-command"]
            p.write_text(json.dumps(data))
        else:
            p=attempt/"terminal.json";data=json.loads(p.read_text());data["Y"]=1.
            p.write_text(json.dumps(data))
        value, evidence = receipt.execute_local_episode(episode, plan, command("fail"))
        terminal = json.loads((attempt/"terminal.json").read_text())
        results["cache-"+change] = dict(accepted=True, Y=value, cache_reused=evidence["cache_reused"],
            worker_calls=int(counter.read_text()), terminal_Y=terminal["Y"],
            stdout_hash_matches=terminal["stdout_sha256"]==sha(attempt/"stdout.log"),
            started_exists=(attempt/"started.json").exists())

    base, plan, command, counter = setup("bad-source-missing-case")
    (base/"fixture.py").write_text(tests["SCRIPT"]+"\n(out/'private-case.json').unlink()\n")
    try:
        receipt.execute_local_episode(base/"episode",plan,command("bad-source"))
    except receipt.UnresolvedEpisode as e:
        terminals=[json.loads(p.read_text()) for p in sorted((base/"episode").glob("attempt-*/terminal.json"))]
        results["bad-source-missing-case"] = dict(error=str(e),worker_calls=int(counter.read_text()),
            statuses=[x["status"] for x in terminals],retry_allowed=[x["retry_allowed"] for x in terminals],
            classified_reasons=[x["evidence"]["failure_reasons"] for x in terminals],
            run_reports_source_changed=[not json.loads(p.read_text())["sources_unchanged"] for p in sorted((base/"episode").glob("attempt-*/flight/run.json"))])

    base, plan, command, counter = setup("start-write-failure")
    original_once, original_popen = receipt.durable_once, receipt.subprocess.Popen
    spawned=[]
    def capture(*args,**kwargs):
        p=original_popen(*args,**kwargs);spawned.append(p);return p
    def fail_start(path,value):
        if Path(path).name=="started.json":raise OSError(errno.ENOSPC,"injected started receipt write failure")
        return original_once(path,value)
    receipt.subprocess.Popen=capture;receipt.durable_once=fail_start
    try:
        try:
            receipt.execute_local_episode(base/"episode",plan,lambda out:[sys.executable,"-c","import time; time.sleep(30)"])
        except OSError as e:
            results["start-write-failure"] = dict(error=str(e),child_alive_after_supervisor_return=spawned[0].poll() is None,
                terminal_exists=(base/"episode/attempt-00/terminal.json").exists())
    finally:
        receipt.subprocess.Popen=original_popen;receipt.durable_once=original_once
        for p in spawned:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
            p.wait(timeout=2)
    try:
        receipt.execute_local_episode(base/"episode",plan,command("ok"))
    except receipt.UnresolvedEpisode as e:
        results["start-write-failure"]["recovery_blocks_unknown"]="unknown attempt" in str(e)
    results["start-write-failure"]["reviewer_cleaned_child"]=all(p.poll() is not None for p in spawned)

    base, plan, command, counter = setup("terminal-write-failure")
    def fail_terminal(path,value):
        if Path(path).name=="terminal.json":raise OSError(errno.ENOSPC,"injected terminal write failure")
        return original_once(path,value)
    receipt.durable_once=fail_terminal
    try:
        try:receipt.execute_local_episode(base/"episode",plan,command("ok"))
        except OSError:pass
    finally:receipt.durable_once=original_once
    try:receipt.execute_local_episode(base/"episode",plan,command("ok"))
    except receipt.UnresolvedEpisode as e:
        results["terminal-write-failure"]=dict(recovery_blocks_unknown="unknown attempt" in str(e),worker_calls=int(counter.read_text()),
            complete_world_exists=(base/"episode/attempt-00/flight/world.json").exists(),
            terminal_exists=(base/"episode/attempt-00/terminal.json").exists())

    base, plan, command, counter = setup("directory-fsync-trace")
    episode=base/"episode";episode.mkdir()
    calls=[];original_fsync=receipt.os.fsync
    def trace_fsync(fd):
        info=os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            root_info=episode.stat();calls.append(dict(is_episode_directory=(info.st_dev,info.st_ino)==(root_info.st_dev,root_info.st_ino),
                attempt_directory_exists=(episode/"attempt-00").exists()))
        return original_fsync(fd)
    receipt.os.fsync=trace_fsync
    try:receipt.execute_local_episode(episode,plan,command("ok"))
    finally:receipt.os.fsync=original_fsync
    results["directory-fsync-trace"]=dict(directory_calls=calls,
        episode_parent_synced_after_attempt_creation=any(x["is_episode_directory"] and x["attempt_directory_exists"] for x in calls))

    results["same-cap-by-condition"]={}
    for condition in ("B0","B1","B2"):
        base, plan, command, counter=setup("same-cap-"+condition)
        plan["condition"]=condition
        (base/"fixture-plan.json").write_text(json.dumps(plan))
        value,evidence=receipt.execute_local_episode(base/"episode",plan,command("first-fail"))
        results["same-cap-by-condition"][condition]=dict(Y=value,worker_calls=int(counter.read_text()),
            actual_attempts=evidence["actual_attempts"],terminal_count=len(list((base/"episode").glob("attempt-*/terminal.json"))))

    base, plan, command, counter=setup("interprocess-lock")
    episode=base/"episode";episode.mkdir();ready=base/"ready"
    code="import fcntl,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); f=p.open('a'); fcntl.flock(f,fcntl.LOCK_EX); pathlib.Path(sys.argv[2]).write_text('ready'); time.sleep(30)"
    owner=subprocess.Popen([sys.executable,"-c",code,str(episode/".lock"),str(ready)])
    try:
        deadline=time.monotonic()+2
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert ready.exists()
        try:receipt.execute_local_episode(episode,plan,command("ok"))
        except receipt.UnresolvedEpisode as e:
            results["interprocess-lock"]=dict(blocked="another supervisor" in str(e),worker_started=counter.exists())
    finally:
        owner.terminate();owner.wait(timeout=2)

    after={str(p.relative_to(REPO)):sha(p) for p in sources}
    assert before==after,"review source changed during probes"
    save(out/"report.json",dict(role="independent_synthetic_receipt_IO_process_review_not_flight_evidence",
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=before,sources_unchanged=True,
        script_sha256=sha(Path(__file__)),python=sys.version,results=results))
    print(json.dumps(results,indent=2))


if __name__=="__main__":main()
