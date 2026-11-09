"""Exact external launchd birth/lifetime matching; no process or service actions."""
from datetime import datetime, timedelta, timezone

LABEL = "local.home-jarvis.xiaomi-voice"
PROGRAM = "/Users/wdblink/Code/my_repo/home-jarvis/.venv/bin/python"


def utc(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("explicit observation timezone required")
    return parsed.astimezone(timezone.utc)


def match_lifetime(events, uid, pid, observations):
    """Require one same-boot launchd instance spanning every given time point."""
    if (type(events) is not list or not events or type(uid) is not int or uid < 0
            or type(pid) is not int or pid <= 1 or not observations):
        raise ValueError("bounded exact process identity and observations required")
    subsystem = f"gui/{uid}/{LABEL} [{pid}]"
    fields = ("timestamp", "eventMessage", "subsystem", "processID", "processImagePath", "senderImagePath", "bootUUID", "machTimestamp")

    def one(message, scope):
        rows = [row for row in events if row.get("eventMessage") == message]
        if (len(rows) != 1 or rows[0].get("subsystem") != scope
                or type(rows[0].get("processID")) is not int or rows[0]["processID"] != 1
                or rows[0].get("processImagePath") != "/sbin/launchd"
                or rows[0].get("senderImagePath") != "/sbin/launchd"):
            raise ValueError("missing, ambiguous or foreign launchd event")
        return {key: rows[0][key] for key in fields}

    proxy = one(f"xpcproxy spawned with pid {pid}", subsystem)
    spawn = one(f"Successfully spawned python[{pid}] because semaphore", subsystem)
    remove = one(f"removing child: pid/{pid}", "system")
    boots = {row["bootUUID"] for row in (proxy, spawn, remove)}
    if len(boots) != 1 or type(next(iter(boots))) is not str or not next(iter(boots)):
        raise ValueError("launchd events do not share a recorded boot identity")
    marks = [row["machTimestamp"] for row in (proxy, spawn, remove)]
    if any(type(mark) is not int or mark <= 0 for mark in marks) or not marks[0] < marks[1] < marks[2]:
        raise ValueError("launchd monotonic event order differs")
    start, end = utc(spawn["timestamp"]), utc(remove["timestamp"])
    times = [utc(value) for value in observations]
    if not utc(proxy["timestamp"]) <= start < end or any(not start <= when < end for when in times):
        raise ValueError("observations lie outside the exact launchd lifetime")
    return {"pid": pid, "gui_uid": uid, "job": LABEL, "xpcproxy": proxy, "python_spawn": spawn,
            "removal": remove, "observations_utc": [when.isoformat() for when in times]}


def match_sidecar(sidecar, tree, context, observer_sha, events, uid):
    """Only a registered, reaped tree with one attributable unclaimed PID."""
    if (sidecar["schema"] != "inspection-passive-close-observation-v1"
            or sidecar["context"] != context or sidecar["adapter_source_sha256"] != observer_sha
            or sidecar["original_result"] != tree or sidecar["owned_births"] != tree["owned_births"]
            or sidecar["original_exception"] is not None or sidecar["observation_errors"] != []
            or tree["status"] != "unknown" or tree["remaining_owned"] != []
            or str(tree["root_pid"]) not in tree["owned_births"]
            or len(tree["unclaimed_new_python_orphans"]) != 1):
        raise ValueError("only one unclaimed process with otherwise complete cleanup may be attributed")
    pid, = tree["unclaimed_new_python_orphans"]
    key = str(pid)
    if key in tree["owned_births"] or key in sidecar["baseline_python_rows"]:
        raise ValueError("process already seen in baseline or owned identities")
    if any(row["group"] == pid or row["parent"] == pid for row in tree["owned_births"].values()):
        raise ValueError("unclaimed process shares an owned identity relationship")
    scans = sidecar["global_scans"]
    if len(scans) < 2 or key not in scans[-1]["rows"]:
        raise ValueError("final original scan with the unclaimed process required")
    start_ns = sidecar["close_monotonic_start_ns"]
    close_end = sidecar["close_monotonic_end_ns"]
    close_utc = utc(sidecar["close_started_at"])
    times, rows = [], []
    for i, scan in enumerate(scans):
        first, last = scan["monotonic_start_ns"], scan["monotonic_end_ns"]
        at = utc(scan["started_at"])
        end = at + timedelta(microseconds=(last - first + 999) // 1000)
        if (scan["ordinal"] != i or not start_ns <= first < last <= close_end
                or "inventory_exception_type" in scan or "observation_error" in scan
                or abs((at - close_utc).total_seconds() - (first - sidecar["close_monotonic_start_ns"]) / 1e9) > .005):
            raise ValueError("invalid original scan bounds or wall/monotonic clock mapping")
        start_ns = last
        row = scan["rows"].get(key)
        if row is None:
            continue
        if (row["uid"] != uid or row["parent"] != 1 or row["group"] != pid
                or row["command"] != PROGRAM or row["status"].startswith("Z")):
            raise ValueError("unclaimed original row does not match the exact external identity")
        times.extend((at.isoformat(), end.isoformat()))
        rows.append({"ordinal": i, "row": row, "start_utc": at.isoformat(), "end_utc": end.isoformat()})
    proof = match_lifetime(events, uid, pid, times)
    proxy_local = datetime.fromisoformat(proof["xpcproxy"]["timestamp"])
    born = f"{proxy_local:%a %b} {proxy_local.day} {proxy_local:%H:%M:%S %Y}"
    if any(r["row"]["born"] != born for r in rows):
        raise ValueError("original ps birth second and launchd xpcproxy birth differ")
    return {**proof, "original_scan_rows": rows}
