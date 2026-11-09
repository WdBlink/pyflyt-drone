"""Install the already-reviewed formal delta only after the actual v4 close-out."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

project=Path.cwd().resolve()
run=project/"paper/research/20261104-inspection"
subject=run/"planning/formal-admission-candidate.v3"
bundle=subject/"response-resource-integration.v1/binding.json"
pilot=run/"qualification/pilot-v4-complete.v1/binding.json"
out=run/"qualification/formal-installation.v1"
hash_bytes=lambda data: sha256(data).hexdigest()
assert not out.exists() and not out.is_symlink()
p=pilot.read_bytes()
assert json.loads(p)["physical_episodes"]==108
b=bundle.read_bytes()
assert hash_bytes(b)=="6c7456781f6dc985b570b5d5d463968ec8fe320b2a8f9a8d16d6fff1c826993d"
manifest=json.loads(b)
prior=json.loads((project/"logs/inspection-eight-condition-pilot.v4/source-hashes.json").read_bytes())
payloads={}
source_count=0
for key,row in manifest["members"].items():
    rel=Path(key)
    assert not rel.is_absolute() and ".." not in rel.parts
    if rel.parts[0]=="source":
        assert rel.parts[1] in ("eval","envs") and rel.suffix==".py"
        name=Path(*rel.parts[1:]); source_count+=1
    else:
        assert rel.parts[0]=="tests" and rel.suffix==".py"
        name=rel
    target=project/name
    frozen=subject/row["frozen_copy"]
    assert target.parent.is_dir() and frozen.resolve().is_relative_to(subject)
    assert not any(part.is_symlink() for path in (target,frozen) for part in (path,*path.parents))
    assert frozen.stat().st_nlink==1
    data=frozen.read_bytes()
    assert len(data)==row["bytes"] and hash_bytes(data)==row["sha256"]
    assert not target.exists() or target.is_file() and target.stat().st_nlink==1
    original=target.read_bytes() if target.exists() else None
    if str(target) in prior:
        assert original is not None and hash_bytes(original)==prior[str(target)]
    assert name not in payloads
    payloads[name]=(original,data,row)
assert len(payloads)==71 and source_count==44
out.mkdir()
rows={}
for name,(original,data,row) in payloads.items():
    if original is not None and original!=data:
        saved=out/"before"/name
        saved.parent.mkdir(parents=True,exist_ok=True)
        with saved.open("xb") as f:f.write(original)
        assert saved.read_bytes()==original
    rows[str(name)]={"before":None if original is None else hash_bytes(original),"after":hash_bytes(data),
        "status":"created" if original is None else "identical" if original==data else "replaced",
        "frozen_copy":row["frozen_copy"]}
with (out/"intent.json").open("x") as f:
    json.dump({"role":"preflight and exact before bytes; not installation completion","members":rows},f,indent=2)
for name,(original,data,_) in payloads.items():
    target=project/name
    assert (target.read_bytes() if target.exists() else None)==original
    if original!=data:
        with NamedTemporaryFile(dir=target.parent,prefix=".inspection-install-",delete=False) as f:
            tmp=Path(f.name)
            f.write(data)
        try:
            tmp.chmod(target.stat().st_mode & 0o777 if original is not None else 0o644)
            os.replace(tmp,target)
        finally:
            tmp.unlink(missing_ok=True)
    assert target.read_bytes()==data
value={"role":"actual verified-byte installation; tests and flight qualification separate",
       "at":datetime.now(timezone.utc).isoformat(),"pilot_sha256":hash_bytes(p),
       "reviewed_integration_binding_sha256":hash_bytes(b),"members":rows,"sources":44,"tests":27}
with (out/"installation.json").open("x") as f:json.dump(value,f,indent=2)
from collections import Counter
print(json.dumps({"installation":str(out/"installation.json"),
    "sha256":hash_bytes((out/"installation.json").read_bytes()),
    "categories":Counter(r["status"] for r in rows.values())}))
