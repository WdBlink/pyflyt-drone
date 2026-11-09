"""Resolve recorded local source bundles for current or archived qualification."""

import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def source_locations(run, source_root=None):
    root = Path(source_root).resolve() if source_root is not None else REPO
    recorded_root = next((Path(name).parent.parent for name in run["sources"]
                          if Path(name).is_absolute() and Path(name).name == "run_inspection_reference.py"), None)
    locations = {}
    for name in run["sources"]:
        path = Path(name)
        if path.is_absolute() and source_root is None:
            actual = path
        else:
            relative = path.relative_to(recorded_root) if path.is_absolute() else path
            if ".." in relative.parts:
                raise ValueError("source bundle cannot escape its root")
            actual = root/relative
            if source_root is not None and not actual.resolve().is_relative_to(root):
                raise ValueError("source bundle symlink escapes root")
        locations[name] = actual
    return locations


def verify_source_bundle(run, source_root=None):
    locations = source_locations(run, source_root)
    for name, source in run["sources"].items():
        if hashlib.sha256(source.encode()).hexdigest() != run["source_hashes"][name]:
            raise ValueError("embedded source hash mismatch: "+name)
        if locations[name].read_text() != source:
            raise ValueError("source differs from recorded attempt: "+name)
    if source_root is not None:
        # The archive's package scaffolding is not executable study code.
        for package in ("envs", "eval"):
            if (Path(source_root)/package/"__init__.py").read_bytes() != b"":
                raise ValueError("archived package scaffolding must be empty")
    return locations


def archive_source_bundle(run,destination):
    """Materialize a new bundle with recorded absolute paths made relative.

    Never join an unchecked absolute source key to a destination: pathlib would
    discard the destination and write back to the original repository instead.
    """
    destination=Path(destination).resolve()
    if destination.exists():
        raise ValueError("source archive destination must be new")
    for name,source in run["sources"].items():
        if hashlib.sha256(source.encode()).hexdigest()!=run["source_hashes"][name]:
            raise ValueError("embedded source hash mismatch")
    locations=source_locations(run,destination)
    destination.mkdir(parents=True)
    for name,path in locations.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(run["sources"][name])
    for package in ("envs","eval"):
        path=destination/package/"__init__.py"
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=b"":
            raise ValueError("archive package scaffolding conflict")
        path.touch()
    verify_source_bundle(run,destination)
    return locations
