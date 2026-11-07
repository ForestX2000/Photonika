#!/usr/bin/env python3
"""Build an audited, self-contained source bundle for rented GPU training.

    python tools/build_training_bundle.py
    python tools/build_training_bundle.py --out results/photonika-a6000.tar.gz

Only the explicit source allowlist below is copied. Training data, weights,
credentials, repository metadata, frontend dependencies and local environments
are never traversed. The SHA256 manifest describes the bytes inside the archive,
including LF normalization applied to the executable Linux entry script.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_ROOT = "photonika-a6000"
ENTRYPOINT = "scripts/train-a6000.sh"
FILE_PATTERNS = (
    "ml/*.py", "ml/assets/**/*", "app/*.py", "app/assets/**/*", "backend/*.py",
    "tests/ml/*.py", "docs/*.md",
)
REQUIRED_FILES = (
    "requirements.txt", "pytest.ini", "README.md", "LICENSE",
    "ml/__init__.py", "ml/model.py", "ml/physics_model.py", "ml/training.py",
    "ml/data.py", "ml/fresnel.py", "ml/losses.py", "ml/config.py",
    "tools/a6000.py", "tools/benchmark_models.py", "tools/verify_submission.py",
    "tools/build_training_bundle.py", ENTRYPOINT,
)
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npy", ".npz",
    ".pem", ".key", ".p12", ".pfx", ".env", ".sqlite", ".db",
    ".zip", ".gz", ".tar", ".tmp", ".part",
}


def _eligible(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return (path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root)
            and not any(part.startswith(".") for part in relative.parts)
            and path.suffix.lower() not in FORBIDDEN_SUFFIXES)


def collect_files(root: Path) -> list[Path]:
    """Select only regular source/assets, failing if the runnable core is absent."""
    root = Path(root).resolve(strict=True)
    selected = set()
    for relative in REQUIRED_FILES:
        path = root / relative
        if not _eligible(root, path):
            raise FileNotFoundError(f"Required regular source file is missing or unsafe: {relative}")
        selected.add(path)
    for pattern in FILE_PATTERNS:
        selected.update(path for path in root.glob(pattern) if _eligible(root, path))
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _file_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() == ".sh":
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def _member(name: str, *, mode: int, size: int = 0, directory: bool = False) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.mode, member.size, member.mtime = mode, size, 0
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    if directory:
        member.type = tarfile.DIRTYPE
    return member


def verify_bundle(archive: Path) -> dict:
    """Inspect every generated member, path, permission and payload SHA256."""
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or str(path) != member.name or "\\" in member.name
                    or any(part in (".", "..") for part in path.parts)
                    or not path.parts or path.parts[0] != ARCHIVE_ROOT):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Nonregular archive member: {member.name}")
            expected_mode = 0o755 if member.isdir() or member.name.endswith(".sh") else 0o644
            if member.mode != expected_mode:
                raise ValueError(f"Unexpected archive permission: {member.name}")
        manifest_name = f"{ARCHIVE_ROOT}/MANIFEST.json"
        stream = bundle.extractfile(manifest_name)
        if stream is None:
            raise ValueError("Bundle manifest is missing")
        manifest = json.load(stream)
        if (manifest.get("schema_version") != 1 or manifest.get("archive_root") != ARCHIVE_ROOT
                or manifest.get("entrypoint") != ENTRYPOINT):
            raise ValueError("Unexpected bundle manifest format")
        payload_names = {member.name for member in members if member.isfile()} - {manifest_name}
        expected_names = {f"{ARCHIVE_ROOT}/{row['path']}" for row in manifest["files"]}
        if len(expected_names) != len(manifest["files"]) or payload_names != expected_names:
            raise ValueError("Manifest does not match archive payload")
        if f"{ARCHIVE_ROOT}/{ENTRYPOINT}" not in payload_names:
            raise ValueError("Training entrypoint is missing")
        for row in manifest["files"]:
            member = bundle.getmember(f"{ARCHIVE_ROOT}/{row['path']}")
            data = bundle.extractfile(member).read()
            if (len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["sha256"]
                    or f"{member.mode:04o}" != row["mode"]):
                raise ValueError(f"Manifest verification failed: {row['path']}")
            if row["path"].endswith(".sh") and b"\r" in data:
                raise ValueError(f"Linux script contains CR line endings: {row['path']}")
        return manifest


def build_bundle(root: Path = ROOT, out: Path | None = None) -> dict:
    """Atomically write a reproducible tar.gz and return its verified manifest."""
    root = Path(root).resolve(strict=True)
    out = Path(out or root / "results" / f"{ARCHIVE_ROOT}.tar.gz").resolve()
    if not out.name.endswith(".tar.gz"):
        raise ValueError("Bundle output must end in .tar.gz")
    sources = collect_files(root)
    payload = []
    for path in sources:
        relative = path.relative_to(root).as_posix()
        data = _file_bytes(path)
        mode = 0o755 if path.suffix.lower() == ".sh" else 0o644
        payload.append((relative, data, mode))
    manifest = dict(schema_version=1, archive_root=ARCHIVE_ROOT, entrypoint=ENTRYPOINT,
                    files=[dict(path=relative, size=len(data), mode=f"{mode:04o}",
                                sha256=hashlib.sha256(data).hexdigest())
                           for relative, data, mode in payload])
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"{out.name}.", suffix=".tmp", dir=out.parent,
                                          delete=False) as stream:
            temporary = Path(stream.name)
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as bundle:
                    directories = {ARCHIVE_ROOT}
                    for relative, _, _ in payload:
                        parent = PurePosixPath(ARCHIVE_ROOT, relative).parent
                        while str(parent) != ".":
                            directories.add(str(parent))
                            parent = parent.parent
                    for directory in sorted(directories):
                        bundle.addfile(_member(directory, mode=0o755, directory=True))
                    for relative, data, mode in payload:
                        bundle.addfile(_member(f"{ARCHIVE_ROOT}/{relative}", mode=mode, size=len(data)),
                                       io.BytesIO(data))
                    bundle.addfile(_member(f"{ARCHIVE_ROOT}/MANIFEST.json", mode=0o644,
                                            size=len(manifest_bytes)), io.BytesIO(manifest_bytes))
        verified = verify_bundle(temporary)
        os.replace(temporary, out)
        return verified
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    output = args.out or args.root / "results" / f"{ARCHIVE_ROOT}.tar.gz"
    manifest = build_bundle(args.root, output)
    print(f"Bundle: {output.resolve()}")
    print(f"Verified files: {len(manifest['files'])}; root: {ARCHIVE_ROOT}/")
    print(f"SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
