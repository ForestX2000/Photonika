"""Portable training bundle allowlist, integrity and executable entrypoint."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools import build_training_bundle as builder

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    for relative in builder.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == builder.ENTRYPOINT:
            path.write_bytes(b"#!/usr/bin/env bash\r\npython3 tools/a6000.py --dry-run\r\n")
        elif relative == "tools/a6000.py":
            path.write_bytes((ROOT / relative).read_bytes())
        elif relative == "ml/physics_model.py":
            path.write_text("class PhysicsUnrolledNetV2: pass\n", encoding="utf-8")
        else:
            path.write_text("# source fixture\n", encoding="utf-8")
    return root


def test_bundle_allowlist_omits_credentials_weights_data_and_other_repositories(source_tree, tmp_path):
    forbidden = (".env", ".ssh/id_rsa", ".git/config", "data/weights/model.pt", "data/photos/image.png",
                 "frontend/node_modules/private.js", "remote_upload/ml/model.py", "ml/assets/.env",
                 "ml/assets/weights.safetensors", "ml/assets/run.npy")
    for relative in forbidden:
        path = source_tree / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PRIVATE CONTENT")
    style = source_tree / "ml/assets/style.css"
    style.parent.mkdir(parents=True, exist_ok=True)
    style.write_text("body { color: black; }", encoding="utf-8")
    output = tmp_path / "bundle.tar.gz"
    manifest = builder.build_bundle(source_tree, output)
    selected = {row["path"] for row in manifest["files"]}
    assert not selected.intersection(forbidden)
    assert "ml/assets/style.css" in selected
    assert "ml/physics_model.py" in selected
    assert "tools/build_training_bundle.py" in selected
    with tarfile.open(output) as archive:
        physics = archive.extractfile(f"{builder.ARCHIVE_ROOT}/ml/physics_model.py").read()
        assert b"PhysicsUnrolledNetV2" in physics
        assert all(b"PRIVATE CONTENT" not in archive.extractfile(member).read()
                   for member in archive.getmembers() if member.isfile())


def test_manifest_describes_actual_normalized_bytes_and_permissions(source_tree, tmp_path):
    output = tmp_path / "bundle.tar.gz"
    manifest = builder.build_bundle(source_tree, output)
    with tarfile.open(output) as archive:
        for row in manifest["files"]:
            member = archive.getmember(f"{builder.ARCHIVE_ROOT}/{row['path']}")
            data = archive.extractfile(member).read()
            assert row["sha256"] == hashlib.sha256(data).hexdigest()
            assert row["size"] == len(data)
            assert row["mode"] == f"{member.mode:04o}"
        script = archive.getmember(f"{builder.ARCHIVE_ROOT}/{builder.ENTRYPOINT}")
        assert script.mode == 0o755
        assert b"\r" not in archive.extractfile(script).read()
    assert builder.verify_bundle(output) == manifest


def test_bundle_is_reproducible(source_tree, tmp_path):
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    builder.build_bundle(source_tree, first)
    builder.build_bundle(source_tree, second)
    assert first.read_bytes() == second.read_bytes()


def test_failed_verification_preserves_previous_archive_and_cleans_temporary(source_tree, tmp_path, monkeypatch):
    output = tmp_path / "bundle.tar.gz"
    output.write_bytes(b"previous artifact")

    def reject(path):
        raise ValueError("verification failure")

    monkeypatch.setattr(builder, "verify_bundle", reject)
    with pytest.raises(ValueError, match="verification failure"):
        builder.build_bundle(source_tree, output)
    assert output.read_bytes() == b"previous artifact"
    assert not list(tmp_path.glob("bundle.tar.gz.*.tmp"))


def test_unsafe_archive_path_is_rejected(tmp_path):
    output = tmp_path / "unsafe.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        member = tarfile.TarInfo(f"{builder.ARCHIVE_ROOT}/../escape")
        member.mode, member.size = 0o644, 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe archive path"):
        builder.verify_bundle(output)


def test_missing_entrypoint_is_an_error(source_tree, tmp_path):
    (source_tree / builder.ENTRYPOINT).unlink()
    with pytest.raises(FileNotFoundError, match="train-a6000.sh"):
        builder.build_bundle(source_tree, tmp_path / "bundle.tar.gz")


def test_extracted_runner_dry_run_needs_no_torch_or_gpu(source_tree, tmp_path):
    output = tmp_path / "bundle.tar.gz"
    builder.build_bundle(source_tree, output)
    extracted = tmp_path / "extracted"
    with tarfile.open(output) as archive:
        for member in archive.getmembers():
            destination = extracted / member.name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
    command = [sys.executable, "-I", str(extracted / builder.ARCHIVE_ROOT / "tools/a6000.py"), "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["common_training"]["model_version"] == 2
    assert plan["common_training"]["photo_split"] == "source"


def test_actual_source_selection_includes_current_model_and_teacher():
    paths = {path.relative_to(ROOT).as_posix() for path in builder.collect_files(ROOT)}
    assert {"ml/physics_model.py", "ml/teacher.py", "tools/a6000.py", builder.ENTRYPOINT} <= paths
