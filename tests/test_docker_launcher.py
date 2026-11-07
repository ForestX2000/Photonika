"""Exercise the server launcher without a Docker daemon or GPU."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = (r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash"))
pytestmark = pytest.mark.skipif(not BASH or not Path(BASH).is_file(), reason="requires Bash")


@pytest.fixture
def launch(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text('''#!/usr/bin/env bash
printf '%s|' "$@" >> "$DOCKER_LOG"
printf '\\n' >> "$DOCKER_LOG"
case "$1 $2" in
  "image inspect") [[ "${MOCK_IMAGE:-1}" == 1 ]] ;;
  "container inspect")
    [[ "${MOCK_CONTAINER:-absent}" != absent ]] || exit 1
    if [[ "${3:-}" == --format ]]; then
      if [[ "$4" == *Labels* ]]; then
        if [[ "${MOCK_CONTAINER}" == foreign ]]; then echo /other/project; else pwd; fi
      elif [[ "${MOCK_CONTAINER}" == running ]]; then echo true; else echo false; fi
    fi ;;
  *) exit 0 ;;
esac
''', encoding="utf-8")
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    bootstrap = '''mock_dir="$1"; script="$2"; shift 2
if command -v cygpath >/dev/null; then
  mock_dir="$(cygpath -u "$mock_dir")"; script="$(cygpath -u "$script")"
fi
export PATH="$mock_dir:$PATH" DOCKER_LOG="$mock_dir/docker.log"
exec bash "$script" "$@"
'''

    def run(*arguments, **environment):
        log.unlink(missing_ok=True)
        result = subprocess.run([BASH, "-c", bootstrap, "_", str(tmp_path),
                                 str(ROOT / "scripts/docker-a6000.sh"), *arguments],
                                cwd=ROOT, env={**os.environ, **environment},
                                capture_output=True, text=True, timeout=15)
        calls = [line.split("|")[:-1] for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_cached_image_starts_without_reinstalling_or_rebuilding_and_preserves_arguments(launch):
    result, calls = launch("run", "--resume", "--minutes", "60")
    assert result.returncode == 0, result.stderr
    assert not any(c[0] == "build" for c in calls)
    run = next(c for c in calls if c[0] == "run")
    assert run[-6:] == ["python", "-u", "tools/a6000.py", "--resume", "--minutes", "60"]
    assert "--gpus" in run and "--mount" in run
    mount = run[run.index("--mount") + 1]
    workdir = run[run.index("--workdir") + 1]
    assert mount == f"type=bind,source={workdir},target={workdir}"
    assert not any("pip" in part for c in calls for part in c)


def test_first_launch_builds_only_training_target_on_preinstalled_pytorch(launch):
    result, calls = launch("run", MOCK_IMAGE="0")
    assert result.returncode == 0, result.stderr
    build = next(c for c in calls if c[0] == "build")
    assert build[build.index("--target") + 1] == "train"
    assert "TORCH_PREINSTALLED=1" in build
    assert "PYTHON_BASE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime" in build
    run = next(c for c in calls if c[0] == "run")
    assert run[-4:] == ["--resume", "--skip-fetch", "--minutes", "60"]


def test_running_container_is_reused_without_starting_duplicate_work(launch):
    result, calls = launch("run", MOCK_CONTAINER="running")
    assert result.returncode == 0 and "Already running" in result.stdout
    assert not any(c[0] in ("run", "build") or c[:2] == ["container", "rm"] for c in calls)


def test_foreign_container_is_never_removed(launch):
    result, calls = launch("run", MOCK_CONTAINER="foreign")
    assert result.returncode != 0 and "another project" in result.stderr
    assert not any(c[:2] == ["container", "rm"] for c in calls)
