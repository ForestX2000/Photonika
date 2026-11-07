#!/usr/bin/env bash
# Run on the rented Linux host. Reuse the image; keep code/data at their original
# absolute path so existing A6000 manifests can resume without path migration.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
image="${PHOTONIKA_IMAGE:-photonika:a6000}"
container_name="${PHOTONIKA_CONTAINER:-photonika-a6000}"
base_image="${PHOTONIKA_PYTORCH_IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime}"
action="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi

build_image() {
    docker build --progress=plain -f docker/Dockerfile --target train \
        --build-arg "PYTHON_BASE=$base_image" --build-arg TORCH_PREINSTALLED=1 \
        -t "$image" .
}

case "$action" in
    build) build_image ;;
    run)
        docker info > /dev/null
        if docker container inspect "$container_name" > /dev/null 2>&1; then
            owner="$(docker container inspect --format '{{index .Config.Labels "org.photonika.project"}}' "$container_name")"
            if [[ "$owner" != "$project_dir" ]]; then
                echo "Container $container_name belongs to another project; set PHOTONIKA_CONTAINER." >&2
                exit 1
            fi
            running="$(docker container inspect --format '{{.State.Running}}' "$container_name")"
            if [[ "$running" == true ]]; then
                echo "Already running: $container_name. New arguments were not applied."
                echo "View progress: bash scripts/docker-a6000.sh logs"
                exit 0
            fi
            # Only this launcher's stopped container is replaced. All experiment
            # data live on the host bind mount and remain in the project folder.
            docker container rm "$container_name"
        fi
        if ! docker image inspect "$image" > /dev/null 2>&1; then
            echo "First launch: building $image. Later starts reuse this image."
            build_image
        fi
        if [[ $# -eq 0 ]]; then set -- --resume --skip-fetch --minutes 60; fi
        docker run -d --name "$container_name" --gpus all --shm-size=1g \
            --label "org.photonika.project=$project_dir" \
            --mount "type=bind,source=$project_dir,target=$project_dir" \
            --workdir "$project_dir" "$image" python -u tools/a6000.py "$@"
        echo "View progress: bash scripts/docker-a6000.sh logs"
        ;;
    logs) exec docker logs --tail 60 -f "$container_name" ;;
    status) exec docker ps -a --filter "name=^/${container_name}$" ;;
    stop) exec docker stop --time 30 "$container_name" ;;
    *) echo "Usage: bash scripts/docker-a6000.sh {build|run [training arguments]|logs|status|stop}" >&2; exit 2 ;;
esac
