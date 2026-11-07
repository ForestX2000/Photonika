#!/usr/bin/env bash
# Use a CUDA PyTorch image/environment supplied by the rented GPU host.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python_bin="${PHOTONIKA_PYTHON:-python3}"
install_dependencies=0
dry_run=0
runner_arguments=()
for argument in "$@"; do
  if [[ "$argument" == "--install" ]]; then
    install_dependencies=1
    continue
  fi
  runner_arguments+=("$argument")
  if [[ "$argument" == "--dry-run" ]]; then
    dry_run=1
  fi
done

# --dry-run deliberately needs neither CUDA nor installed Python dependencies,
# including when --install was supplied in any position.
if [[ "$dry_run" == 1 ]]; then
  exec "$python_bin" -u tools/a6000.py "${runner_arguments[@]}"
fi

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required; use a CUDA image from the GPU host"; print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__, "| CUDA:", torch.version.cuda)'
if [[ "$install_dependencies" == 1 ]]; then
  "$python_bin" -m pip install -r requirements.txt
fi
export PYTHONUTF8=1
export MPLBACKEND=Agg
exec "$python_bin" -u tools/a6000.py "${runner_arguments[@]}"
