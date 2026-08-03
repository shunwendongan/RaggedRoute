#!/usr/bin/env bash
set -euo pipefail

if [[ $(id -u) -ne 0 ]]; then
  echo "run as root: wsl -d Ubuntu -u root -- bash scripts/setup_wsl_nsight.sh" >&2
  exit 2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
  -o "$tmp/cuda-keyring.deb"
dpkg -i "$tmp/cuda-keyring.deb"
apt-get update
apt-get install -y nsight-compute-2026.2.1 nsight-systems-2026.1.3
/opt/nvidia/nsight-compute/2026.2.1/ncu --version
nsys --version
