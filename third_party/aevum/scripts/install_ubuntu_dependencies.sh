#!/usr/bin/env bash
set -euo pipefail
sudo apt update
sudo apt install -y build-essential g++ make python3 pkg-config clang ocl-icd-opencl-dev opencl-headers clinfo
