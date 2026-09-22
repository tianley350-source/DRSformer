#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-Options/Deraining.yml}"

export NCCL_P2P_DISABLE=1

python setup.py develop --no_cuda_ext
python -m basicsr.train -opt "$CONFIG" --launcher none
