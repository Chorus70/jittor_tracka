#!/usr/bin/env bash
set -euo pipefail
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3,4,5,6,7,8}"
LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-0}}"
GPU_INDEX=$((LOCAL_RANK % ${#GPU_LIST[@]}))
export CUDA_VISIBLE_DEVICES="${GPU_LIST[$GPU_INDEX]}"
exec "$@"
