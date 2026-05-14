#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/qiaojiaxuan/miniconda3/envs/jt/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/logs}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7,8}"
NPROC="${NPROC:-9}"

mkdir -p "${OUTPUT_DIR}" "${JITTOR_HOME:-/data/qiaojiaxuan/jittor_home}"
export HOME="${HOME_OVERRIDE:-/data/qiaojiaxuan}"
export JITTOR_HOME="${JITTOR_HOME:-/data/qiaojiaxuan/jittor_home}"
export PATH="/data/qiaojiaxuan/miniconda3/envs/jt/bin:${PATH}"
export mpicc_path="${mpicc_path:-/data/qiaojiaxuan/miniconda3/envs/jt/bin/mpicc}"
export cache_name="${cache_name:-track_a_ompi4}"
export OMPI_CC="${OMPI_CC:-/usr/bin/gcc}"
export OMPI_CXX="${OMPI_CXX:-/usr/bin/g++}"
export OMPI_MCA_opal_cuda_support="${OMPI_MCA_opal_cuda_support:-true}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export GPUS

cd "${ROOT_DIR}"
mpirun --mca opal_cuda_support 1 \
  --mca btl_vader_single_copy_mechanism none \
  --bind-to none -np "${NPROC}" \
  "${ROOT_DIR}/scripts/mpi_rank.sh" "${PYTHON_BIN}" run.py \
  --task configs/task/train_vm_robust_fast.yaml \
  2>&1 | tee "${OUTPUT_DIR}/train_robust_mpi.log"
