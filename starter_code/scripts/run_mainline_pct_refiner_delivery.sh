#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export HOME="${HOME_OVERRIDE:-/data/qiaojiaxuan}"
export JITTOR_HOME="${JITTOR_HOME:-/data/qiaojiaxuan/jittor_home}"
export PATH="/data/qiaojiaxuan/miniconda3/envs/jt/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8}"
export mpicc_path="${mpicc_path:-/data/qiaojiaxuan/miniconda3/envs/jt/bin/mpicc}"
export cache_name="${cache_name:-track_a_ompi4_allvisible}"
export OMPI_CC="${OMPI_CC:-/usr/bin/gcc}"
export OMPI_CXX="${OMPI_CXX:-/usr/bin/g++}"
export OMPI_MCA_opal_cuda_support="${OMPI_MCA_opal_cuda_support:-true}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"

PYTHON_BIN="${PYTHON_BIN:-/data/qiaojiaxuan/miniconda3/envs/jt/bin/python}"
NPROC="${NPROC:-9}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/mainline_pct_refiner_${STAMP}"
mkdir -p "${LOG_DIR}" "${JITTOR_HOME}"

TRAIN_TASK="${TRAIN_TASK:-configs/task/train_vm_refiner_pct_neighbor_pointgate_pos_ft19.yaml}"
LOCAL_PRED_TASK="${LOCAL_PRED_TASK:-configs/task/predict_local_eval_testcats_100_vm_refiner_pct_neighbor_pointgate_pos_ft19_e0_s1200.yaml}"
TEST_PRED_TASK="${TEST_PRED_TASK:-configs/task/predict_test_vm_refiner_pct_neighbor_pointgate_pos_ft19_e0_s1200.yaml}"
LOCAL_PRED_DIR="results_local_eval_testcats_100_refiner_pct_neighbor_pointgate_pos_ft19_e0_s1200"
TEST_PRED_DIR="results_test_pct_pointgate_pos_e0_s1200"
CANDIDATE_ZIP="result_mainline_pct_pointgate_pos_e0_s1200_${STAMP}.zip"
CURRENT_BEST_CD="${CURRENT_BEST_CD:-56.49}"

echo "[mainline] root=${ROOT_DIR}"
echo "[mainline] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[mainline] nproc=${NPROC}"
echo "[mainline] logs=${LOG_DIR}"
echo "[mainline] train_task=${TRAIN_TASK}"

if [ "${NPROC}" -gt 1 ]; then
  mpirun --mca opal_cuda_support 1 \
    --mca btl_vader_single_copy_mechanism none \
    --bind-to none -np "${NPROC}" \
    "${PYTHON_BIN}" run.py --task "${TRAIN_TASK}" 2>&1 | tee "${LOG_DIR}/train.log"
else
  "${PYTHON_BIN}" run.py --task "${TRAIN_TASK}" 2>&1 | tee "${LOG_DIR}/train.log"
fi

echo "[mainline] local prediction"
"${PYTHON_BIN}" run.py --task "${LOCAL_PRED_TASK}" 2>&1 | tee "${LOG_DIR}/predict_local.log"

echo "[mainline] local CD evaluation"
"${PYTHON_BIN}" evaluate.py \
  --pred_dir "${LOCAL_PRED_DIR}" \
  --gt_dir local_eval_testcats_100/gt \
  --noisy_dir local_eval_testcats_100/noisy \
  --list local_eval_testcats_100/test.txt \
  --workers 8 2>&1 | tee "${LOG_DIR}/eval_local_cd.log"

echo "[mainline] full test prediction"
"${PYTHON_BIN}" run.py --task "${TEST_PRED_TASK}" 2>&1 | tee "${LOG_DIR}/predict_test.log"

echo "[mainline] package candidate zip"
rm -f "${CANDIDATE_ZIP}"
(cd "${TEST_PRED_DIR}" && zip -qr "../${CANDIDATE_ZIP}" shapenet)

echo "[mainline] validate candidate zip"
"${PYTHON_BIN}" - <<'PY' "${CANDIDATE_ZIP}" "${LOG_DIR}/eval_local_cd.log" "${CURRENT_BEST_CD}"
import os
import re
import sys
import zipfile

zip_path, eval_log, best_cd_s = sys.argv[1:4]
best_cd = float(best_cd_s)

if not os.path.exists(zip_path):
    raise SystemExit(f"candidate zip missing: {zip_path}")

with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    npy = [n for n in names if n.endswith("denoised.npy")]
    if not names or not any(n.startswith("shapenet/") for n in names):
        raise SystemExit("candidate zip has invalid root layout")
    print(f"[mainline] candidate_zip={zip_path}")
    print(f"[mainline] denoised_file_count={len(npy)}")

text = open(eval_log, "r", encoding="utf-8", errors="ignore").read()
m = re.search(r"CD 得分:\s*([0-9.]+)", text)
if not m:
    raise SystemExit("could not parse local CD score")
cd_score = float(m.group(1))
print(f"[mainline] local_cd_score={cd_score:.4f}")
print(f"[mainline] current_best_cd_threshold={best_cd:.4f}")

if cd_score > best_cd:
    if os.path.exists("result.zip"):
        backup = "result_backup_before_mainline_pct_pointgate_pos.zip"
        if not os.path.exists(backup):
            os.replace("result.zip", backup)
        else:
            os.remove("result.zip")
    os.replace(zip_path, "result.zip")
    print("[mainline] result.zip updated with candidate")
else:
    print("[mainline] candidate did not beat threshold; existing result.zip preserved")
PY

echo "[mainline] finished"
