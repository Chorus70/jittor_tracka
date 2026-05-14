#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export HOME="${HOME_OVERRIDE:-/data/qiaojiaxuan}"
export JITTOR_HOME="${JITTOR_HOME:-/data/qiaojiaxuan/jittor_home}"
export PATH="/data/qiaojiaxuan/miniconda3/envs/jt/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/data/qiaojiaxuan/miniconda3/envs/jt/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/mainline_pct_refiner_post_${STAMP}"
mkdir -p "${LOG_DIR}" "${JITTOR_HOME}"

LOCAL_PRED_TASK="configs/task/predict_local_eval_testcats_100_vm_refiner_pct_neighbor_pointgate_pos_ft19_e3_s600.yaml"
TEST_PRED_TASK="configs/task/predict_test_vm_refiner_pct_neighbor_pointgate_pos_ft19_e3_s600.yaml"
LOCAL_PRED_DIR="results_local_eval_testcats_100_refiner_pct_neighbor_pointgate_pos_ft19_e3_s600"
TEST_PRED_DIR="results_test_pct_pointgate_pos_e3_s600"
CANDIDATE_ZIP="result_mainline_pct_pointgate_pos_e3_s600_${STAMP}.zip"
CURRENT_BEST_CD="${CURRENT_BEST_CD:-56.49}"

echo "[post] root=${ROOT_DIR}"
echo "[post] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[post] logs=${LOG_DIR}"

echo "[post] local prediction"
"${PYTHON_BIN}" run.py --task "${LOCAL_PRED_TASK}" 2>&1 | tee "${LOG_DIR}/predict_local.log"

echo "[post] local CD evaluation"
"${PYTHON_BIN}" evaluate.py \
  --pred_dir "${LOCAL_PRED_DIR}" \
  --gt_dir local_eval_testcats_100/gt \
  --noisy_dir local_eval_testcats_100/noisy \
  --list local_eval_testcats_100/test.txt \
  --workers 8 2>&1 | tee "${LOG_DIR}/eval_local_cd.log"

echo "[post] full test prediction"
"${PYTHON_BIN}" run.py --task "${TEST_PRED_TASK}" 2>&1 | tee "${LOG_DIR}/predict_test.log"

echo "[post] package candidate zip"
rm -f "${CANDIDATE_ZIP}"
(cd "${TEST_PRED_DIR}" && zip -qr "../${CANDIDATE_ZIP}" shapenet)

echo "[post] validate candidate zip"
"${PYTHON_BIN}" - <<'PY' "${CANDIDATE_ZIP}" "${LOG_DIR}/eval_local_cd.log" "${CURRENT_BEST_CD}"
import os
import re
import sys
import zipfile

zip_path, eval_log, best_cd_s = sys.argv[1:4]
best_cd = float(best_cd_s)

with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    npy = [n for n in names if n.endswith("denoised.npy")]
    if not names or not any(n.startswith("shapenet/") for n in names):
        raise SystemExit("candidate zip has invalid root layout")
    print(f"[post] candidate_zip={zip_path}")
    print(f"[post] denoised_file_count={len(npy)}")

text = open(eval_log, "r", encoding="utf-8", errors="ignore").read()
m = re.search(r"CD 得分:\s*([0-9.]+)", text)
if not m:
    raise SystemExit("could not parse local CD score")
cd_score = float(m.group(1))
print(f"[post] local_cd_score={cd_score:.4f}")
print(f"[post] current_best_cd_threshold={best_cd:.4f}")

if cd_score > best_cd:
    if os.path.exists("result.zip"):
        backup = "result_backup_before_mainline_pct_pointgate_pos_e3_s600.zip"
        if not os.path.exists(backup):
            os.replace("result.zip", backup)
        else:
            os.remove("result.zip")
    os.replace(zip_path, "result.zip")
    print("[post] result.zip updated with candidate")
else:
    print("[post] candidate did not beat threshold; existing result.zip preserved")
PY

echo "[post] finished"
