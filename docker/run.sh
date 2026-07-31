#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${MVAA_INPUT_DIR:-/input}"
OUTPUT_DIR="${MVAA_OUTPUT_DIR:-/output}"

mkdir -p "${OUTPUT_DIR}/t1_ct" "${OUTPUT_DIR}/t2_tee" "${OUTPUT_DIR}/t3_vid"

python /workspace/src/infer.py \
  --input "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}"
