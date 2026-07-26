#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-/mnt/cache/taghavi}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${PROJECT_ROOT}/.hf_token_unused}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  HF_TOKEN_STATUS="<set>"
else
  HF_TOKEN_STATUS="<not set>"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_NAME="qasper"
CONFIG_PATH="${CONFIG_PATH:-configs/experiments/qasper_retrieval_ablation.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-late_chunk_runs}"
CHUNK_SIZE="${CHUNK_SIZE:-250}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-0}"
CHUNK_TOKENIZER_NAME="${CHUNK_TOKENIZER_NAME:-jinaai/jina-embeddings-v2-small-en}"
RETRIEVE_K="${RETRIEVE_K:-10}"
RETRIEVAL_SCOPE="${RETRIEVAL_SCOPE:-per_document}"
MAX_DOCS="${MAX_DOCS:-25}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
LATE_MAX_TOKENS_PER_FORWARD="${LATE_MAX_TOKENS_PER_FORWARD:-8192}"
LATE_WINDOW_OVERLAP_TOKENS="${LATE_WINDOW_OVERLAP_TOKENS:-256}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

retriever_names=(
  "jina-v3"
  "qwen"
  "contriever"
  "bm25"
  "bge-m3"
)

retriever_specs=(
  "jina-v3"
  "qwen"
  "contriever"
  "bm25"
  "name=bge-m3,type=dense,model_name=BAAI/bge-m3,tokenizer_name=BAAI/bge-m3,normalize=true,distance_metric=cosine"
)

if [[ "${#retriever_names[@]}" -ne "${#retriever_specs[@]}" ]]; then
  printf 'Internal script error: retriever_names and retriever_specs have different lengths.\n'
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
from packaging.version import Version
import importlib.util
import os

import transformers

installed = Version(transformers.__version__)
if installed >= Version("5.0.0"):
    raise SystemExit(
        "Jina retrievers in this project require transformers<5.0.0, "
        f"but this environment has transformers=={transformers.__version__}."
    )

if installed < Version("4.51.0"):
    raise SystemExit(
        "The selected retriever list includes qwen, which requires transformers>=4.51.0, "
        f"but this environment has transformers=={transformers.__version__}. "
        "Use transformers>=4.51.0,<5."
    )

visible_devices = [
    part.strip()
    for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if part.strip()
]
if len(visible_devices) > 1 and importlib.util.find_spec("accelerate") is None:
    raise SystemExit(
        "qwen multi-GPU sharding requires the 'accelerate' package, but it is not installed."
    )
PY
fi

chunk_folder="c${CHUNK_SIZE}_o${CHUNK_OVERLAP}"
total_runs="${#retriever_names[@]}"
failed_runs=0

printf 'Running QASPER late-chunking c250 selected retrievers:\n'
printf '  DATASET_NAME=%s\n' "${DATASET_NAME}"
printf '  RETRIEVERS=%s\n' "${retriever_names[*]}"
printf '  CHUNK_SIZE=%s\n' "${CHUNK_SIZE}"
printf '  CHUNK_OVERLAP=%s\n' "${CHUNK_OVERLAP}"
printf '  CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '  OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
printf '  HF_HOME=%s\n' "${HF_HOME}"
printf '  HF_TOKEN_PATH=%s\n' "${HF_TOKEN_PATH}"
printf '  HF_TOKEN=%s\n' "${HF_TOKEN_STATUS}"
printf '  TOTAL_RUNS=%s\n\n' "${total_runs}"

for idx in "${!retriever_names[@]}"; do
  retriever_name="${retriever_names[$idx]}"
  retriever_spec="${retriever_specs[$idx]}"
  run_name="${retriever_name}/${chunk_folder}"

  cmd=(
    "${PYTHON_BIN}"
    "run_late_chunking_experiment.py"
    "--dataset-name" "${DATASET_NAME}"
    "--default-experiment" "${CONFIG_PATH}"
    "--output-root" "${OUTPUT_ROOT}"
    "--chunking-strategy" "fixed"
    "--chunk-size" "${CHUNK_SIZE}"
    "--chunk-overlap" "${CHUNK_OVERLAP}"
    "--chunk-tokenizer-name" "${CHUNK_TOKENIZER_NAME}"
    "--retrieve-k" "${RETRIEVE_K}"
    "--retrieval-scope" "${RETRIEVAL_SCOPE}"
    "--max-docs" "${MAX_DOCS}"
    "--late-max-tokens-per-forward" "${LATE_MAX_TOKENS_PER_FORWARD}"
    "--late-window-overlap-tokens" "${LATE_WINDOW_OVERLAP_TOKENS}"
    "--run-name" "${run_name}"
    "--retriever" "${retriever_spec}"
  )

  if [[ -n "${MAX_QUESTIONS}" ]]; then
    cmd+=("--max-questions" "${MAX_QUESTIONS}")
  fi

  if [[ "${RESUME}" == "0" ]]; then
    cmd+=("--no-resume")
  else
    cmd+=("--resume")
  fi

  printf '[%s/%s] retriever=%s run_name=%s\n' \
    "$((idx + 1))" \
    "${total_runs}" \
    "${retriever_name}" \
    "${run_name}"
  printf '  Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  DRY_RUN=1, skipping execution.\n\n'
    continue
  fi

  if "${cmd[@]}"; then
    printf '  Completed.\n\n'
    continue
  fi

  failed_runs=$((failed_runs + 1))
  printf '  Failed.\n\n'
  if [[ "${STOP_ON_ERROR}" == "1" ]]; then
    printf 'Stopping after first failure because STOP_ON_ERROR=1.\n'
    exit 1
  fi
done

printf 'Finished selected QASPER c250 late-chunking runs. failed_runs=%s total_runs=%s\n' \
  "${failed_runs}" \
  "${total_runs}"

if [[ "${failed_runs}" -gt 0 ]]; then
  exit 1
fi
