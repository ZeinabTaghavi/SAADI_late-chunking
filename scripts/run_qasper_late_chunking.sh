#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-/mnt/cache/taghavi}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NOVELHOPQA_BOOKS_ROOT="${NOVELHOPQA_BOOKS_ROOT:-../passing_meta_tag/novelhopqa/book-corpus-root}"
export NOVELHOPQA_SUBSET_MODE="${NOVELHOPQA_SUBSET_MODE:-1}"

DATASET_NAME="${DATASET_NAME:-qasper}"
case "${DATASET_NAME}" in
  qasper)
    DEFAULT_CONFIG_PATH="configs/experiments/qasper_retrieval_ablation.yaml"
    DEFAULT_CHUNK_SIZE="200"
    ;;
  loogle)
    DEFAULT_CONFIG_PATH="configs/experiments/loogle_retrieval_ablation.yaml"
    DEFAULT_CHUNK_SIZE="300"
    ;;
  narrativeqa)
    DEFAULT_CONFIG_PATH="configs/experiments/nqa_retrieval_ablation.yaml"
    DEFAULT_CHUNK_SIZE="300"
    ;;
  quality)
    DEFAULT_CONFIG_PATH="configs/experiments/quality_retrieval_ablation.yaml"
    DEFAULT_CHUNK_SIZE="300"
    ;;
  novelqa|novelhopqa)
    DEFAULT_CONFIG_PATH="configs/experiments/novelqa_retrieval_ablation.yaml"
    DEFAULT_CHUNK_SIZE="500"
    ;;
  *)
    DEFAULT_CONFIG_PATH="configs/experiments/${DATASET_NAME}_retrieval_ablation.yaml"
    DEFAULT_CHUNK_SIZE="200"
    ;;
esac

CONFIG_PATH="${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-late_chunk_runs}"
RUN_NAME="${RUN_NAME:-}"
RETRIEVERS="${RETRIEVERS:-jina}"
CHUNKING_STRATEGY="${CHUNKING_STRATEGY:-fixed}"
CHUNK_SIZE="${CHUNK_SIZE:-${DEFAULT_CHUNK_SIZE}}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-0}"
CHUNK_TOKENIZER_NAME="${CHUNK_TOKENIZER_NAME:-jinaai/jina-embeddings-v2-small-en}"
N_SENTENCES="${N_SENTENCES:-}"
SENTENCE_OVERLAP="${SENTENCE_OVERLAP:-}"
RETRIEVE_K="${RETRIEVE_K:-10}"
RETRIEVAL_SCOPE="${RETRIEVAL_SCOPE:-per_document}"
MAX_DOCS="${MAX_DOCS:-25}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
LATE_MAX_TOKENS_PER_FORWARD="${LATE_MAX_TOKENS_PER_FORWARD:-8192}"
LATE_WINDOW_OVERLAP_TOKENS="${LATE_WINDOW_OVERLAP_TOKENS:-256}"
RESUME="${RESUME:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ " ${RETRIEVERS} " == *" qwen "* || " ${RETRIEVERS} " == *" jina "* || " ${RETRIEVERS} " == *" jina-base "* || " ${RETRIEVERS} " == *" jina-v3 "* ]]; then
  "${PYTHON_BIN}" - <<'PY'
from packaging.version import Version
import importlib.util
import os

import transformers

installed = Version(transformers.__version__)
retrievers = {item.strip() for item in os.environ.get("RETRIEVERS", "").split() if item.strip()}

if "qwen" in retrievers:
    required = Version("4.51.0")
    if installed < required:
        raise SystemExit(
            "Qwen/Qwen3-Embedding-8B requires transformers>=4.51.0, "
            f"but this environment has transformers=={transformers.__version__}. "
            "Upgrade the environment or remove 'qwen' from RETRIEVERS."
        )

if {"jina", "jina-base", "jina-v3"} & retrievers:
    disallowed = Version("5.0.0")
    if installed >= disallowed:
        raise SystemExit(
            "The Jina retrievers used by this project require transformers<5.0.0, "
            f"but this environment has transformers=={transformers.__version__}. "
            "Use a 4.x release such as 'transformers>=4.51.0,<5'."
        )

visible_devices = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
if len(visible_devices) > 1 and importlib.util.find_spec("accelerate") is None:
    raise SystemExit(
        "Qwen multi-GPU sharding requires the 'accelerate' package, but it is not "
        "installed in this environment. Install accelerate or use a single visible GPU."
    )
PY
fi

cmd=(
  "${PYTHON_BIN}"
  "run_late_chunking_experiment.py"
  "--dataset-name" "${DATASET_NAME}"
  "--default-experiment" "${CONFIG_PATH}"
  "--output-root" "${OUTPUT_ROOT}"
  "--chunking-strategy" "${CHUNKING_STRATEGY}"
  "--chunk-size" "${CHUNK_SIZE}"
  "--chunk-overlap" "${CHUNK_OVERLAP}"
  "--chunk-tokenizer-name" "${CHUNK_TOKENIZER_NAME}"
  "--retrieve-k" "${RETRIEVE_K}"
  "--retrieval-scope" "${RETRIEVAL_SCOPE}"
  "--max-docs" "${MAX_DOCS}"
  "--late-max-tokens-per-forward" "${LATE_MAX_TOKENS_PER_FORWARD}"
  "--late-window-overlap-tokens" "${LATE_WINDOW_OVERLAP_TOKENS}"
)

if [[ -n "${N_SENTENCES}" ]]; then
  cmd+=("--n-sentences" "${N_SENTENCES}")
fi

if [[ -n "${SENTENCE_OVERLAP}" ]]; then
  cmd+=("--sentence-overlap" "${SENTENCE_OVERLAP}")
fi

if [[ -n "${MAX_QUESTIONS}" ]]; then
  cmd+=("--max-questions" "${MAX_QUESTIONS}")
fi

if [[ -n "${RUN_NAME}" ]]; then
  cmd+=("--run-name" "${RUN_NAME}")
fi

if [[ "${RESUME}" == "0" ]]; then
  cmd+=("--no-resume")
else
  cmd+=("--resume")
fi

for retriever in ${RETRIEVERS}; do
  cmd+=("--retriever" "${retriever}")
done

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running late-chunking experiment with environment defaults:\n'
printf '  HF_HOME=%s\n' "${HF_HOME}"
printf '  HF_HUB_CACHE=%s\n' "${HF_HUB_CACHE}"
printf '  TRANSFORMERS_CACHE=%s\n' "${TRANSFORMERS_CACHE}"
printf '  HF_DATASETS_CACHE=%s\n' "${HF_DATASETS_CACHE}"
printf '  CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '  DATASET_NAME=%s\n' "${DATASET_NAME}"
printf '  CONFIG_PATH=%s\n' "${CONFIG_PATH}"
printf '  RETRIEVERS=%s\n' "${RETRIEVERS}"
printf '  CHUNKING_STRATEGY=%s\n' "${CHUNKING_STRATEGY}"
printf '  CHUNK_SIZE=%s\n' "${CHUNK_SIZE}"
printf '  CHUNK_OVERLAP=%s\n' "${CHUNK_OVERLAP}"
printf '  CHUNK_TOKENIZER_NAME=%s\n' "${CHUNK_TOKENIZER_NAME}"
printf '  RETRIEVE_K=%s\n' "${RETRIEVE_K}"
printf '  RETRIEVAL_SCOPE=%s\n' "${RETRIEVAL_SCOPE}"
printf '  MAX_DOCS=%s\n' "${MAX_DOCS}"
printf '  MAX_QUESTIONS=%s\n' "${MAX_QUESTIONS:-<config default>}"
printf '  N_SENTENCES=%s\n' "${N_SENTENCES:-<unused>}"
printf '  SENTENCE_OVERLAP=%s\n' "${SENTENCE_OVERLAP:-<unused>}"
printf '  LATE_MAX_TOKENS_PER_FORWARD=%s\n' "${LATE_MAX_TOKENS_PER_FORWARD}"
printf '  LATE_WINDOW_OVERLAP_TOKENS=%s\n' "${LATE_WINDOW_OVERLAP_TOKENS}"
printf '\nCommand:\n  '
printf '%q ' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}"
