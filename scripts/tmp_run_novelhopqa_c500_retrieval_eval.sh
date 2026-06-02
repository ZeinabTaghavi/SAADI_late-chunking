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
export NOVELHOPQA_BOOKS_ROOT="${NOVELHOPQA_BOOKS_ROOT:-/home/iataghav/data/passing_meta_tag/novelhopqa/book-corpus-root}"
export NOVELHOPQA_SUBSET_MODE="${NOVELHOPQA_SUBSET_MODE:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_NAME="${DATASET_NAME:-novelhopqa}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiments/novelqa_retrieval_ablation.yaml}"
RUN_SINGLE_SCRIPT="${RUN_SINGLE_SCRIPT:-scripts/run_qasper_late_chunking.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-late_chunk_runs}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}}"
EVAL_ROOT="${EVAL_ROOT:-late_chunk_evaluations}"
RETRIEVERS="${RETRIEVERS:-jina qwen}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
CHUNK_OVERLAPS="${CHUNK_OVERLAPS:-0 50 100}"
CHUNK_TOKENIZER_NAME="${CHUNK_TOKENIZER_NAME:-jinaai/jina-embeddings-v2-small-en}"
RETRIEVE_K="${RETRIEVE_K:-10}"
RETRIEVAL_SCOPE="${RETRIEVAL_SCOPE:-per_document}"
MAX_DOCS="${MAX_DOCS:-25}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
LATE_MAX_TOKENS_PER_FORWARD="${LATE_MAX_TOKENS_PER_FORWARD:-8192}"
LATE_WINDOW_OVERLAP_TOKENS="${LATE_WINDOW_OVERLAP_TOKENS:-256}"
RESUME="${RESUME:-1}"
METHOD_NAME="${METHOD_NAME:-late_chunking}"
SPLIT="${SPLIT:-test}"
KS="${KS:-5 10}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_INCOMPATIBLE_QWEN="${SKIP_INCOMPATIBLE_QWEN:-1}"

IFS=' ' read -r -a retriever_array <<< "${RETRIEVERS}"
IFS=' ' read -r -a overlap_array <<< "${CHUNK_OVERLAPS}"
IFS=' ' read -r -a k_array <<< "${KS}"

RETRIEVERS_EFFECTIVE="${RETRIEVERS}"
if [[ " ${RETRIEVERS} " == *" qwen "* ]]; then
  TRANSFORMERS_VERSION="$("${PYTHON_BIN}" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || true)"
  if ! "${PYTHON_BIN}" -c 'from packaging.version import Version; import transformers; raise SystemExit(0 if Version(transformers.__version__) >= Version("4.51.0") else 1)' 2>/dev/null; then
    if [[ "${SKIP_INCOMPATIBLE_QWEN}" == "1" ]]; then
      filtered_retrievers=()
      for retriever_name in "${retriever_array[@]}"; do
        if [[ "${retriever_name}" != "qwen" ]]; then
          filtered_retrievers+=("${retriever_name}")
        fi
      done
      retriever_array=("${filtered_retrievers[@]}")
      RETRIEVERS_EFFECTIVE="${retriever_array[*]}"
      printf 'Skipping qwen because this environment has transformers==%s and qwen requires transformers>=4.51.0.\n' "${TRANSFORMERS_VERSION:-unknown}"
      printf 'Upgrade transformers or run with SKIP_INCOMPATIBLE_QWEN=0 to keep the hard failure.\n\n'
    else
      printf 'qwen requires transformers>=4.51.0, but this environment has transformers==%s.\n' "${TRANSFORMERS_VERSION:-unknown}"
      exit 1
    fi
  fi
fi

if [[ "${#retriever_array[@]}" -eq 0 ]]; then
  printf 'No retrievers left to run after compatibility filtering.\n'
  exit 1
fi

failed_runs=0
total_runs=$(( ${#retriever_array[@]} * ${#overlap_array[@]} ))

run_or_record_failure() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  DRY_RUN=1, skipping execution.\n\n'
    return 0
  fi

  if "$@"; then
    printf '  Completed.\n\n'
    return 0
  fi

  failed_runs=$((failed_runs + 1))
  printf '  Failed.\n\n'
  if [[ "${STOP_ON_ERROR}" == "1" ]]; then
    printf 'Stopping after first failure because STOP_ON_ERROR=1.\n'
    exit 1
  fi
  return 1
}

printf 'Running NovelHopQA c500 late-chunking retrieval grid:\n'
printf '  DATASET_NAME=%s\n' "${DATASET_NAME}"
printf '  NOVELHOPQA_BOOKS_ROOT=%s\n' "${NOVELHOPQA_BOOKS_ROOT}"
printf '  NOVELHOPQA_SUBSET_MODE=%s\n' "${NOVELHOPQA_SUBSET_MODE}"
printf '  RETRIEVERS=%s\n' "${RETRIEVERS_EFFECTIVE}"
printf '  CHUNK_OVERLAPS=%s\n' "${CHUNK_OVERLAPS}"
printf '  OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
printf '  EVAL_ROOT=%s\n' "${EVAL_ROOT}"
printf '  TOTAL_RUNS=%s\n\n' "${total_runs}"

run_index=0
for chunk_overlap in "${overlap_array[@]}"; do
  run_index=$((run_index + 1))
  printf '[run %s/%s] dataset=%s chunk_size=%s chunk_overlap=%s\n' \
    "${run_index}" "${#overlap_array[@]}" "${DATASET_NAME}" "${CHUNK_SIZE}" "${chunk_overlap}"

  cmd=(
    env
    "DATASET_NAME=${DATASET_NAME}"
    "CONFIG_PATH=${CONFIG_PATH}"
    "RETRIEVERS=${RETRIEVERS_EFFECTIVE}"
    "CHUNK_SIZE=${CHUNK_SIZE}"
    "CHUNK_OVERLAP=${chunk_overlap}"
    "CHUNK_TOKENIZER_NAME=${CHUNK_TOKENIZER_NAME}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "RETRIEVE_K=${RETRIEVE_K}"
    "RETRIEVAL_SCOPE=${RETRIEVAL_SCOPE}"
    "MAX_DOCS=${MAX_DOCS}"
    "RESUME=${RESUME}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "LATE_MAX_TOKENS_PER_FORWARD=${LATE_MAX_TOKENS_PER_FORWARD}"
    "LATE_WINDOW_OVERLAP_TOKENS=${LATE_WINDOW_OVERLAP_TOKENS}"
    "HF_HOME=${HF_HOME}"
    "HF_HUB_CACHE=${HF_HUB_CACHE}"
    "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
    "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
    "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "NOVELHOPQA_BOOKS_ROOT=${NOVELHOPQA_BOOKS_ROOT}"
    "NOVELHOPQA_SUBSET_MODE=${NOVELHOPQA_SUBSET_MODE}"
  )
  if [[ -n "${MAX_QUESTIONS}" ]]; then
    cmd+=("MAX_QUESTIONS=${MAX_QUESTIONS}")
  fi
  cmd+=(bash "${RUN_SINGLE_SCRIPT}")

  printf '  Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  run_or_record_failure "${cmd[@]}" || true
done

printf 'Evaluating NovelHopQA c500 retrieval runs:\n\n'
eval_index=0
for retriever_name in "${retriever_array[@]}"; do
  for chunk_overlap in "${overlap_array[@]}"; do
    eval_index=$((eval_index + 1))
    run_dir="${RUN_ROOT}/${DATASET_NAME}/${retriever_name}/c${CHUNK_SIZE}_o${chunk_overlap}"
    output_dir="${EVAL_ROOT}/${DATASET_NAME}/${retriever_name}/c${CHUNK_SIZE}_o${chunk_overlap}"

    printf '[eval %s/%s] %s\n' "${eval_index}" "${total_runs}" "${run_dir}"
    if [[ "${DRY_RUN}" != "1" && ! -f "${run_dir}/run_manifest.json" ]]; then
      failed_runs=$((failed_runs + 1))
      printf '  Missing run manifest: %s/run_manifest.json\n\n' "${run_dir}"
      if [[ "${STOP_ON_ERROR}" == "1" ]]; then
        exit 1
      fi
      continue
    fi

    cmd=(
      "${PYTHON_BIN}"
      "evaluate_retrieval_run.py"
      "--run-dir" "${run_dir}"
      "--output-dir" "${output_dir}"
      "--dataset-name" "${DATASET_NAME}"
      "--method-name" "${METHOD_NAME}"
      "--split" "${SPLIT}"
      "--ks"
      "${k_array[@]}"
    )

    printf '  Output: %s\n' "${output_dir}"
    printf '  Command: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    run_or_record_failure "${cmd[@]}" || true
  done
done

printf 'Finished NovelHopQA c500 rerun. failed_runs=%s total_steps=%s\n' \
  "${failed_runs}" \
  "$(( ${#overlap_array[@]} + total_runs ))"

if [[ "${failed_runs}" -gt 0 ]]; then
  exit 1
fi
