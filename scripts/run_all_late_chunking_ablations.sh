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

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-late_chunk_runs}"
RESUME="${RESUME:-1}"
RETRIEVE_K="${RETRIEVE_K:-10}"
RETRIEVAL_SCOPE="${RETRIEVAL_SCOPE:-per_document}"
MAX_DOCS="${MAX_DOCS:-25}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
LATE_MAX_TOKENS_PER_FORWARD="${LATE_MAX_TOKENS_PER_FORWARD:-8192}"
LATE_WINDOW_OVERLAP_TOKENS="${LATE_WINDOW_OVERLAP_TOKENS:-256}"
CHUNK_TOKENIZER_NAME="${CHUNK_TOKENIZER_NAME:-jinaai/jina-embeddings-v2-small-en}"
DATASETS="${DATASETS:-qasper loogle narrativeqa quality novelqa}"
RETRIEVER_GRID="${RETRIEVER_GRID:-jina qwen}"
CHUNK_SIZES="${CHUNK_SIZES:-200 300 500}"
CHUNK_OVERLAPS="${CHUNK_OVERLAPS:-0 50 100}"
RUN_SINGLE_SCRIPT="${RUN_SINGLE_SCRIPT:-scripts/run_qasper_late_chunking.sh}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${RUN_SINGLE_SCRIPT}" ]]; then
  chmod +x "${RUN_SINGLE_SCRIPT}"
fi

IFS=' ' read -r -a dataset_array <<< "${DATASETS}"
IFS=' ' read -r -a retriever_array <<< "${RETRIEVER_GRID}"
IFS=' ' read -r -a chunk_size_array <<< "${CHUNK_SIZES}"
IFS=' ' read -r -a chunk_overlap_array <<< "${CHUNK_OVERLAPS}"

total_runs=$(( ${#dataset_array[@]} * ${#retriever_array[@]} * ${#chunk_size_array[@]} * ${#chunk_overlap_array[@]} ))
run_index=0
failed_runs=0

printf 'Running late-chunking ablation grid:\n'
printf '  DATASETS=%s\n' "${DATASETS}"
printf '  RETRIEVER_GRID=%s\n' "${RETRIEVER_GRID}"
printf '  CHUNK_SIZES=%s\n' "${CHUNK_SIZES}"
printf '  CHUNK_OVERLAPS=%s\n' "${CHUNK_OVERLAPS}"
printf '  OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
printf '  RESUME=%s\n' "${RESUME}"
printf '  TOTAL_RUNS=%s\n' "${total_runs}"
printf '\n'

for dataset_name in "${dataset_array[@]}"; do
  for retriever_name in "${retriever_array[@]}"; do
    for chunk_size in "${chunk_size_array[@]}"; do
      for chunk_overlap in "${chunk_overlap_array[@]}"; do
        run_index=$((run_index + 1))
        printf '[%s/%s] dataset=%s retriever=%s chunk_size=%s chunk_overlap=%s\n' \
          "${run_index}" \
          "${total_runs}" \
          "${dataset_name}" \
          "${retriever_name}" \
          "${chunk_size}" \
          "${chunk_overlap}"

        cmd=(
          env
          "DATASET_NAME=${dataset_name}"
          "RETRIEVERS=${retriever_name}"
          "CHUNK_SIZE=${chunk_size}"
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
          "${RUN_SINGLE_SCRIPT}"
        )

        if [[ -n "${MAX_QUESTIONS}" ]]; then
          cmd=(
            env
            "DATASET_NAME=${dataset_name}"
            "RETRIEVERS=${retriever_name}"
            "CHUNK_SIZE=${chunk_size}"
            "CHUNK_OVERLAP=${chunk_overlap}"
            "CHUNK_TOKENIZER_NAME=${CHUNK_TOKENIZER_NAME}"
            "OUTPUT_ROOT=${OUTPUT_ROOT}"
            "RETRIEVE_K=${RETRIEVE_K}"
            "RETRIEVAL_SCOPE=${RETRIEVAL_SCOPE}"
            "MAX_DOCS=${MAX_DOCS}"
            "MAX_QUESTIONS=${MAX_QUESTIONS}"
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
            "${RUN_SINGLE_SCRIPT}"
          )
        fi

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
    done
  done
done

printf 'Finished ablation grid. failed_runs=%s total_runs=%s\n' "${failed_runs}" "${total_runs}"
if [[ "${failed_runs}" -gt 0 ]]; then
  exit 1
fi
