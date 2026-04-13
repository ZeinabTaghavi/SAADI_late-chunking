#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-late_chunk_runs}"
EVAL_ROOT="${EVAL_ROOT:-late_chunk_evaluations}"
DATASET_NAMES="${DATASET_NAMES:-novelqa novelhopqa}"
RETRIEVERS="${RETRIEVERS:-jina qwen}"
METHOD_NAME="${METHOD_NAME:-late_chunking}"
SPLIT="${SPLIT:-test}"
KS="${KS:-5 10}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"

IFS=' ' read -r -a dataset_array <<< "${DATASET_NAMES}"
IFS=' ' read -r -a retriever_array <<< "${RETRIEVERS}"
IFS=' ' read -r -a k_array <<< "${KS}"

all_run_dirs=()
all_dataset_names=()

for dataset_name in "${dataset_array[@]}"; do
  found_for_dataset=0
  for retriever_name in "${retriever_array[@]}"; do
    retriever_root="${RUN_ROOT}/${dataset_name}/${retriever_name}"
    if [[ ! -d "${retriever_root}" ]]; then
      printf 'Skipping retriever %s for dataset %s because no run directory exists at %s\n' \
        "${retriever_name}" \
        "${dataset_name}" \
        "${retriever_root}"
      continue
    fi

    while IFS= read -r manifest_path; do
      run_dir="$(dirname "${manifest_path}")"
      all_run_dirs+=("${run_dir}")
      all_dataset_names+=("${dataset_name}")
      found_for_dataset=1
    done < <(find "${retriever_root}" -type f -name 'run_manifest.json' | sort)
  done

  if [[ "${found_for_dataset}" == "0" ]]; then
    printf 'No run manifests were found under %s/%s for retrievers: %s\n' \
      "${RUN_ROOT}" \
      "${dataset_name}" \
      "${RETRIEVERS}"
  fi
done

if [[ "${#all_run_dirs[@]}" -eq 0 ]]; then
  printf 'No NovelHopQA/NovelQA run manifests were found under %s for datasets: %s\n' \
    "${RUN_ROOT}" \
    "${DATASET_NAMES}"
  exit 1
fi

printf 'Running batch retrieval evaluation for NovelHopQA/NovelQA:\n'
printf '  DATASET_NAMES=%s\n' "${DATASET_NAMES}"
printf '  RETRIEVERS=%s\n' "${RETRIEVERS}"
printf '  RUN_ROOT=%s\n' "${RUN_ROOT}"
printf '  EVAL_ROOT=%s\n' "${EVAL_ROOT}"
printf '  METHOD_NAME=%s\n' "${METHOD_NAME}"
printf '  SPLIT=%s\n' "${SPLIT}"
printf '  KS=%s\n' "${KS}"
printf '  TOTAL_RUNS=%s\n' "${#all_run_dirs[@]}"
printf '\n'

run_index=0
failed_runs=0

for idx in "${!all_run_dirs[@]}"; do
  run_dir="${all_run_dirs[$idx]}"
  dataset_name="${all_dataset_names[$idx]}"
  run_index=$((run_index + 1))
  relative_suffix="${run_dir#${RUN_ROOT}/}"
  if [[ "${relative_suffix}" == "${run_dir}" ]]; then
    printf 'Could not map run directory %s relative to RUN_ROOT=%s\n' "${run_dir}" "${RUN_ROOT}"
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      exit 1
    fi
    failed_runs=$((failed_runs + 1))
    continue
  fi
  output_dir="${EVAL_ROOT}/${relative_suffix}"

  cmd=(
    "${PYTHON_BIN}"
    "evaluate_retrieval_run.py"
    "--run-dir" "${run_dir}"
    "--output-dir" "${output_dir}"
    "--dataset-name" "${dataset_name}"
    "--method-name" "${METHOD_NAME}"
    "--split" "${SPLIT}"
    "--ks"
    "${k_array[@]}"
  )

  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi

  printf '[%s/%s] %s\n' "${run_index}" "${#all_run_dirs[@]}" "${run_dir}"
  printf '  Dataset: %s\n' "${dataset_name}"
  printf '  Output: %s\n' "${output_dir}"
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

printf 'Finished NovelHopQA/NovelQA retrieval evaluation batch. failed_runs=%s total_runs=%s\n' \
  "${failed_runs}" \
  "${#all_run_dirs[@]}"

if [[ "${failed_runs}" -gt 0 ]]; then
  exit 1
fi
