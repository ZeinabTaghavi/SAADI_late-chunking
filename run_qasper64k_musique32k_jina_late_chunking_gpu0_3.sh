#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GPU_IDS_CSV="${GPU_IDS_CSV:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
OVERLAPS_CSV="${OVERLAPS_CSV:-0,25,50}"
RETRIEVER="${RETRIEVER:-jina}"
RETRIEVE_K="${RETRIEVE_K:-10}"
RETRIEVAL_SCOPE="${RETRIEVAL_SCOPE:-per_document}"
LATE_MAX_TOKENS_PER_FORWARD="${LATE_MAX_TOKENS_PER_FORWARD:-8192}"
LATE_WINDOW_OVERLAP_TOKENS="${LATE_WINDOW_OVERLAP_TOKENS:-256}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
GENERATE_TABLE="${GENERATE_TABLE:-1}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/late_chunk_runs}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${ROOT_DIR}/late_chunk_evaluations}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
TABLE_OUTPUT="${TABLE_OUTPUT:-${ROOT_DIR}/tables/table_jina_expanded_main_retrieval.tex}"
QASPER_64K_PREPARED_ROOT="${QASPER_64K_PREPARED_ROOT:-${ROOT_DIR}/data/qasper_64k}"
MUSIQUE_32K_PREPARED_ROOT="${MUSIQUE_32K_PREPARED_ROOT:-${ROOT_DIR}/data/musique_32k_saadi}"

HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
TORCH_HOME="${TORCH_HOME:-${ROOT_DIR}/.cache/torch}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

if [[ "${CHUNK_SIZE}" != "500" ]]; then
  printf 'This main-table launcher requires CHUNK_SIZE=500; got %s.\n' "${CHUNK_SIZE}" >&2
  exit 2
fi
if [[ "${RETRIEVER}" != "jina" ]]; then
  printf 'This launcher is intentionally restricted to RETRIEVER=jina; got %s.\n' "${RETRIEVER}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a OVERLAPS <<< "${OVERLAPS_CSV}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  printf 'GPU_IDS_CSV must contain at least one GPU id.\n' >&2
  exit 2
fi
for overlap in "${OVERLAPS[@]}"; do
  case "${overlap}" in
    0|25|50) ;;
    *)
      printf 'Only overlaps 0, 25, and 50 are allowed; got %s.\n' "${overlap}" >&2
      exit 2
      ;;
  esac
done

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export QASPER_64K_PREPARED_ROOT MUSIQUE_32K_PREPARED_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE TORCH_HOME
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${EVALUATION_ROOT}" \
  "${LOG_DIR}" \
  "$(dirname -- "${TABLE_OUTPUT}")" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}"

config_for_dataset() {
  case "$1" in
    qasper_64k)
      printf '%s/configs/experiments/qasper_64k_late_chunking.yaml\n' "${ROOT_DIR}"
      ;;
    musique_32k)
      printf '%s/configs/experiments/musique_32k_late_chunking.yaml\n' "${ROOT_DIR}"
      ;;
    *)
      printf 'Unsupported dataset: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

split_for_dataset() {
  case "$1" in
    qasper_64k) printf 'test\n' ;;
    musique_32k) printf 'validation\n' ;;
    *) return 2 ;;
  esac
}

expected_queries_for_dataset() {
  case "$1" in
    qasper_64k) printf '1372\n' ;;
    musique_32k) printf '900\n' ;;
    *) return 2 ;;
  esac
}

run_is_complete() {
  local run_dir="$1"
  local expected_queries="$2"
  local payload_file="${run_dir}/retrieval/retrieval_payloads__jina__late_chunking__per_document.jsonl"
  local manifest_file="${run_dir}/run_manifest.json"
  [[ -s "${manifest_file}" && -s "${payload_file}" ]] || return 1
  [[ "$(wc -l < "${payload_file}" | tr -d ' ')" == "${expected_queries}" ]]
}

evaluation_is_complete() {
  local metrics_file="$1"
  local dataset="$2"
  local run_name="$3"
  local expected_queries="$4"
  [[ -s "${metrics_file}" ]] || return 1
  "${PYTHON_BIN}" -B -c '
import json
import sys

path, dataset, run_name, expected_queries = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
valid = (
    payload.get("dataset_name") == dataset
    and payload.get("run_name") == run_name
    and int(payload.get("n_queries", -1)) == int(expected_queries)
    and {5, 10}.issubset({int(value) for value in payload.get("k_values", [])})
)
raise SystemExit(0 if valid else 1)
' "${metrics_file}" "${dataset}" "${run_name}" "${expected_queries}"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_job() {
  local dataset="$1"
  local overlap="$2"
  local gpu_id="$3"
  local config_path split expected_queries run_name run_dir evaluation_dir metrics_file
  local log_file

  config_path="$(config_for_dataset "${dataset}")"
  split="$(split_for_dataset "${dataset}")"
  expected_queries="$(expected_queries_for_dataset "${dataset}")"
  run_name="jina/c${CHUNK_SIZE}_o${overlap}"
  run_dir="${OUTPUT_ROOT}/${dataset}/${run_name}"
  evaluation_dir="${EVALUATION_ROOT}/${dataset}/${run_name}"
  metrics_file="${evaluation_dir}/metrics_summary.json"

  retrieval_command=(
    "${PYTHON_BIN}" -B "${ROOT_DIR}/run_late_chunking_experiment.py"
    --dataset-name "${dataset}"
    --default-experiment "${config_path}"
    --retriever "${RETRIEVER}"
    --run-name "${run_name}"
    --output-root "${OUTPUT_ROOT}"
    --chunking-strategy fixed
    --chunk-size "${CHUNK_SIZE}"
    --chunk-overlap "${overlap}"
    --chunk-tokenizer-name jinaai/jina-embeddings-v2-small-en
    --retrieve-k "${RETRIEVE_K}"
    --retrieval-scope "${RETRIEVAL_SCOPE}"
    --late-max-tokens-per-forward "${LATE_MAX_TOKENS_PER_FORWARD}"
    --late-window-overlap-tokens "${LATE_WINDOW_OVERLAP_TOKENS}"
  )
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    retrieval_command+=(--no-resume)
  else
    retrieval_command+=(--resume)
  fi

  printf '[gpu %s] dataset=%s chunk=%s overlap=%s\n' \
    "${gpu_id}" "${dataset}" "${CHUNK_SIZE}" "${overlap}"
  if [[ "${FORCE_RERUN}" != "1" ]] && run_is_complete "${run_dir}" "${expected_queries}"; then
    printf '  Skipping completed retrieval: %s\n' "${run_dir}"
  elif [[ "${DRY_RUN}" == "1" ]]; then
    printf '  DRY_RUN retrieval command:\n'
    print_command env "CUDA_VISIBLE_DEVICES=${gpu_id}" "${retrieval_command[@]}"
  else
    log_file="${LOG_DIR}/late_${dataset}_jina_c${CHUNK_SIZE}_o${overlap}_$(date -u +%Y%m%dT%H%M%SZ).log"
    env CUDA_VISIBLE_DEVICES="${gpu_id}" "${retrieval_command[@]}" 2>&1 | tee "${log_file}"
    if ! run_is_complete "${run_dir}" "${expected_queries}"; then
      printf 'Retrieval did not produce all %s rows: %s\n' "${expected_queries}" "${run_dir}" >&2
      return 1
    fi
  fi

  evaluation_command=(
    "${PYTHON_BIN}" -B "${ROOT_DIR}/evaluate_retrieval_run.py"
    --run-dir "${run_dir}"
    --output-dir "${evaluation_dir}"
    --method-name late_chunking
    --dataset-name "${dataset}"
    --split "${split}"
    --run-name "${run_name}"
    --ks 5 10
  )
  if [[ "${RUN_EVALUATION}" == "1" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '  DRY_RUN evaluation command:\n'
      print_command "${evaluation_command[@]}"
    elif run_is_complete "${run_dir}" "${expected_queries}"; then
      if [[ "${FORCE_RERUN}" != "1" ]] && evaluation_is_complete \
        "${metrics_file}" "${dataset}" "${run_name}" "${expected_queries}"; then
        printf '  Skipping completed evaluation: %s\n' "${evaluation_dir}"
      else
        "${evaluation_command[@]}"
      fi
      if ! evaluation_is_complete \
        "${metrics_file}" "${dataset}" "${run_name}" "${expected_queries}"; then
        printf 'Evaluation is incomplete: %s\n' "${evaluation_dir}" >&2
        return 1
      fi
    fi
  fi
}

printf 'Validating the exact main-SAADI expansion bundles.\n'
"${PYTHON_BIN}" -B "${ROOT_DIR}/validate_expanded_datasets.py"

printf 'Running Jina late-chunking expansion grid.\n'
printf '  GPU_IDS_CSV=%s\n' "${GPU_IDS_CSV}"
printf '  datasets=qasper_64k,musique_32k\n'
printf '  retriever=jina (jinaai/jina-embeddings-v2-small-en)\n'
printf '  chunk_size=%s\n' "${CHUNK_SIZE}"
printf '  overlaps=%s\n' "${OVERLAPS_CSV}"
printf '  output_root=%s\n' "${OUTPUT_ROOT}"

JOB_DATASETS=()
JOB_OVERLAPS=()
for dataset in qasper_64k musique_32k; do
  for overlap in "${OVERLAPS[@]}"; do
    JOB_DATASETS+=("${dataset}")
    JOB_OVERLAPS+=("${overlap}")
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  for ((job_index = 0; job_index < ${#JOB_DATASETS[@]}; job_index++)); do
    gpu_index=$((job_index % ${#GPU_IDS[@]}))
    run_job "${JOB_DATASETS[$job_index]}" "${JOB_OVERLAPS[$job_index]}" "${GPU_IDS[$gpu_index]}"
  done
else
  worker_pids=()
  for ((gpu_index = 0; gpu_index < ${#GPU_IDS[@]}; gpu_index++)); do
    (
      for ((job_index = gpu_index; job_index < ${#JOB_DATASETS[@]}; job_index += ${#GPU_IDS[@]})); do
        run_job \
          "${JOB_DATASETS[$job_index]}" \
          "${JOB_OVERLAPS[$job_index]}" \
          "${GPU_IDS[$gpu_index]}"
      done
    ) &
    worker_pids+=("$!")
  done

  failed_workers=0
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
      failed_workers=$((failed_workers + 1))
    fi
  done
  if [[ "${failed_workers}" -ne 0 ]]; then
    printf '%s GPU worker(s) failed; table generation was not attempted.\n' "${failed_workers}" >&2
    exit 1
  fi
fi

if [[ "${RUN_EVALUATION}" == "1" && "${GENERATE_TABLE}" == "1" && "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" -B "${ROOT_DIR}/tables/generate_jina_expanded_main_table.py" \
    --input-root "${EVALUATION_ROOT}" \
    --output-file "${TABLE_OUTPUT}" \
    --strict
fi

printf 'Jina expanded late-chunking matrix finished.\n'
printf 'Table: %s\n' "${TABLE_OUTPUT}"
