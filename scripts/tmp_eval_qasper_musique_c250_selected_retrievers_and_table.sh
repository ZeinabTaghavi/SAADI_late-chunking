#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-late_chunk_runs}"
EVAL_ROOT="${EVAL_ROOT:-late_chunk_evaluations}"
TABLE_PATH="${TABLE_PATH:-docs/qasper_musique_c250_retrieval_table.tex}"
TABLE_JSON="${TABLE_JSON:-docs/qasper_musique_c250_retrieval_table.json}"
METHOD_NAME="${METHOD_NAME:-late_chunking}"
KS="${KS:-5 10}"
CHUNK_FOLDER="${CHUNK_FOLDER:-c250_o0}"
DATASETS="${DATASETS:-qasper musique_2hop musique_3hop musique_4hop}"
RETRIEVERS="${RETRIEVERS:-jina-v3 qwen contriever bm25 bge-m3}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
STOP_ON_MISSING="${STOP_ON_MISSING:-1}"
DRY_RUN="${DRY_RUN:-0}"

IFS=' ' read -r -a dataset_array <<< "${DATASETS}"
IFS=' ' read -r -a retriever_array <<< "${RETRIEVERS}"
IFS=' ' read -r -a k_array <<< "${KS}"

dataset_split() {
  case "$1" in
    qasper)
      printf 'test'
      ;;
    musique|musique_2hop|musique_3hop|musique_4hop)
      printf 'validation'
      ;;
    *)
      printf 'test'
      ;;
  esac
}

printf 'Evaluating QASPER/MuSiQue c250 retrieval runs:\n'
printf '  DATASETS=%s\n' "${DATASETS}"
printf '  RETRIEVERS=%s\n' "${RETRIEVERS}"
printf '  RUN_ROOT=%s\n' "${RUN_ROOT}"
printf '  EVAL_ROOT=%s\n' "${EVAL_ROOT}"
printf '  CHUNK_FOLDER=%s\n' "${CHUNK_FOLDER}"
printf '  TABLE_PATH=%s\n' "${TABLE_PATH}"
printf '  TABLE_JSON=%s\n' "${TABLE_JSON}"
printf '  KS=%s\n\n' "${KS}"

total_runs=$(( ${#dataset_array[@]} * ${#retriever_array[@]} ))
run_index=0
failed_runs=0
missing_runs=0

for dataset_name in "${dataset_array[@]}"; do
  split="$(dataset_split "${dataset_name}")"
  for retriever_name in "${retriever_array[@]}"; do
    run_index=$((run_index + 1))
    run_dir="${RUN_ROOT}/${dataset_name}/${retriever_name}/${CHUNK_FOLDER}"
    output_dir="${EVAL_ROOT}/${dataset_name}/${retriever_name}/${CHUNK_FOLDER}"
    ranking_file="${run_dir}/retrieval/retrieval_payloads__${retriever_name}__late_chunking__per_document.jsonl"
    metrics_file="${output_dir}/metrics_summary.json"

    printf '[%s/%s] dataset=%s retriever=%s\n' \
      "${run_index}" \
      "${total_runs}" \
      "${dataset_name}" \
      "${retriever_name}"
    printf '  Run: %s\n' "${run_dir}"

    if [[ ! -f "${run_dir}/run_manifest.json" ]]; then
      missing_runs=$((missing_runs + 1))
      printf '  Missing: %s/run_manifest.json\n\n' "${run_dir}"
      if [[ "${STOP_ON_MISSING}" == "1" ]]; then
        printf 'Stopping after missing run because STOP_ON_MISSING=1.\n'
        exit 1
      fi
      continue
    fi

    if [[ "${DRY_RUN}" != "1" ]] && ! "${PYTHON_BIN}" verify_late_chunk_run.py \
      --run-dir "${run_dir}" \
      --retriever-name "${retriever_name}" \
      --quiet; then
      missing_runs=$((missing_runs + 1))
      printf '  Incomplete retrieval artifacts; evaluation cannot run.\n\n'
      if [[ "${STOP_ON_MISSING}" == "1" ]]; then
        printf 'Stopping after incomplete run because STOP_ON_MISSING=1.\n'
        exit 1
      fi
      continue
    fi

    if [[
      "${DRY_RUN}" != "1"
      && -s "${metrics_file}"
      && -s "${output_dir}/metrics_per_query.jsonl"
      && -s "${output_dir}/leaderboard_row.json"
      && -s "${output_dir}/evaluation_manifest.json"
      && "${metrics_file}" -nt "${ranking_file}"
    ]]; then
      printf '  Evaluation already complete and current; skipping.\n\n'
      continue
    fi

    cmd=(
      "${PYTHON_BIN}"
      "evaluate_retrieval_run.py"
      "--run-dir" "${run_dir}"
      "--output-dir" "${output_dir}"
      "--dataset-name" "${dataset_name}"
      "--method-name" "${METHOD_NAME}"
      "--split" "${split}"
      "--ks"
      "${k_array[@]}"
    )

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
      printf 'Stopping after first evaluation failure because STOP_ON_ERROR=1.\n'
      exit 1
    fi
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN=1, skipping table generation.\n'
  exit 0
fi

if [[ "${failed_runs}" -gt 0 || "${missing_runs}" -gt 0 ]]; then
  printf 'Not generating table because failed_runs=%s missing_runs=%s.\n' \
    "${failed_runs}" \
    "${missing_runs}"
  exit 1
fi

table_cmd=(
  "${PYTHON_BIN}"
  "tables/generate_qasper_musique_summary.py"
  "--input-root" "${EVAL_ROOT}"
  "--output-tex" "${TABLE_PATH}"
  "--output-json" "${TABLE_JSON}"
  "--chunk-folder" "${CHUNK_FOLDER}"
  "--print-table"
  "--retrievers"
  "${retriever_array[@]}"
)
printf 'Table command: '
printf '%q ' "${table_cmd[@]}"
printf '\n'
"${table_cmd[@]}"

printf 'Finished QASPER/MuSiQue evaluation and table generation.\n'
