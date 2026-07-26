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
DATASETS="${DATASETS:-qasper musique_2hop musique_3hop musique_4hop}"
OVERLAPS="${OVERLAPS:-0 25 50}"
TABLE_OVERLAPS="${TABLE_OVERLAPS:-0 25 50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/late_chunk_runs}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${PROJECT_ROOT}/late_chunk_evaluations}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TABLE_OUTPUT="${TABLE_OUTPUT:-${PROJECT_ROOT}/docs/qasper_musique_c250_retrieval_table.tex}"
TABLE_JSON_OUTPUT="${TABLE_JSON_OUTPUT:-${PROJECT_ROOT}/docs/qasper_musique_c250_retrieval_table.json}"
CHUNK_SIZE="${CHUNK_SIZE:-250}"
CHUNK_TOKENIZER_NAME="${CHUNK_TOKENIZER_NAME:-jinaai/jina-embeddings-v2-small-en}"
RETRIEVE_K="${RETRIEVE_K:-10}"
RETRIEVAL_SCOPE="${RETRIEVAL_SCOPE:-per_document}"
QASPER_MAX_DOCS="${QASPER_MAX_DOCS:-25}"
MUSIQUE_MAX_DOCS="${MUSIQUE_MAX_DOCS:-}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
LATE_MAX_TOKENS_PER_FORWARD="${LATE_MAX_TOKENS_PER_FORWARD:-8192}"
LATE_WINDOW_OVERLAP_TOKENS="${LATE_WINDOW_OVERLAP_TOKENS:-256}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
GENERATE_TABLE="${GENERATE_TABLE:-1}"
KS="${KS:-5 10}"

mkdir -p "${OUTPUT_ROOT}" "${EVALUATION_ROOT}" "${LOG_DIR}" \
  "$(dirname -- "${TABLE_OUTPUT}")" \
  "$(dirname -- "${TABLE_JSON_OUTPUT}")"
MASTER_LOG="${LOG_DIR}/qasper_musique_c${CHUNK_SIZE}_overlaps_0_25_50.log"
if [[ "${QASPER_MUSIQUE_OVERLAP_LOG_ACTIVE:-0}" != "1" ]]; then
  export QASPER_MUSIQUE_OVERLAP_LOG_ACTIVE=1
  set +e
  bash "$0" "$@" 2>&1 | tee -a "${MASTER_LOG}"
  script_status="${PIPESTATUS[0]}"
  exit "${script_status}"
fi
printf '\n===== QASPER/MuSiQue overlap resume session %s =====\n' \
  "$(date '+%Y-%m-%dT%H:%M:%S%z')"

dataset_config_path() {
  case "$1" in
    qasper)
      printf 'configs/experiments/qasper_retrieval_ablation.yaml'
      ;;
    musique_2hop)
      printf 'configs/experiments/musique_expand60k_2hop_retrieval_ablation.yaml'
      ;;
    musique_3hop)
      printf 'configs/experiments/musique_expand60k_3hop_retrieval_ablation.yaml'
      ;;
    musique_4hop)
      printf 'configs/experiments/musique_expand60k_4hop_retrieval_ablation.yaml'
      ;;
    *)
      printf 'configs/experiments/%s_retrieval_ablation.yaml' "$1"
      ;;
  esac
}

dataset_split() {
  case "$1" in
    qasper)
      printf 'test'
      ;;
    musique_2hop|musique_3hop|musique_4hop)
      printf 'validation'
      ;;
    *)
      printf 'test'
      ;;
  esac
}

dataset_max_docs() {
  case "$1" in
    qasper)
      printf '%s' "${QASPER_MAX_DOCS}"
      ;;
    musique_2hop|musique_3hop|musique_4hop)
      printf '%s' "${MUSIQUE_MAX_DOCS}"
      ;;
    *)
      printf ''
      ;;
  esac
}

IFS=' ' read -r -a dataset_array <<< "${DATASETS}"
IFS=' ' read -r -a overlap_array <<< "${OVERLAPS}"
IFS=' ' read -r -a table_overlap_array <<< "${TABLE_OVERLAPS}"
IFS=' ' read -r -a k_array <<< "${KS}"

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
  printf 'Internal script error: retriever names/specs have different lengths.\n'
  exit 1
fi

if [[ ! "${CHUNK_SIZE}" =~ ^[0-9]+$ || "${CHUNK_SIZE}" -le 0 ]]; then
  printf 'CHUNK_SIZE must be a positive integer, got: %s\n' "${CHUNK_SIZE}"
  exit 1
fi
for overlap in "${overlap_array[@]}" "${table_overlap_array[@]}"; do
  if [[ ! "${overlap}" =~ ^[0-9]+$ || "${overlap}" -ge "${CHUNK_SIZE}" ]]; then
    printf 'Each overlap must be an integer in [0, CHUNK_SIZE), got: %s\n' \
      "${overlap}"
    exit 1
  fi
done

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
        "The selected retriever list includes qwen, which requires "
        "transformers>=4.51.0,<5, but this environment has "
        f"transformers=={transformers.__version__}."
    )

visible_devices = [
    part.strip()
    for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if part.strip()
]
if len(visible_devices) > 1 and importlib.util.find_spec("accelerate") is None:
    raise SystemExit(
        "qwen multi-GPU sharding requires the 'accelerate' package, "
        "but it is not installed."
    )
PY
fi

total_runs=$(( ${#dataset_array[@]} * ${#overlap_array[@]} * ${#retriever_names[@]} ))
run_index=0
failed_runs=0

printf 'Running QASPER and MuSiQue late-chunking overlap additions:\n'
printf '  DATASETS=%s\n' "${DATASETS}"
printf '  RETRIEVERS=%s\n' "${retriever_names[*]}"
printf '  CHUNK_SIZE=%s\n' "${CHUNK_SIZE}"
printf '  OVERLAPS_TO_ENSURE=%s\n' "${OVERLAPS}"
printf '  OVERLAPS_IN_TABLE=%s\n' "${TABLE_OVERLAPS}"
printf '  QASPER_MAX_DOCS=%s\n' "${QASPER_MAX_DOCS:-<all>}"
printf '  MUSIQUE_MAX_DOCS=%s\n' "${MUSIQUE_MAX_DOCS:-<all>}"
printf '  CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '  OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
printf '  EVALUATION_ROOT=%s\n' "${EVALUATION_ROOT}"
printf '  LOG=%s\n' "${MASTER_LOG}"
printf '  TABLE_OUTPUT=%s\n' "${TABLE_OUTPUT}"
printf '  TABLE_JSON_OUTPUT=%s\n' "${TABLE_JSON_OUTPUT}"
printf '  RUN_EVALUATION=%s\n' "${RUN_EVALUATION}"
printf '  GENERATE_TABLE=%s\n' "${GENERATE_TABLE}"
printf '  KS=%s\n' "${KS}"
printf '  HF_HOME=%s\n' "${HF_HOME}"
printf '  HF_TOKEN_PATH=%s\n' "${HF_TOKEN_PATH}"
printf '  HF_TOKEN=%s\n' "${HF_TOKEN_STATUS}"
printf '  TOTAL_CONFIGURATIONS_TO_ENSURE=%s\n\n' "${total_runs}"

for dataset_name in "${dataset_array[@]}"; do
  config_path="$(dataset_config_path "${dataset_name}")"
  split="$(dataset_split "${dataset_name}")"
  max_docs="$(dataset_max_docs "${dataset_name}")"
  for chunk_overlap in "${overlap_array[@]}"; do
    chunk_folder="c${CHUNK_SIZE}_o${chunk_overlap}"
    for idx in "${!retriever_names[@]}"; do
      retriever_name="${retriever_names[$idx]}"
      retriever_spec="${retriever_specs[$idx]}"
      run_name="${retriever_name}/${chunk_folder}"
      run_dir="${OUTPUT_ROOT}/${dataset_name}/${run_name}"
      evaluation_dir="${EVALUATION_ROOT}/${dataset_name}/${run_name}"
      ranking_file="${run_dir}/retrieval/retrieval_payloads__${retriever_name}__late_chunking__per_document.jsonl"
      metrics_file="${evaluation_dir}/metrics_summary.json"
      run_index=$((run_index + 1))

      cmd=(
        "${PYTHON_BIN}"
        "run_late_chunking_experiment.py"
        "--dataset-name" "${dataset_name}"
        "--default-experiment" "${config_path}"
        "--output-root" "${OUTPUT_ROOT}"
        "--chunking-strategy" "fixed"
        "--chunk-size" "${CHUNK_SIZE}"
        "--chunk-overlap" "${chunk_overlap}"
        "--chunk-tokenizer-name" "${CHUNK_TOKENIZER_NAME}"
        "--retrieve-k" "${RETRIEVE_K}"
        "--retrieval-scope" "${RETRIEVAL_SCOPE}"
        "--late-max-tokens-per-forward" "${LATE_MAX_TOKENS_PER_FORWARD}"
        "--late-window-overlap-tokens" "${LATE_WINDOW_OVERLAP_TOKENS}"
        "--run-name" "${run_name}"
        "--retriever" "${retriever_spec}"
      )
      if [[ -n "${max_docs}" ]]; then
        cmd+=("--max-docs" "${max_docs}")
      fi
      if [[ -n "${MAX_QUESTIONS}" ]]; then
        cmd+=("--max-questions" "${MAX_QUESTIONS}")
      fi
      if [[ "${RESUME}" == "0" ]]; then
        cmd+=("--no-resume")
      else
        cmd+=("--resume")
      fi

      printf '[%s/%s] dataset=%s overlap=%s retriever=%s\n' \
        "${run_index}" \
        "${total_runs}" \
        "${dataset_name}" \
        "${chunk_overlap}" \
        "${retriever_name}"
      printf '  Run: %s\n' "${run_dir}"
      printf '  Command: '
      printf '%q ' "${cmd[@]}"
      printf '\n'

      if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  DRY_RUN=1, skipping execution.\n\n'
        continue
      fi

      retrieval_complete=0
      if [[ "${RESUME}" == "1" ]] && "${PYTHON_BIN}" verify_late_chunk_run.py \
        --run-dir "${run_dir}" \
        --retriever-name "${retriever_name}" \
        --quiet; then
        retrieval_complete=1
        printf '  Retrieval already complete and verified; skipping.\n'
      else
        printf '  Retrieval incomplete; running with document-level resume.\n'
        if "${cmd[@]}" && "${PYTHON_BIN}" verify_late_chunk_run.py \
          --run-dir "${run_dir}" \
          --retriever-name "${retriever_name}"; then
          retrieval_complete=1
          printf '  Retrieval completed and verified.\n'
        else
          printf '  Retrieval failed or produced incomplete artifacts.\n'
        fi
      fi

      if [[ "${retrieval_complete}" != "1" ]]; then
        failed_runs=$((failed_runs + 1))
        printf '  Failed.\n\n'
        if [[ "${STOP_ON_ERROR}" == "1" ]]; then
          printf 'Stopping after first failure because STOP_ON_ERROR=1.\n'
          exit 1
        fi
        continue
      fi

      if [[ "${RUN_EVALUATION}" == "1" ]]; then
        if [[
          -s "${metrics_file}"
          && -s "${evaluation_dir}/metrics_per_query.jsonl"
          && -s "${evaluation_dir}/leaderboard_row.json"
          && -s "${evaluation_dir}/evaluation_manifest.json"
          && "${metrics_file}" -nt "${ranking_file}"
        ]]; then
          printf '  Evaluation already complete and current; skipping.\n'
        else
          evaluation_cmd=(
            "${PYTHON_BIN}"
            "evaluate_retrieval_run.py"
            "--run-dir" "${run_dir}"
            "--output-dir" "${evaluation_dir}"
            "--method-name" "${retriever_name}"
            "--dataset-name" "${dataset_name}"
            "--split" "${split}"
            "--run-name" "${run_name}"
            "--ks"
            "${k_array[@]}"
          )
          printf '  Evaluation command: '
          printf '%q ' "${evaluation_cmd[@]}"
          printf '\n'
          if "${evaluation_cmd[@]}"; then
            printf '  Evaluation completed: %s\n' "${evaluation_dir}"
          else
            failed_runs=$((failed_runs + 1))
            printf '  Evaluation failed.\n\n'
            if [[ "${STOP_ON_ERROR}" == "1" ]]; then
              printf 'Stopping after first evaluation failure because STOP_ON_ERROR=1.\n'
              exit 1
            fi
            continue
          fi
        fi
      fi

      printf '  Completed.\n\n'
    done
  done
done

if [[
  "${DRY_RUN}" != "1"
  && "${RUN_EVALUATION}" == "1"
  && "${GENERATE_TABLE}" == "1"
  && "${failed_runs}" == "0"
]]; then
  table_chunk_folders=()
  for table_overlap in "${table_overlap_array[@]}"; do
    table_chunk_folders+=("c${CHUNK_SIZE}_o${table_overlap}")
  done
  table_cmd=(
    "${PYTHON_BIN}"
    "tables/generate_qasper_musique_summary.py"
    "--input-root" "${EVALUATION_ROOT}"
    "--output-tex" "${TABLE_OUTPUT}"
    "--output-json" "${TABLE_JSON_OUTPUT}"
    "--print-table"
    "--chunk-folders"
    "${table_chunk_folders[@]}"
    "--retrievers"
    "${retriever_names[@]}"
  )
  printf 'Table command: '
  printf '%q ' "${table_cmd[@]}"
  printf '\n'
  if "${table_cmd[@]}"; then
    printf 'Updated commit-ready table: %s\n' "${TABLE_OUTPUT}"
    printf 'Updated table audit data: %s\n' "${TABLE_JSON_OUTPUT}"
    printf 'To commit the report: git add %q %q\n' \
      "${TABLE_OUTPUT}" \
      "${TABLE_JSON_OUTPUT}"
  else
    failed_runs=$((failed_runs + 1))
    printf 'Combined overlap table generation failed.\n'
  fi
elif [[ "${failed_runs}" -gt 0 ]]; then
  printf 'Skipping table generation because failed_runs=%s.\n' "${failed_runs}"
fi

printf 'Finished QASPER/MuSiQue overlap runs. failed_runs=%s total_runs=%s\n' \
  "${failed_runs}" \
  "${total_runs}"

if [[ "${failed_runs}" -gt 0 ]]; then
  exit 1
fi
