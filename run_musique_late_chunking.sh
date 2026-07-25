#!/usr/bin/env bash
set -euo pipefail

# --------------------------- user configuration ---------------------------
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
HOPS_CSV="${HOPS_CSV:-2,3,4}"
RETRIEVERS_CSV="${RETRIEVERS_CSV:-jina,qwen}"
CHUNK_SIZES_CSV="${CHUNK_SIZES_CSV:-200,300,500}"
OVERLAPS_CSV="${OVERLAPS_CSV:-0,50,100}"
RETRIEVE_K="${RETRIEVE_K:-10}"
RETRIEVAL_SCOPE="${RETRIEVAL_SCOPE:-per_document}"
CHUNK_TOKENIZER_NAME="${CHUNK_TOKENIZER_NAME:-jinaai/jina-embeddings-v2-small-en}"
LATE_MAX_TOKENS_PER_FORWARD="${LATE_MAX_TOKENS_PER_FORWARD:-8192}"
LATE_WINDOW_OVERLAP_TOKENS="${LATE_WINDOW_OVERLAP_TOKENS:-256}"

# All default data, caches, runs, evaluations, logs, and tables are local.
HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
TORCH_HOME="${TORCH_HOME:-${ROOT_DIR}/.cache/torch}"
MUSIQUE_PREPARED_ROOT="${MUSIQUE_PREPARED_ROOT:-${ROOT_DIR}/data/musique_expand60k/prepared_hops300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/late_chunk_runs}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${ROOT_DIR}/late_chunk_evaluations}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
TABLE_OUTPUT="${TABLE_OUTPUT:-${ROOT_DIR}/tables/late_chunking_mega_table.txt}"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
RESUME="${RESUME:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
GENERATE_TABLE="${GENERATE_TABLE:-1}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
# -------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export TRANSFORMERS_CACHE TORCH_HOME MUSIQUE_PREPARED_ROOT
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${OUTPUT_ROOT}" \
  "${EVALUATION_ROOT}" "${LOG_DIR}" "$(dirname -- "${TABLE_OUTPUT}")"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "Running standalone MuSiQue Late Chunking"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  HF_HOME=${HF_HOME}"
echo "  MUSIQUE_PREPARED_ROOT=${MUSIQUE_PREPARED_ROOT}"
echo "  HOPS_CSV=${HOPS_CSV}"
echo "  RETRIEVERS_CSV=${RETRIEVERS_CSV}"
echo "  CHUNK_SIZES_CSV=${CHUNK_SIZES_CSV}"
echo "  OVERLAPS_CSV=${OVERLAPS_CSV}"
echo "  RETRIEVE_K=${RETRIEVE_K}"
echo "  LATE_MAX_TOKENS_PER_FORWARD=${LATE_MAX_TOKENS_PER_FORWARD}"
echo "  LATE_WINDOW_OVERLAP_TOKENS=${LATE_WINDOW_OVERLAP_TOKENS}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  EVALUATION_ROOT=${EVALUATION_ROOT}"

"${PYTHON_BIN}" "${ROOT_DIR}/validate_musique.py"
if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  exit 0
fi

IFS=',' read -r -a HOPS <<< "${HOPS_CSV}"
IFS=',' read -r -a RETRIEVERS <<< "${RETRIEVERS_CSV}"
IFS=',' read -r -a CHUNK_SIZES <<< "${CHUNK_SIZES_CSV}"
IFS=',' read -r -a OVERLAPS <<< "${OVERLAPS_CSV}"

config_for_hop() {
  case "$1" in
    2|3|4) printf '%s/configs/experiments/musique_expand60k_%shop_retrieval_ablation.yaml\n' "${ROOT_DIR}" "$1" ;;
    *) echo "Unsupported hop: $1" >&2; exit 1 ;;
  esac
}

for hop in "${HOPS[@]}"; do
  dataset="musique_${hop}hop"
  config_path="$(config_for_hop "${hop}")"
  for retriever in "${RETRIEVERS[@]}"; do
    for chunk_size in "${CHUNK_SIZES[@]}"; do
      for overlap in "${OVERLAPS[@]}"; do
        if (( overlap >= chunk_size )); then
          echo "Invalid chunk configuration c${chunk_size}_o${overlap}" >&2
          exit 1
        fi
        run_leaf="c${chunk_size}_o${overlap}"
        run_dir="${OUTPUT_ROOT}/${dataset}/${retriever}/${run_leaf}"
        evaluation_dir="${EVALUATION_ROOT}/${dataset}/${retriever}/${run_leaf}"
        ranking_file="${run_dir}/retrieval/retrieval_payloads__${retriever}__late_chunking__per_document.jsonl"
        metrics_file="${evaluation_dir}/metrics_summary.json"

        cmd=(
          "${PYTHON_BIN}" "${ROOT_DIR}/run_late_chunking_experiment.py"
          --dataset-name "${dataset}"
          --default-experiment "${config_path}"
          --retriever "${retriever}"
          --run-name "${retriever}/${run_leaf}"
          --output-root "${OUTPUT_ROOT}"
          --chunking-strategy fixed
          --chunk-size "${chunk_size}"
          --chunk-overlap "${overlap}"
          --chunk-tokenizer-name "${CHUNK_TOKENIZER_NAME}"
          --retrieve-k "${RETRIEVE_K}"
          --retrieval-scope "${RETRIEVAL_SCOPE}"
          --late-max-tokens-per-forward "${LATE_MAX_TOKENS_PER_FORWARD}"
          --late-window-overlap-tokens "${LATE_WINDOW_OVERLAP_TOKENS}"
        )
        if [[ "${FORCE_RERUN}" == "1" || "${RESUME}" == "0" ]]; then
          cmd+=(--no-resume)
        else
          cmd+=(--resume)
        fi

        if [[ -f "${ranking_file}" && "${FORCE_RERUN}" != "1" ]]; then
          echo "Skipping completed retrieval: ${run_dir}"
        elif [[ "${DRY_RUN}" == "1" ]]; then
          printf 'DRY_RUN command:'
          printf ' %q' "${cmd[@]}"
          printf '\n'
        else
          log_file="${LOG_DIR}/late_${dataset}_${retriever}_${run_leaf}_$(date +%Y%m%d_%H%M%S).log"
          "${cmd[@]}" 2>&1 | tee "${log_file}"
        fi

        if [[ "${RUN_EVALUATION}" == "1" && "${DRY_RUN}" != "1" && -f "${ranking_file}" ]]; then
          if [[ ! -f "${metrics_file}" || "${FORCE_RERUN}" == "1" ]]; then
            "${PYTHON_BIN}" "${ROOT_DIR}/evaluate_retrieval_run.py" \
              --run-dir "${run_dir}" \
              --output-dir "${evaluation_dir}" \
              --method-name "${retriever}" \
              --dataset-name "${dataset}" \
              --split validation \
              --run-name "${retriever}/${run_leaf}" \
              --ks 5 10
          else
            echo "Skipping completed evaluation: ${evaluation_dir}"
          fi
        fi
      done
    done
  done
done

if [[ "${RUN_EVALUATION}" == "1" && "${GENERATE_TABLE}" == "1" && "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/tables/generate_late_chunk_mega_table.py" \
    --input-root "${EVALUATION_ROOT}" \
    --output-file "${TABLE_OUTPUT}"
fi
