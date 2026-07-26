#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-late_chunk_runs}"
EVAL_ROOT="${EVAL_ROOT:-late_chunk_evaluations}"
TABLE_PATH="${TABLE_PATH:-docs/qasper_musique_c250_retrieval_table.tex}"
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

export EVAL_ROOT TABLE_PATH CHUNK_FOLDER DATASETS RETRIEVERS
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

eval_root = Path(os.environ["EVAL_ROOT"])
table_path = Path(os.environ["TABLE_PATH"])
chunk_folder = os.environ["CHUNK_FOLDER"]
datasets = os.environ["DATASETS"].split()
retrievers = os.environ["RETRIEVERS"].split()

dataset_labels = {
    "qasper": "QASPER",
    "musique_2hop": "MuSiQue-2hop",
    "musique_3hop": "MuSiQue-3hop",
    "musique_4hop": "MuSiQue-4hop",
}
retriever_labels = {
    "jina-v3": "Jina-v3",
    "qwen": "Qwen",
    "contriever": "Contriever",
    "bm25": "BM25",
    "bge-m3": "BGE-M3",
}
keys = [
    "gold_ndcg@10",
    "gold_recall@10",
    "silver_loose_ndcg@10",
    "silver_loose_recall@10",
    "union_loose_ndcg@10",
    "union_loose_recall@10",
    "gold_hit@5",
    "gold_hit@10",
    "silver_strict_hit@5",
    "silver_strict_hit@10",
    "strict_union_hit@5",
    "strict_union_hit@10",
]


def fmt(value):
    if value is None:
        return "--"
    return f"{100 * float(value):.1f}"


rows = []
missing = []
for dataset in datasets:
    for retriever in retrievers:
        path = eval_root / dataset / retriever / chunk_folder / "leaderboard_row.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            [
                dataset_labels.get(dataset, dataset),
                retriever_labels.get(retriever, retriever),
                "Late Chunking (250/0)",
                *[fmt(row.get(key)) for key in keys],
            ]
        )

if missing:
    raise SystemExit(
        "Missing evaluated leaderboard files:\n" + "\n".join(f"  {item}" for item in missing)
    )
if not rows:
    raise SystemExit("No leaderboard rows found; table was not generated.")

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Main retrieval summary for QASPER and MuSiQue with chunk size 250. Values are percentages.}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lllrrrrrrrrrrrr}",
    r"\toprule",
    r"Dataset & Retriever & Method & \multicolumn{6}{c}{Ranking Metrics} & \multicolumn{6}{c}{Binary Metrics} \\",
    r"\cmidrule(lr){4-9}\cmidrule(lr){10-15}",
    r"& & & \multicolumn{2}{c}{Gold} & \multicolumn{2}{c}{Silver-L} & \multicolumn{2}{c}{Union-L} & \multicolumn{2}{c}{Gold} & \multicolumn{2}{c}{Silver-S} & \multicolumn{2}{c}{Union-S} \\",
    r"& & & NDCG@10 & Recall@10 & NDCG@10 & Recall@10 & NDCG@10 & Recall@10 & HR@5 & HR@10 & HR@5 & HR@10 & HR@5 & HR@10 \\",
    r"\midrule",
]

last_dataset = None
last_retriever = None
for row in rows:
    display = list(row)
    if display[0] == last_dataset:
        display[0] = ""
    else:
        last_dataset = row[0]
        last_retriever = None
    if display[1] == last_retriever:
        display[1] = ""
    else:
        last_retriever = row[1]
    lines.append(" & ".join(display) + r" \\")

lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
)

table_path.parent.mkdir(parents=True, exist_ok=True)
table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(table_path)
PY

printf 'Finished QASPER/MuSiQue evaluation and table generation.\n'
