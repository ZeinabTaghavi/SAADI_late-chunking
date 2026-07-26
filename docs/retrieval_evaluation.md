# Retrieval Evaluation

This repo already writes reusable raw retrieval artifacts under a run directory, so evaluation can be added afterward without rerunning chunking, indexing, or retrieval.

The compact evaluator prefers:

- `retrieval/retrieval_payloads__<retriever>__late_chunking__per_document.jsonl`
- fallback: `retrieval/retrieval_results_raw__<retriever>__late_chunking__per_document.json`

Both are read from an existing `--run-dir`. The evaluator can either join with an external labels file or generate labels inside the pipeline for supported datasets such as QASPER, MuSiQue, LooGLE, NarrativeQA, QuALITY, and NovelHopQA. By default, it mirrors the run path under `late_chunk_evaluations/` before writing the four compact outputs:

- `metrics_summary.json`
- `metrics_per_query.jsonl`
- `leaderboard_row.json`
- `evaluation_manifest.json`

The summary now reports the two metric families in parallel:

- Ranking metrics over loose relevance targets:
  - `gold`: Recall, MRR, and NDCG over `gold_chunk_ids`
  - `silver_loose`: Recall, MRR, and NDCG over flattened `silver_chunk_ids`
  - `union_loose`: Recall, MRR, and NDCG over the flattened union of gold and silver-loose ids
- `gold_hit`: HitRate@k for any gold chunk in top-k
- `silver_strict_hit`: HitRate@k for retrieving an entire `silver_chunk_group` in top-k
- `strict_union_hit`: HitRate@k for `gold_hit OR silver_strict_hit`

Example:

- run input: `late_chunk_runs/qasper/jina/c300_o0`
- default evaluation output: `late_chunk_evaluations/qasper/jina/c300_o0`

## Usage

```bash
python3 evaluate_retrieval_run.py \
  --run-dir late_chunk_runs/qasper/jina/c300_o0 \
  --method-name late_chunking \
  --dataset-name qasper \
  --split test \
  --ks 5 10
```

Or with the shell wrapper:

```bash
bash scripts/run_retrieval_evaluation.sh \
  --run-dir late_chunk_runs/qasper/jina/c300_o0 \
  --method-name late_chunking \
  --dataset-name qasper \
  --split test \
  --ks 5 10
```

If you want a non-default location, you can still pass `--output-dir`.

If you already have a separate labels file and want to use it instead of in-process generation, pass `--labels-file path/to/labels.json`.

## QASPER And Aggregated MuSiQue Table

The raw run, evaluation, and log directories are intentionally ignored by Git because they can be very large. Generate the small commit-ready report under `docs/` with:

```bash
bash scripts/tmp_eval_qasper_musique_c250_selected_retrievers_and_table.sh
```

The script verifies every retrieval run, skips evaluations whose four artifacts are already complete and newer than their retrieval payload, evaluates only missing or stale results, and then writes and prints:

- `docs/qasper_musique_c250_retrieval_table.tex`
- `docs/qasper_musique_c250_retrieval_table.json`

QASPER has one row per retriever. MuSiQue also has one row per retriever: its 2-hop, 3-hop, and 4-hop per-query results are concatenated and micro-averaged. The JSON file records the query count contributed by each hop and the non-null denominator for every metric.

To rebuild only the table from evaluation artifacts that already exist:

```bash
python3 tables/generate_qasper_musique_summary.py \
  --input-root late_chunk_evaluations \
  --output-tex docs/qasper_musique_c250_retrieval_table.tex \
  --output-json docs/qasper_musique_c250_retrieval_table.json \
  --chunk-folder c250_o0 \
  --print-table \
  --retrievers jina-v3 qwen contriever bm25 bge-m3
```

Unlike `late_chunk_runs/`, `late_chunk_evaluations/`, `logs/`, and `tables/late_chunking_mega_table.txt`, these two `docs/` outputs are not ignored. They therefore appear in `git status` and can be committed normally.

## Batch QASPER Runs

To evaluate every existing QASPER run you already have for both Jina and Qwen, use:

```bash
bash scripts/run_all_qasper_retrieval_evaluations.sh
```

By default it scans:

- `late_chunk_runs/qasper/jina/...`
- `late_chunk_runs/qasper/qwen/...`

and writes mirrored compact outputs under:

- `late_chunk_evaluations/qasper/jina/...`
- `late_chunk_evaluations/qasper/qwen/...`

Useful overrides:

- `RETRIEVERS="jina"` to evaluate only Jina runs
- `DRY_RUN=1` to print commands without executing them
- `STOP_ON_ERROR=0` to continue past failures
- `RUN_ROOT=/path/to/runs` and `EVAL_ROOT=/path/to/evals` to customize roots

## Batch LooGLE Runs

To evaluate every existing LooGLE run you already have for both Jina and Qwen, use:

```bash
bash scripts/run_all_loogle_retrieval_evaluations.sh
```

By default it scans:

- `late_chunk_runs/loogle/jina/...`
- `late_chunk_runs/loogle/qwen/...`

and writes mirrored compact outputs under:

- `late_chunk_evaluations/loogle/jina/...`
- `late_chunk_evaluations/loogle/qwen/...`

Useful overrides:

- `RETRIEVERS="jina"` to evaluate only Jina runs
- `DRY_RUN=1` to print commands without executing them
- `STOP_ON_ERROR=0` to continue past failures
- `RUN_ROOT=/path/to/runs` and `EVAL_ROOT=/path/to/evals` to customize roots

## Batch NarrativeQA Runs

To evaluate every existing NarrativeQA run you already have for both Jina and Qwen, use:

```bash
bash scripts/run_all_narrativeqa_retrieval_evaluations.sh
```

By default it scans:

- `late_chunk_runs/narrativeqa/jina/...`
- `late_chunk_runs/narrativeqa/qwen/...`

and writes mirrored compact outputs under:

- `late_chunk_evaluations/narrativeqa/jina/...`
- `late_chunk_evaluations/narrativeqa/qwen/...`

Useful overrides:

- `RETRIEVERS="jina"` to evaluate only Jina runs
- `DRY_RUN=1` to print commands without executing them
- `STOP_ON_ERROR=0` to continue past failures
- `RUN_ROOT=/path/to/runs` and `EVAL_ROOT=/path/to/evals` to customize roots

## Batch QuALITY Runs

To evaluate every existing QuALITY run you already have for both Jina and Qwen, use:

```bash
bash scripts/run_all_quality_retrieval_evaluations.sh
```

By default it scans:

- `late_chunk_runs/quality/jina/...`
- `late_chunk_runs/quality/qwen/...`

and writes mirrored compact outputs under:

- `late_chunk_evaluations/quality/jina/...`
- `late_chunk_evaluations/quality/qwen/...`

Useful overrides:

- `RETRIEVERS="jina"` to evaluate only Jina runs
- `DRY_RUN=1` to print commands without executing them
- `STOP_ON_ERROR=0` to continue past failures
- `RUN_ROOT=/path/to/runs` and `EVAL_ROOT=/path/to/evals` to customize roots

## Batch NovelHopQA Runs

To evaluate every existing NovelHopQA or NovelQA run you already have for both Jina and Qwen, use:

```bash
bash scripts/run_all_novelhopqa_retrieval_evaluations.sh
```

By default it scans both dataset root names:

- `late_chunk_runs/novelqa/jina/...`
- `late_chunk_runs/novelqa/qwen/...`
- `late_chunk_runs/novelhopqa/jina/...`
- `late_chunk_runs/novelhopqa/qwen/...`

and writes mirrored compact outputs under the matching dataset path in `late_chunk_evaluations/...`.

Useful overrides:

- `DATASET_NAMES="novelqa"` to scan only the `novelqa` tree
- `RETRIEVERS="jina"` to evaluate only Jina runs
- `DRY_RUN=1` to print commands without executing them
- `STOP_ON_ERROR=0` to continue past failures
- `RUN_ROOT=/path/to/runs` and `EVAL_ROOT=/path/to/evals` to customize roots

## Labels

When `--labels-file` is provided, the evaluator accepts JSON or JSONL label rows keyed by `query_id`. It prefers:

- `gold_chunk_ids`
- fallback `silver_chunk_ids`
- fallback `relevant_ids`

If a requested primary relevance field is unavailable for a query, the affected relevance-view metrics become `null` for that query and the reason is recorded in `evaluation_manifest.json`.

`silver_chunk_groups` are preserved in label loading for strict hit evaluation. They are not flattened into binary ranking relevance automatically unless `silver_chunk_ids` or fallback `relevant_ids` are present.

## Internal QASPER, MuSiQue, LooGLE, NarrativeQA, QuALITY, And NovelHopQA Labels

For QASPER, MuSiQue, LooGLE, NarrativeQA, QuALITY, and NovelHopQA runs, labels are generated from:

- `selection/qa_entries.json` for the selected questions and evidence spans
- `chunking/<doc_id>/chunks.jsonl` for the exact chunk boundaries of that experiment

This means `gold_chunk_ids`, `silver_chunk_ids`, and `silver_chunk_groups` are recomputed per run, so changes in chunk size or overlap automatically change the labels as well.

For NarrativeQA specifically, the loader does not expose retrieval evidence spans, so internal labeling falls back to answer-text matching against the run's chunks. That keeps the evaluation run-specific, but it is weaker supervision than span-based evidence labels.

For QuALITY specifically, the loader does not expose retrieval evidence spans, so internal labeling falls back to matching the gold answer choice text against the run's chunks. That keeps the evaluation run-specific, but it is weaker supervision than span-based evidence labels and may leave some queries without usable labels when the answer choice is abstractive.

For NovelHopQA specifically, the loader provides `gold_context_window` / `retrieval_spans` with `retrieval_span_mode="window"`, so labels are generated against the run's exact chunk boundaries using that context-window overlap logic.
