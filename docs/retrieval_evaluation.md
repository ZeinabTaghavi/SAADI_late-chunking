# Retrieval Evaluation

This repo already writes reusable raw retrieval artifacts under a run directory, so evaluation can be added afterward without rerunning chunking, indexing, or retrieval.

The compact evaluator prefers:

- `retrieval/retrieval_payloads__<retriever>__late_chunking__per_document.jsonl`
- fallback: `retrieval/retrieval_results_raw__<retriever>__late_chunking__per_document.json`

Both are read from an existing `--run-dir`. The evaluator can either join with an external labels file or generate labels inside the pipeline for supported datasets such as QASPER, LooGLE, and NarrativeQA. By default, it mirrors the run path under `late_chunk_evaluations/` before writing the four compact outputs:

- `metrics_summary.json`
- `metrics_per_query.jsonl`
- `leaderboard_row.json`
- `evaluation_manifest.json`

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

## Labels

When `--labels-file` is provided, the evaluator accepts JSON or JSONL label rows keyed by `query_id`. It prefers:

- `gold_chunk_ids`
- fallback `silver_chunk_ids`
- fallback `relevant_ids`

If a requested primary relevance field is unavailable for a query, that query gets `null` metrics and the reason is recorded in `evaluation_manifest.json`.

`silver_chunk_groups` are preserved in label loading for provenance, but they are not enough by themselves to compute the requested binary ranking metrics. If a labels file only contains grouped silver support without flat relevant ids, the manifest records that limitation and the summary metrics stay `null`.

## Internal QASPER, LooGLE, And NarrativeQA Labels

For QASPER, LooGLE, and NarrativeQA runs, labels are generated from:

- `selection/qa_entries.json` for the selected questions and evidence spans
- `chunking/<doc_id>/chunks.jsonl` for the exact chunk boundaries of that experiment

This means `gold_chunk_ids`, `silver_chunk_ids`, and `silver_chunk_groups` are recomputed per run, so changes in chunk size or overlap automatically change the labels as well.

For NarrativeQA specifically, the loader does not expose retrieval evidence spans, so internal labeling falls back to answer-text matching against the run's chunks. That keeps the evaluation run-specific, but it is weaker supervision than span-based evidence labels.
