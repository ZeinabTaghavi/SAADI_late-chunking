# Retrieval Evaluation

This repo already writes reusable raw retrieval artifacts under a run directory, so evaluation can be added afterward without rerunning chunking, indexing, or retrieval.

The compact evaluator prefers:

- `retrieval/retrieval_payloads__<retriever>__late_chunking__per_document.jsonl`
- fallback: `retrieval/retrieval_results_raw__<retriever>__late_chunking__per_document.json`

Both are read from an existing `--run-dir`. The evaluator joins those rows with an external labels file and writes only four compact outputs:

- `metrics_summary.json`
- `metrics_per_query.jsonl`
- `leaderboard_row.json`
- `evaluation_manifest.json`

## Usage

```bash
python3 evaluate_retrieval_run.py \
  --run-dir late_chunk_runs/qasper/jina/c300_o0 \
  --labels-file path/to/labels.json \
  --output-dir late_chunk_runs/qasper/jina/c300_o0/evaluation \
  --method-name late_chunking \
  --dataset-name qasper \
  --split test \
  --ks 5 10
```

Or with the shell wrapper:

```bash
bash scripts/run_retrieval_evaluation.sh \
  --run-dir late_chunk_runs/qasper/jina/c300_o0 \
  --labels-file path/to/labels.json \
  --output-dir late_chunk_runs/qasper/jina/c300_o0/evaluation \
  --method-name late_chunking \
  --dataset-name qasper \
  --split test \
  --ks 5 10
```

## Labels

The evaluator accepts JSON or JSONL label rows keyed by `query_id`. It prefers:

- `gold_chunk_ids`
- fallback `silver_chunk_ids`
- fallback `relevant_ids`

If a requested primary relevance field is unavailable for a query, that query gets `null` metrics and the reason is recorded in `evaluation_manifest.json`.

`silver_chunk_groups` are preserved in label loading for provenance, but they are not enough by themselves to compute the requested binary ranking metrics. If a labels file only contains grouped silver support without flat relevant ids, the manifest records that limitation and the summary metrics stay `null`.
