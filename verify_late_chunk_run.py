from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def verify_run(run_dir: Path, retriever_name: str) -> List[str]:
    errors: List[str] = []
    required_files = [
        run_dir / "run_manifest.json",
        run_dir / "config" / "late_chunking_run.yaml",
        run_dir / "selection" / "selected_doc_ids.json",
        run_dir / "selection" / "qa_entries.json",
        run_dir / "indexing" / "index_manifest.json",
        run_dir / "indexing" / retriever_name / "index_stats.json",
    ]
    payload_path = (
        run_dir
        / "retrieval"
        / f"retrieval_payloads__{retriever_name}__late_chunking__per_document.jsonl"
    )
    raw_results_path = (
        run_dir
        / "retrieval"
        / f"retrieval_results_raw__{retriever_name}__late_chunking__per_document.json"
    )
    required_files.extend([payload_path, raw_results_path])

    for path in required_files:
        if not path.is_file():
            errors.append(f"missing required artifact: {path}")
    if errors:
        return errors

    try:
        manifest = _read_json(run_dir / "run_manifest.json")
        if retriever_name not in manifest.get("retrievers_used", []):
            errors.append(
                f"run manifest does not list retriever {retriever_name!r}"
            )

        qa_entries = _read_json(run_dir / "selection" / "qa_entries.json")
        if not isinstance(qa_entries, list):
            errors.append("selection/qa_entries.json is not a list")
            qa_entries = []
        expected_query_ids = [str(row["query_id"]) for row in qa_entries]

        payload_rows = _read_jsonl(payload_path)
        payload_query_ids = [str(row.get("query_id")) for row in payload_rows]
        if payload_query_ids != expected_query_ids:
            errors.append(
                "retrieval payload query ids/count do not match selected QA entries"
            )

        raw_results = _read_json(raw_results_path)
        if not isinstance(raw_results, list):
            errors.append("raw retrieval results are not a list")
        elif [str(row.get("query_id")) for row in raw_results] != expected_query_ids:
            errors.append(
                "raw retrieval result query ids/count do not match selected QA entries"
            )

        if retriever_name != "bm25":
            matrix_path = (
                run_dir
                / "indexing"
                / retriever_name
                / "chunk_embeddings.npy"
            )
            chunk_ids_path = (
                run_dir / "indexing" / retriever_name / "chunk_ids.json"
            )
            if not matrix_path.is_file():
                errors.append(f"missing dense embedding matrix: {matrix_path}")
            if not chunk_ids_path.is_file():
                errors.append(f"missing dense chunk-id file: {chunk_ids_path}")

            if matrix_path.is_file() and chunk_ids_path.is_file():
                chunk_ids = _read_json(chunk_ids_path)
                if not isinstance(chunk_ids, list):
                    errors.append("dense chunk_ids.json is not a list")
                    chunk_ids = []
                matrix = np.load(
                    matrix_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                if matrix.ndim != 2 or matrix.shape[0] != len(chunk_ids):
                    errors.append(
                        "dense embedding matrix row count does not match chunk_ids.json"
                    )

                selected_doc_ids = _read_json(
                    run_dir / "selection" / "selected_doc_ids.json"
                )
                expected_chunk_ids = []
                for doc_id in selected_doc_ids:
                    chunk_path = (
                        run_dir / "chunking" / str(doc_id) / "chunks.jsonl"
                    )
                    if not chunk_path.is_file():
                        errors.append(f"missing document chunks: {chunk_path}")
                        continue
                    expected_chunk_ids.extend(
                        str(row["chunk_id"]) for row in _read_jsonl(chunk_path)
                    )
                if chunk_ids != expected_chunk_ids:
                    errors.append(
                        "dense chunk_ids.json does not match the selected document chunks"
                    )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"artifact validation failed: {exc}")

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that a late-chunking retrieval run is complete and reusable."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--retriever-name", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run_dir = args.run_dir.resolve()
    errors = verify_run(run_dir, args.retriever_name)
    if errors:
        if not args.quiet:
            print(f"Incomplete run: {run_dir}")
            for error in errors:
                print(f"  - {error}")
        return 1
    if not args.quiet:
        print(f"Complete run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
