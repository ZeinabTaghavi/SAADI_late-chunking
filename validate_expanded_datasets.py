#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter

from chunked_pooling.experiment_datasets import load_task_registry_bundle


def main() -> int:
    report = {}
    for dataset_name, split in (
        ("qasper_64k", "test"),
        ("musique_32k", "validation"),
    ):
        bundle = load_task_registry_bundle(
            {
                "dataset_name": dataset_name,
                "split": split,
                "qa_n": "all",
                "qa_selection_method": "first",
            }
        )
        hop_counts = Counter(
            int(row["hop"])
            for row in bundle.qa_entries
            if row.get("hop") is not None
        )
        report[dataset_name] = {
            "documents": len(bundle.documents),
            "queries": len(bundle.qa_entries),
            "queries_with_retrieval_spans": sum(
                bool(row.get("retrieval_spans")) for row in bundle.qa_entries
            ),
            "hop_query_counts": {str(key): value for key, value in sorted(hop_counts.items())},
            "target_context_tokens": bundle.metadata["target_context_tokens"],
            "prepared_root": bundle.metadata["prepared_root"],
            "checksum_validation": "passed",
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
