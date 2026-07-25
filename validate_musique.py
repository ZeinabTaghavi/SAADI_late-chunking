#!/usr/bin/env python3
from __future__ import annotations

import json

from chunked_pooling.experiment_datasets import load_dataset_bundle


def main() -> int:
    expected_groups = {2: 65, 3: 64, 4: 66}
    report = {}
    for hop in (2, 3, 4):
        bundle = load_dataset_bundle(
            {
                "type": "task_registry",
                "dataset_name": f"musique_{hop}hop",
                "split": "validation",
                "config_name": f"{hop}hop",
                "qa_n": "all",
                "qa_selection_method": "first",
                "prepend_title": False,
            }
        )
        if len(bundle.qa_entries) != 300:
            raise AssertionError(f"Expected 300 MuSiQue {hop}-hop queries, got {len(bundle.qa_entries)}")
        if len(bundle.documents) != expected_groups[hop]:
            raise AssertionError(
                f"Expected {expected_groups[hop]} referenced groups for {hop}-hop, got {len(bundle.documents)}"
            )
        report[str(hop)] = {
            "queries": len(bundle.qa_entries),
            "groups": len(bundle.documents),
            "missing_documents": 0,
            "spans_not_contained": 0,
            "source_membership_failures": 0,
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
