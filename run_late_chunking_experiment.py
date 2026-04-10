import click

from chunked_pooling.experiment_config import (
    load_yaml_file,
    parse_retriever_specs,
    resolve_run_config,
)
from chunked_pooling.late_chunk_runner import run_late_chunking_experiment


@click.command()
@click.option("--dataset-name", required=True, help="Dataset preset or loader dataset name.")
@click.option(
    "--default-experiment",
    "default_experiment_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to the reference YAML that provides the default experiment settings.",
)
@click.option(
    "--retriever",
    "retriever_specs",
    multiple=True,
    help=(
        "Retriever spec. Supports aliases like 'jina', 'contriever', 'bm25' or "
        "comma-separated key=value pairs like "
        "'name=bge,type=dense,model_name=BAAI/bge-base-en-v1.5'."
    ),
)
@click.option("--run-name", default=None, help="Optional explicit run name.")
@click.option(
    "--output-root",
    default=None,
    type=click.Path(file_okay=False, path_type=str),
    help="Optional override for the run output root directory.",
)
@click.option("--resume/--no-resume", default=True, help="Reuse existing artifacts when practical.")
def main(
    dataset_name,
    default_experiment_path,
    retriever_specs,
    run_name,
    output_root,
    resume,
):
    default_experiment = load_yaml_file(default_experiment_path)
    retrievers = parse_retriever_specs(retriever_specs, default_experiment)
    resolved_config, notes = resolve_run_config(
        dataset_name=dataset_name,
        default_experiment=default_experiment,
        retrievers=retrievers,
        run_name_override=run_name,
        output_root_override=output_root,
        resume=resume,
    )
    run_dir = run_late_chunking_experiment(
        resolved_config=resolved_config,
        default_experiment_path=default_experiment_path,
        notes=notes,
    )
    click.echo(str(run_dir))


if __name__ == "__main__":
    raise SystemExit(main())
