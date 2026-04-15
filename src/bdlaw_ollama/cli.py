from __future__ import annotations

import json
from pathlib import Path

import click
from .config import load_app_config


@click.group()
def main() -> None:
    """Bangladesh law hybrid RAG + Ollama CLI."""


@main.command("prepare-data")
@click.option("--root", default=".", type=click.Path(path_type=Path))
def prepare_data(root: Path) -> None:
    try:
        from .chunking import build_chunks, write_chunks
        from .ingest import load_canonical_laws, write_canonical_laws

        config = load_app_config(root)
        laws = load_canonical_laws(config)
        write_canonical_laws(config, laws)
        chunks = build_chunks(laws, config.rag)
        write_chunks(config, chunks)
        click.echo(json.dumps({"laws": len(laws), "chunks": len(chunks)}, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("build-index")
@click.option("--root", default=".", type=click.Path(path_type=Path))
def build_index_cmd(root: Path) -> None:
    try:
        from .retrieval import build_index

        config = load_app_config(root)
        result = build_index(config)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("make-sft")
@click.option("--root", default=".", type=click.Path(path_type=Path))
def make_sft(root: Path) -> None:
    try:
        from .sft_data import make_sft_datasets

        config = load_app_config(root)
        result = make_sft_datasets(config)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("train")
@click.option("--root", default=".", type=click.Path(path_type=Path))
@click.option("--config", "train_config", default=None, type=click.Path(path_type=Path))
@click.option("--smoke", is_flag=True, default=False)
@click.option("--max-train-samples", default=None, type=int)
def train_cmd(root: Path, train_config: Path | None, smoke: bool, max_train_samples: int | None) -> None:
    try:
        from .train import run_training

        config = load_app_config(root)
        result = run_training(config, train_config_path=train_config, smoke=smoke, max_train_samples=max_train_samples)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("export-ollama")
@click.option("--root", default=".", type=click.Path(path_type=Path))
@click.option("--llama-cpp-dir", default=None, type=click.Path(path_type=Path))
@click.option("--skip-create", is_flag=True, default=False)
@click.option("--push-target", default=None)
def export_ollama_cmd(root: Path, llama_cpp_dir: Path | None, skip_create: bool, push_target: str | None) -> None:
    try:
        from .export_ollama import export_for_ollama

        config = load_app_config(root)
        result = export_for_ollama(
            config,
            llama_cpp_dir=str(llama_cpp_dir) if llama_cpp_dir else None,
            skip_create=skip_create,
            push_target=push_target,
        )
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("serve-api")
@click.option("--root", default=".", type=click.Path(path_type=Path))
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
def serve_api(root: Path, host: str, port: int) -> None:
    try:
        import uvicorn

        from .api import create_app

        app = create_app(root=root)
        uvicorn.run(app, host=host, port=port)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("query")
@click.argument("question")
@click.option("--root", default=".", type=click.Path(path_type=Path))
@click.option("--model", default=None)
@click.option("--top-k", default=None, type=int)
def query_cmd(question: str, root: Path, model: str | None, top_k: int | None) -> None:
    try:
        from .api import run_query_pipeline

        result = run_query_pipeline(question, root=root, model_name=model, top_k=top_k)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("preflight")
def preflight() -> None:
    try:
        from .train import preflight_training

        click.echo(json.dumps(preflight_training(), ensure_ascii=False, indent=2))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("evaluate")
@click.option("--root", default=".", type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
@click.option("--model", default=None)
def evaluate_cmd(root: Path, limit: int | None, model: str | None) -> None:
    try:
        from .evaluation import run_benchmark

        config = load_app_config(root)
        result = run_benchmark(config, limit=limit, model_name=model)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
