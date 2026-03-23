"""CLI entry point for the Samhati training pipeline."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from samhati_pipeline.collector import SwarmRoundCollector, stats
from samhati_pipeline.config import DomainConfig, PipelineConfig

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _load_config(config_path: str | None) -> PipelineConfig:
    if config_path:
        return PipelineConfig.from_yaml(Path(config_path))
    return PipelineConfig()


@click.group()
@click.option("--config", "config_path", default=None, help="Path to pipeline config YAML.")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """Samhati self-evolving training pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config_path)


@cli.command()
@click.option("--round-file", required=True, type=click.Path(exists=True), help="Path to a round JSON file.")
@click.pass_context
def collect(ctx: click.Context, round_file: str) -> None:
    """Process a swarm round file and collect training examples."""
    cfg: PipelineConfig = ctx.obj["config"]
    collector = SwarmRoundCollector(cfg.data_dir, cfg.min_confidence)

    with open(round_file) as f:
        data = json.load(f)

    # Support both single round and list of rounds
    rounds = data if isinstance(data, list) else [data]
    collected = 0

    for round_data in rounds:
        example = collector.collect_and_save(round_data)
        if example is not None:
            collected += 1
            console.print(
                f"  [green]+[/green] {example.domain_tags[0]:12s} "
                f"conf={example.confidence:.2f} "
                f"round={example.round_id[:12]}..."
            )

    console.print(f"\nCollected [bold]{collected}[/bold] / {len(rounds)} rounds.")


@cli.command(name="stats")
@click.pass_context
def show_stats(ctx: click.Context) -> None:
    """Show dataset statistics across all domains."""
    cfg: PipelineConfig = ctx.obj["config"]
    data = stats(cfg.data_dir)

    if data["total"] == 0:
        console.print("[yellow]No training examples collected yet.[/yellow]")
        return

    table = Table(title="Samhati Training Data")
    table.add_column("Domain", style="cyan")
    table.add_column("Examples", justify="right")
    table.add_column("Earliest", style="dim")
    table.add_column("Latest", style="dim")

    for domain, info in sorted(data["domains"].items()):
        table.add_row(
            domain,
            str(info["count"]),
            (info.get("earliest") or "")[:10],
            (info.get("latest") or "")[:10],
        )

    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{data['total']}[/bold]",
        (data.get("earliest") or "")[:10],
        (data.get("latest") or "")[:10],
    )

    console.print(table)


@cli.command()
@click.option("--domain", required=True, help="Domain to train.")
@click.option("--base-model", default=None, help="Override base model.")
@click.pass_context
def train(ctx: click.Context, domain: str, base_model: str | None) -> None:
    """Train a domain specialist SLM (SFT + DPO)."""
    from samhati_pipeline.collector import load_examples
    from samhati_pipeline.dataset import build_dpo_dataset, build_sft_dataset
    from samhati_pipeline.trainer import SamhatiTrainer

    cfg: PipelineConfig = ctx.obj["config"]
    effective_model = base_model or cfg.base_model

    console.print(f"Loading examples for domain [cyan]{domain}[/cyan]...")
    examples = load_examples(cfg.data_dir, domain)
    if not examples:
        console.print(f"[red]No examples found for domain '{domain}'.[/red]")
        sys.exit(1)

    console.print(f"Found [bold]{len(examples)}[/bold] examples.")

    # SFT
    sft_dataset = build_sft_dataset(examples, model_name=effective_model)
    console.print(f"SFT dataset: {len(sft_dataset)} rows")

    trainer = SamhatiTrainer(cfg)
    sft_path = trainer.train_sft(sft_dataset, domain, base_model=effective_model)
    console.print(f"[green]SFT adapter saved:[/green] {sft_path}")

    # DPO
    dpo_dataset = build_dpo_dataset(examples, model_name=effective_model)
    if len(dpo_dataset) > 0:
        console.print(f"DPO dataset: {len(dpo_dataset)} rows")
        dpo_path = trainer.train_dpo(dpo_dataset, domain, sft_path, base_model=effective_model)
        console.print(f"[green]DPO adapter saved:[/green] {dpo_path}")
    else:
        console.print("[yellow]No DPO pairs available, SFT-only model.[/yellow]")


@cli.command()
@click.option("--model-path", required=True, help="Path to merged model or adapter.")
@click.option("--domain", required=True, help="Domain name.")
@click.option("--base-model", default=None, help="Base model (needed if model-path is an adapter).")
@click.option("--quantization", default="Q4_K_M", help="GGUF quantization level.")
@click.option("--push/--no-push", default=False, help="Push to HuggingFace Hub.")
@click.pass_context
def export(
    ctx: click.Context,
    model_path: str,
    domain: str,
    base_model: str | None,
    quantization: str,
    push: bool,
) -> None:
    """Export a trained model to GGUF format."""
    from datetime import datetime, timezone

    from samhati_pipeline.exporter import (
        export_gguf,
        generate_model_card,
        push_to_hub,
    )
    from samhati_pipeline.trainer import SamhatiTrainer

    cfg: PipelineConfig = ctx.obj["config"]

    # If it's an adapter, merge first
    model_p = Path(model_path)
    if (model_p / "adapter_config.json").exists():
        if not base_model:
            base_model = cfg.base_model
        console.print("Merging adapter into base model...")
        model_path = SamhatiTrainer.merge_adapter(base_model, model_path)

    output = str(cfg.output_dir / f"gguf-{domain}")
    console.print(f"Exporting to GGUF ({quantization})...")
    gguf_path = export_gguf(model_path, output, quantization)
    console.print(f"[green]GGUF exported:[/green] {gguf_path}")

    if push:
        version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        card = generate_model_card(domain, {"base_model": base_model or cfg.base_model})
        url = push_to_hub(gguf_path, cfg.hf_repo_prefix, domain, version, card)
        console.print(f"[green]Pushed to:[/green] {url}")


@cli.command()
@click.option("--domain", required=True, help="Domain to run full pipeline for.")
@click.option("--base-model", default=None, help="Override base model.")
@click.pass_context
def run(ctx: click.Context, domain: str, base_model: str | None) -> None:
    """Run the full pipeline for a domain: collect, train, export."""
    from samhati_pipeline.scheduler import TrainingScheduler

    cfg: PipelineConfig = ctx.obj["config"]
    domains_file = Path("configs/domains.yaml")
    domain_configs = DomainConfig.load_all(domains_file) if domains_file.exists() else {}

    scheduler = TrainingScheduler(cfg, domain_configs)
    console.print(f"Running full pipeline for domain [cyan]{domain}[/cyan]...")

    result = scheduler.run_domain_pipeline(domain, base_model=base_model)
    console.print(f"\n[green]Pipeline complete![/green]")
    console.print(json.dumps(result, indent=2))


@cli.command()
@click.option("--check-interval", default=86400, type=int, help="Seconds between checks (default: 1 day).")
@click.pass_context
def schedule(ctx: click.Context, check_interval: int) -> None:
    """Run in daemon mode, checking daily for domains ready to train."""
    from samhati_pipeline.scheduler import TrainingScheduler

    cfg: PipelineConfig = ctx.obj["config"]
    domains_file = Path("configs/domains.yaml")
    domain_configs = DomainConfig.load_all(domains_file) if domains_file.exists() else {}

    scheduler = TrainingScheduler(cfg, domain_configs)
    console.print("[bold]Samhati training scheduler started.[/bold]")
    console.print(f"Checking every {check_interval}s for domains with enough data.\n")

    while True:
        console.print(f"[dim]Checking domains...[/dim]")
        results = scheduler.check_and_run()
        for domain, info in results.items():
            status = info.get("status", "unknown")
            if status == "success":
                console.print(f"  [green]{domain}[/green]: trained in {info.get('elapsed_seconds', '?')}s")
            elif status == "skipped":
                console.print(f"  [dim]{domain}[/dim]: skipped ({info.get('reason', '')})")
            else:
                console.print(f"  [red]{domain}[/red]: {info.get('error', 'unknown error')}")

        console.print(f"\nNext check in {check_interval}s...")
        time.sleep(check_interval)
