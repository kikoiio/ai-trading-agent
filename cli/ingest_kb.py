"""CLI script for ingesting PPTX files into the Al Brooks knowledge base.

Usage:
    python -m cli.ingest_kb --source ./Al_Brooks_trading_course/
    python -m cli.ingest_kb --source ./Al_Brooks_trading_course/ --reset
    python -m cli.ingest_kb --source ./Al_Brooks_trading_course/ --skip-images
    python -m cli.ingest_kb --source ./Al_Brooks_trading_course/ --chapters 01,05,12
    python -m cli.ingest_kb --stats
"""

import argparse
import logging
import sys
import pathlib

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Al Brooks PPTX course materials into the trading knowledge base"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Path to Al_Brooks_trading_course/ folder containing chapter subfolders with PPTX files",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="Vision model for image descriptions (default: from KB_VISION_MODEL env or openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing KB before ingesting",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip vision processing of images (text-only ingestion)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between vision API calls (default: 1.0)",
    )
    parser.add_argument(
        "--chapters",
        type=str,
        default=None,
        help="Comma-separated chapter numbers to process (e.g., 01,05,12)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show knowledge base statistics and exit",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from src.config_loader import CONFIG

    data_dir = CONFIG.get("kb_data_dir") or "data/kb"

    if args.stats:
        from src.kb.vectorstore import KBVectorStore
        import os

        store = KBVectorStore(persist_dir=os.path.join(data_dir, "chroma_db"))
        stats = store.stats()

        table = Table(title="Knowledge Base Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Entries", str(stats["total_entries"]))
        table.add_row("Chapters", str(stats["num_chapters"]))
        for ch in stats.get("chapters", []):
            table.add_row(f"  - {ch}", "")
        console.print(table)
        return

    if not args.source:
        parser.error("--source is required (or use --stats)")

    chapters_filter = None
    if args.chapters:
        chapters_filter = [c.strip().zfill(2) for c in args.chapters.split(",")]

    from src.kb.ingest import ingest_pptx_folder

    console.print(f"\n[bold]Al Brooks Knowledge Base Ingestion[/bold]")
    console.print(f"Source: {args.source}")
    console.print(f"Vision model: {args.vision_model or CONFIG.get('kb_vision_model') or 'openai/gpt-4o-mini'}")
    console.print(f"Skip images: {args.skip_images}")
    console.print(f"Reset: {args.reset}")
    if chapters_filter:
        console.print(f"Chapters filter: {chapters_filter}")
    console.print()

    try:
        ingest_pptx_folder(
            source_dir=args.source,
            data_dir=data_dir,
            vision_model=args.vision_model,
            skip_images=args.skip_images,
            reset=args.reset,
            delay=args.delay,
            chapters_filter=chapters_filter,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Ingestion interrupted by user[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
