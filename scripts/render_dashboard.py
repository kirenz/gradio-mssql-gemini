#!/usr/bin/env python
"""
Helper script to run the PDF briefing Quarto example with friendly checks.

Usage:
    uv run python scripts/render_dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import shutil
from sqlalchemy import text


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


# Make shared helpers importable even when the script is executed via `uv run`.
_ensure_src_on_path()


def _check_quarto() -> None:
    if not shutil.which("quarto"):
        raise RuntimeError(
            "Quarto CLI is not on your PATH. Install it from https://quarto.org/ before rendering."
        )


def _check_database() -> None:
    get_settings()
    try:
        engine = create_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Unable to connect to SQL Server using your .env configuration. "
            "Update MSSQL_* variables and ensure the ODBC driver is installed."
        ) from exc
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


def render(example_path: Optional[Path] = None) -> Path:
    """Render the Quarto PDF briefing and return the generated file path."""

    repo_root = Path(__file__).resolve().parents[1]
    example_path = example_path or repo_root / "examples" / "pdf-briefing" / "sales_pdf.qmd"

    output_dir = example_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["quarto", "render", str(example_path), "--execute"]
    result = subprocess.run(cmd, cwd=example_path.parent, capture_output=True, text=True)
    if result.returncode != 0:  # pragma: no cover - rely on subprocess for detail
        raise RuntimeError(f"Quarto render failed:\n{result.stderr.strip()}")

    pdf_path = output_dir / "sales_pdf.pdf"
    if not pdf_path.exists():
        fallback_pdf = example_path.with_suffix(".pdf")
        if fallback_pdf.exists():
            shutil.move(str(fallback_pdf), pdf_path)
        else:
            raise FileNotFoundError(
                f"Expected PDF not found at {pdf_path}. Inspect Quarto output for clues."
            )
    return pdf_path


if __name__ == "__main__":
    _ensure_src_on_path()
    from quarto_mssql_gemini import create_engine, get_settings

    try:
        _check_quarto()
        _check_database()
        pdf_file = render()
    except Exception as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    print(f"✅ Render complete! Open {pdf_file}")
