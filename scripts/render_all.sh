#!/usr/bin/env bash
# Render all Quarto examples in one go.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Rendering PDF briefing..."
uv run python "${ROOT_DIR}/scripts/render_dashboard.py"

echo "Rendering PPT deck..."
uv run quarto render "${ROOT_DIR}/examples/ppt-deck/ppt_deck.qmd" --execute

echo "All renders completed."
