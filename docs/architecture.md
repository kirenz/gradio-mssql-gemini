# Architecture

The repository splits reusable code from learn-by-doing examples so you can grow in either direction.

## Packages

- `src/quarto_mssql_gemini/` – Python helpers shared by Quarto notebooks, Gradio apps, and scripts.
  - `config.py` loads environment variables exactly once and exposes typed settings.
  - `data_access.py` builds SQLAlchemy engines, runs parameterised queries, and aggregates the sales dataset.
  - `ai/` wraps Gemini SDKs (both `google-genai` and `google-generativeai`) behind simple `generate_text`/`get_plot_description` helpers.

Import the package from anywhere (Quarto, scripts, Gradio) after adding `src/` to `PYTHONPATH`. The helper scripts and examples do this automatically.

## Examples

- `examples/sql-assistant/` – Gradio chat surfaces that translate natural language to SQL and optionally plot results or stream speech.
- `examples/sales-dashboard/` – Interactive SARIMAX forecasting lab with image/CSV/Excel exports for ad-hoc analysis.
- `examples/pdf-briefing/` – Quarto + Gemini workflow that parameterises a templated PDF briefing.
- `examples/ppt-deck/` – Placeholder for a forthcoming Quarto-to-PowerPoint pipeline (structure in place for contributors).

Each example folder includes:

- `README.md` outlining prerequisites, what you will learn, and run commands.
- An `outputs/` directory (gitignored) for generated artefacts.
- Supporting configuration files such as `current_filters.json` or Quarto notebooks.

## Assets and Scripts

- `assets/` stores shared resources such as the database schema description used by the SQL assistant prompts.
- `scripts/` offers onboarding-focused helpers:
  - `render_dashboard.py` makes sure tooling is installed, checks the database connection, and renders the PDF briefing.
  - `render_all.sh` runs every render helper sequentially.

## Documentation Loop

- `docs/quickstart.md` – fast onboarding checklist.
- `docs/architecture.md` – this document.
- `docs/extending.md` – guidance for adding new prompts, charts, or Quarto stories while staying idiomatic to the repo.
