# Gradio + Gemini + MSSQL Toolkit

Reusable helpers, Quarto stories, and Gradio apps that sit on top of a Microsoft SQL Server dataset. Gemini handles the natural-language and narrative heavy lifting while SQL Server remains the single source of truth.

**Quick Start**

1. Install [uv](https://github.com/astral-sh/uv), [Quarto CLI](https://quarto.org/docs/download/), and an ODBC driver for SQL Server.

2. Clone the toolkit:
```bash
git clone https://github.com/kirenz/gradio-mssql-gemini.git
```
3. Enter the project folder:
```bash
cd gradio-mssql-gemini
```
4. Install dependencies:
```bash
uv sync
```
5. Copy the environment template:
```bash
cp .env.example .env
```
6. Edit `.env` to add MSSQL credentials and `GEMINI_API_KEY`.

7. Open VS Code in the project folder

```bash
code .
```

8. Render the sample dashboard:
```bash
uv run python scripts/render_dashboard.py
```

9. The render step opens `examples/pdf-briefing/outputs/sales_pdf.pdf`.

## Choose Application

| Experience | What happens | Command |
| --- | --- | --- |
| [SQL Assistant](examples/sql-assistant/README.md) | Chat with your data, auto-generate SQL, optional plots & speech. | `uv run python examples/sql-assistant/app_df.py` |
| [Sales Forecasting Lab](examples/sales-dashboard/README.md) | Filter, forecast (SARIMAX), and export artefacts. | `uv run python examples/sales-dashboard/forecasting_app.py` |
| [PDF Briefing](examples/pdf-briefing/README.md) | Parameterised Quarto + Typst report with Gemini commentary. | `uv run python examples/pdf-briefing/app.py` or `uv run python scripts/render_dashboard.py` |
| [PPT Deck](examples/ppt-deck/README.md) | Placeholder for a Quarto → PowerPoint workflow. | _Contribute your story_ |

All examples add `src/` to `PYTHONPATH` on launch, so imports Just Work™ once you have `uv sync`’d.

## Reusable helpers

- `src/quarto_mssql_gemini/config.py` – typed settings loader for MSSQL and Gemini credentials.
- `src/quarto_mssql_gemini/data_access.py` – database engine creation, query runners, schema introspection.
- `src/quarto_mssql_gemini/ai/` – wrappers around `google-genai`/`google-generativeai` for consistent copy.

Import from the package anywhere (Gradio app, Quarto notebook, script) to avoid duplicating boilerplate.

## Tooling aids

- `scripts/render_dashboard.py` – validates the environment and renders the PDF briefing end-to-end.
- `scripts/render_all.sh` – runs every render helper; extend it as you add new stories.
- `.gitignore` keeps generated artefacts in `examples/*/outputs/` out of version control. Each folder ships with an `outputs/.gitignore` so you can safely generate locally.

## Learn more

- [Quick Start](docs/quickstart.md) – screenshot-free checklist to get productive fast.
- [Architecture](docs/architecture.md) – explains the new project layout.
- [Extending](docs/extending.md) – how to add prompts, scripts, or brand-new examples.


