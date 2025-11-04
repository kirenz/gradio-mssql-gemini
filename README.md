# Gradio + Gemini + MSSQL Toolkit

Reusable helpers, Quarto stories, and Gradio apps that sit on top of a Microsoft SQL Server dataset. Gemini handles the natural-language and narrative heavy lifting while SQL Server remains the single source of truth.

**Quick Start**

1. Install [uv](https://github.com/astral-sh/uv), [Quarto CLI](https://quarto.org/docs/download/), and an ODBC driver for SQL Server.
   - PDF output requires a LaTeX distribution; run `quarto install tinytex` or install TeX Live/MiKTeX manually before rendering.

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

[SQL Assistant](examples/sql-assistant/README.md) helps you chat with your data, auto-generate SQL, and optionally return plots and speech.

```bash
uv run python examples/sql-assistant/app_df.py
```

[Sales Forecasting Lab](examples/sales-dashboard/README.md) lets you filter, run SARIMAX forecasts, and export artefacts.

```bash
uv run python examples/sales-dashboard/forecasting_app.py
```

[PDF Briefing](examples/pdf-briefing/README.md) renders a parameterised Quarto + Typst report with Gemini commentary.

```bash
uv run python examples/pdf-briefing/app.py
```
or render the dashboard directly

```bash
uv run python scripts/render_dashboard.py
```

[PPT Deck](examples/ppt-deck/README.md) ships as a Quarto template for a PowerPoint briefing. Rendering creates interim PNGs in `examples/ppt-deck/outputs/plots`, which is why the empty folder stays in version control.

```bash
quarto render examples/ppt-deck/ppt_deck.qmd
```

We’ll wire this into a `uv run ...` helper once the workflow is automated end-to-end.

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
