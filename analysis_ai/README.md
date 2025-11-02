# Analysis AI Assets

This directory contains pre-rendered collateral and supporting scripts for Germany-focused sales analysis. Use it when you need a ready-to-share HTML dashboard, PDF briefing pack, or PowerPoint deck that combines SQL Server data with Gemini-generated commentary.

## Prerequisites
- Load the project environment (`uv sync`) so `sqlalchemy`, `pandas`, `google-genai`, and `ipykernel` are available.
- Install the [Quarto CLI](https://quarto.org/docs/download/) and ensure the `quarto` command is on your `PATH`.
- Provide database credentials and your Gemini API key in `.env`:
  - `MSSQL_SERVER`, `MSSQL_DATABASE`, `MSSQL_USERNAME`, `MSSQL_PASSWORD`, `MSSQL_DRIVER`
  - `GEMINI_API_KEY`

## Python helpers
- `db_setup.py`: Opens the SQL Server connection via SQLAlchemy and exposes `get_germany_sales_data()` which returns the Germany slice of `DataSet_Monthly_Sales_and_Quota`.
- `ai_analysis.py`: Wraps Gemini (`gemini-2.0-flash`) to build multi-sentence narrative summaries for broader metric blocks.
- `plot_description.py`: Sends compact metric payloads to Gemini for concise chart annotations; used inside the Quarto documents.

Call these helpers from an interactive shell after activating the uv environment, e.g.

```bash
uv run python -c "from analysis_ai.db_setup import get_germany_sales_data; print(get_germany_sales_data().head())"
```

## Quarto sources and outputs
- `germany_sales_dashboard.qmd` → renders the interactive dashboard (`germany_sales_dashboard.html` plus `germany_sales_dashboard_files/`). Refresh with:
  ```bash
  quarto render analysis_ai/germany_sales_dashboard.qmd --to html
  ```
- `germany_sales_pdf.qmd` → creates a typst-formatted PDF (`germany_sales_pdf.pdf`) with sectioned AI commentary.
- `germany_sales_pdf_all.qmd` / `germany_sales_pdf_all_altair.qmd` → extended PDF variants; the Altair edition also writes chart images to `plots/` before embedding them.
- `germany_sales_ppt_altair.qmd` → builds a presentation (`germany_sales_ppt_altair.pptx`) using exported Altair charts and the `ppt/template.pptx` reference theme. `germany_sales_ppt.pptx` is a pre-rendered version you can tweak directly if needed.

Re-render any PDF or PPT asset by running, for example:

```bash
quarto render analysis_ai/germany_sales_pdf_all_altair.qmd
quarto render analysis_ai/germany_sales_ppt_altair.qmd --to pptx
```

> **Note:** The `.qmd` front matter pins the `gradio-mssql-gemini` Jupyter kernel. If you recreate the environment, rerun the kernel registration command from step 4 so Quarto keeps using the uv interpreter.

## Refresh workflow
1. Update `.env` with valid database credentials and a Gemini key.
2. In a terminal, move to the repository root (e.g. `cd /Users/jankirenz/code/gradio/gradio-mssql-gemini`).
3. Ensure dependencies are installed once via `uv sync` (skip if already done).
4. Register the uv environment as a Jupyter kernel (one-time step, re-run after dependency changes):
   ```bash
   uv run python -m ipykernel install --user --name gradio-mssql-gemini --display-name "gradio-mssql-gemini"
   ```
5. Render the desired file from the repository root, for example:
   ```bash
   quarto render analysis_ai/germany_sales_dashboard.qmd
   ```
   You can substitute another `.qmd` filename or add format flags such as `--to pptx`.
6. Review the regenerated outputs in this folder (`.html`, `.pdf`, `.pptx`, along with supporting `plots/` exports) before distributing them.

Keep generated artifacts under version control only if you need to share static deliverables; otherwise you can regenerate them on demand with the steps above.
