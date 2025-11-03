# Grado + MSSQL + Gemini 

This repository demonstrates how to build AI-powered data applications using Gradio front ends, a Microsoft SQL Server backend, and Google Gemini for natural language understanding.

## Quick Start

Get to a rendered PDF report in roughly five minutes.

1. **Install tooling**
   - [uv](https://github.com/astral-sh/uv) for dependency management.
   - [Quarto CLI](https://quarto.org/docs/download/) and ensure `quarto` is on your `PATH`.
   - A Microsoft SQL Server ODBC driver (e.g. `ODBC Driver 18 for SQL Server` on Windows or `msodbcsql17` via Homebrew on macOS).
2. **Clone and sync the project**
   ```bash
   git clone https://github.com/kirenz/gradio-mssql-gemini.git
   cd gradio-mssql-gemini
   uv sync
   ```
3. **Configure secrets**
   ```bash
   cp .env.example .env
   ```
   Fill in the `MSSQL_*` connection details and your `GEMINI_API_KEY`. The helper scripts read these values automatically.
4. **Render the PDF briefing**

   ```bash
   uv run python scripts/render_dashboard.py
   ```
   The script checks for Quarto, validates the database connection, and renders `examples/pdf-briefing/sales_pdf.qmd`. When it finishes you will find `examples/pdf-briefing/outputs/sales_pdf.pdf`.

5. **Try the Gradio front ends (optional)**

   ```bash
   uv run python examples/sql-assistant/app_df.py
   ```
   Launches the SQL chat assistant in your browser. Explore the other apps by swapping in `app_plot.py`, `app_speech.py`, or the forecasting app in `examples/sales-dashboard/forecasting_app.py`.

Need a deeper walkthrough? Continue with [Architecture](architecture.md) or [Extending the Examples](extending.md).
