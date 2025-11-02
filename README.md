# Gradio MSSQL Gemini Apps

This repository hosts a small suite of Gradio front ends that sit on top of a Microsoft SQL Server backend. You can:

- ask natural-language questions and get SQL + data back,
- build time-series forecasts with guided filters, and
- generate parameterised PDF reports driven by Quarto.

Gemini powers the SQL generation and automated insights while SQL Server remains the single source of truth.

> [!IMPORTANT]
> Install [uv](https://github.com/astral-sh/uv) first. uv creates the virtual environment declared in `pyproject.toml`, pins dependencies via `uv.lock`, and executes the apps (`uv run ...`).

## Setup

If you are on macOS, open **Terminal**. On Windows, use **PowerShell** or **Git Bash**.

1. **Clone the repository**

   ```bash
   git clone https://github.com/kirenz/gradio-mssql-gemini.git
   ```

   ```bash
   cd gradio-mssql-gemini
   ```

2. **Install the project environment**

   ```bash
   uv sync
   ```

   All required Python packages (Gradio, Gemini SDK, SQL tooling, charting, etc.) are installed into an isolated uv environment.

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and supply your credentials:

   - Natural-language assistants (`natural_language/`):
     - `MSSQL_SERVER`, `MSSQL_DATABASE`, `MSSQL_USERNAME`, `MSSQL_PASSWORD`, `MSSQL_DRIVER`, optional `TRUST_SERVER_CERTIFICATE`
     - `GEMINI_API_KEY` (create one in [Google AI Studio](https://aistudio.google.com/))
   - Forecasting + reporting (`forecasting/`, `reporting/`):
     - Use the same SQL Server credentials: `MSSQL_SERVER`, `MSSQL_DATABASE`, `MSSQL_USERNAME`, `MSSQL_PASSWORD`, `MSSQL_DRIVER`, optional `TRUST_SERVER_CERTIFICATE`

   Keep API keys and passwords private; avoid committing `.env`.

4. **(Optional) Open the project in your editor**

   ```bash
   code .
   ```

   Any IDE works as long as it points to this folder.

## Run the apps

Each app lives in its own subdirectory. Change into that folder before launching so relative paths resolve correctly.

### Natural-language SQL assistants

These rely on Gemini to translate plain-language prompts into SQL and then package the results in different ways.

```bash
cd natural_language
```

```bash
uv run python app_df.py
```

Launches the streamlined chat assistant. It returns Gemini’s SQL alongside a DataFrame preview you can explore or download.

```bash
uv run python app_plot.py
```

Adds automated visualisation. Gemini selects an Altair chart type, renders it, and writes a short narrative that highlights trends.

```bash
uv run python app_speech.py
```

Enables microphone capture and audio playback so you can converse with the data hands-free—handy for demos or accessibility needs.

The scripts print a local Gradio URL (default `http://127.0.0.1:7860`). Open it in your browser, try the prebuilt examples to confirm connectivity, then ask your own questions.

### Forecasting dashboard

```bash
cd forecasting
```

```bash
uv run python forecasting_app.py
```

Opens an interactive time-series lab. Choose organisational filters, generate SARIMAX forecasts with configurable horizons, grab Excel summaries, and review the historical, forecast, and seasonal plots saved to `forecasting/outputs/`.

### Reporting automation

```bash
cd reporting
```

```bash
uv run python app.py
```

Presents a Gradio front end for templated reporting. Select valid data combinations, persist them to `current_filters.json`, run Quarto against `sales_pdf.qmd`, and retrieve a polished `sales_pdf.pdf`. 

Install the [Quarto CLI](https://quarto.org/docs/download/) separately and ensure it is on your `PATH`.

### Analysis AI collateral

Static deliverables (dashboards, PDFs, PowerPoints) live in `analysis_ai/`. They render prebuilt Quarto documents for Germany-only slices of the data, call Gemini (`gemini-2.0-flash`) for concise commentary, and save the outputs to disk—no Gradio interface. Use this folder when you need a ready-to-hand asset rather than an interactive app.

By contrast, everything in `reporting/` is dynamic: the Gradio UI filters any geography or product combination, writes those filters into `current_filters.json`, regenerates `sales_pdf.pdf` on demand, and relies on Gemini for contextual explanations. Launch `reporting/app.py` when you need live filtering and on-the-fly report generation.


## Core dependencies

- `gradio` – UI framework for the assistants, forecasting dashboard, and reporting controls.
- `google-genai` – Official SDK for interacting with Gemini models.
- `sqlalchemy` + `pyodbc` – Database connectivity for SQL Server.
- `pandas` / `numpy` – Data handling for queries, forecasts, and report assembly.
- `altair` – Charting for the plotting assistant and forecast visualisations.
- `statsmodels` – SARIMAX implementation used in the forecasting module.
- `python-dotenv` – Loads `.env` so secrets stay out of source control.
- `soundfile` – Audio I/O used by the speech-enabled assistant.

## Next steps

- Fine-tune the Gemini prompts inside `natural_language/` to enforce custom SQL policies or guardrails.
- Extend the forecasting filters and model configuration in `forecasting/sales_forecaster.py` for your business metrics.
- Customise the Quarto template in `reporting/sales_pdf.qmd` to match your preferred layout and branding.
- Package the apps for deployment (e.g., Docker, Gradio Hub, Azure App Service) once the workflows match your production needs.
