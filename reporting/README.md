# Reporting App

Interactive Gradio front end that assembles a filtered sales report and renders it to PDF through Quarto/Typst.

## Prerequisites
- Python environment managed by `uv` (already defined in the project `pyproject.toml`)
- [Quarto CLI](https://quarto.org/docs/download/) with Typst support
- SQL Server credentials supplied via `.env` (see project root README)
- Optional: `GEMINI_API_KEY` if you want AI commentary in the PDF

## Setup
Run these commands once from the repository root:

```bash
uv sync
uv run python -m ipykernel install --user --name gradio-mssql-gemini --display-name "gradio-mssql-gemini"
```

The registered kernel lets Quarto execute `sales_pdf.qmd` inside the same environment that Gradio uses.

## Launch the app
From the repository root:

```bash
uv run python reporting/app.py
```

The app starts on an available port (7860–7870 by default). Open the printed URL in your browser.

## Using the interface
- Pick filters in the order Sales Organisation ➜ Country ➜ Region ➜ City. Each dropdown is multiselect and includes `All` to keep a dimension unrestricted.
- Product Line and Product Category filters work the same way; select one or more values or leave `All`.
- Click **Generate PDF Report**. The app writes `current_filters.json`, runs Quarto against `sales_pdf.qmd`, and exposes `sales_pdf.pdf` for download alongside a status message.

## Troubleshooting
- Ensure Quarto is on your `PATH`; run `quarto check` if the PDF fails to render.
- Typst warnings about fonts (e.g., Agbalumo) do not block generation. Install the font locally if you prefer clean logs.
- If Quarto complains about Gemini imports, confirm `uv sync` completed and that the `GEMINI_API_KEY` is set when you expect AI-generated commentary.

## Manual PDF regeneration
You can re-render the report outside Gradio with the most recent filters:

```bash
uv run quarto render reporting/sales_pdf.qmd
```

The output appears at `reporting/sales_pdf.pdf`, and intermediate plots are saved in `reporting/plots/`.
