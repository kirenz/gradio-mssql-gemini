# PDF Briefing

> **What you’ll learn**
> - Parameterise a Quarto + Typst template with filters captured from a Gradio UI.
> - Generate AI commentary through the shared Gemini helper functions.
> - Keep generated PDFs and plots out of version control via the `outputs/` staging area.

## Inputs that matter

| Variable | Purpose |
| --- | --- |
| `MSSQL_SERVER`, `MSSQL_DATABASE`, `MSSQL_USERNAME`, `MSSQL_PASSWORD`, `MSSQL_DRIVER` | Source data for the briefing. |
| `TRUST_SERVER_CERTIFICATE` (optional) | Enable for self-signed SQL Server certificates. |
| `GEMINI_API_KEY` | Adds AI commentary inside the PDF; omit to skip that section. |

## Run it

Gradio front end:

```bash
uv run python examples/pdf-briefing/app.py
```

- The interface saves the chosen filters to `current_filters.json`, renders `sales_pdf.qmd`, and drops `outputs/sales_pdf.pdf`.
- If you want to render without the UI, use the helper script:
  ```bash
  uv run python scripts/render_dashboard.py
  ```

## Files of interest

- `sales_pdf.qmd` – Typst-flavoured Quarto document that imports helpers from `quarto_mssql_gemini`.
- `create_combinations.py` – Generates `valid_combinations.csv` from the live database.
- `outputs/` – Holds generated PDFs and plots (gitignored).

## Troubleshooting

- Run `quarto check` if Typst compilation fails.
- The renderer prints friendly error messages when Gemini is unavailable; ensure the package `google-genai` is installed to get AI insights.
