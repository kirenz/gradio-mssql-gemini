# Sales Forecasting Lab

> **What you’ll learn**
> - Filter a SQL Server dataset and build SARIMAX forecasts with reusable helpers.
> - Export forecast artefacts (PNG, CSV, Excel) to the `outputs/` directory.
> - Extend the forecasting logic without rewriting database plumbing.

## Inputs that matter

| Variable | Purpose |
| --- | --- |
| `MSSQL_SERVER`, `MSSQL_DATABASE`, `MSSQL_USERNAME`, `MSSQL_PASSWORD`, `MSSQL_DRIVER` | Connection details for the AdventureWorks sales dataset. |
| `TRUST_SERVER_CERTIFICATE` (optional) | Allow self-signed instances during development. |

## Run it

```bash
uv run python examples/sales-dashboard/forecasting_app.py
```

- The app spins up on the first available port between 7860 and 7870.
- Generated charts and spreadsheets land in `examples/sales-dashboard/outputs/`.
- Forecaster internals import `quarto_mssql_gemini.create_engine()` for consistency with other examples.

## Extend it

- Add guardrails or alternative models inside `sales_forecaster.py`.
- Call `run_query` for bespoke feature engineering before building forecasts.
