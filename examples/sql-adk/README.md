# SQL ADK Assistant

> **What you’ll learn**
> - Use the Google Agent Development Kit (ADK) with an annotated MSSQL schema to obtain executable SQL.
> - Launch Gradio front ends (text, plots, speech) driven by the shared helper package.
> - Capture result artefacts without polluting the repository.

## Inputs that matter

| Variable | Purpose |
| --- | --- |
| `MSSQL_SERVER`, `MSSQL_DATABASE`, `MSSQL_USERNAME`, `MSSQL_PASSWORD`, `MSSQL_DRIVER` | Connection details for the AdventureWorks-style dataset. |
| `TRUST_SERVER_CERTIFICATE` (optional) | Set to `true` when developing against self-signed instances. |
| `GEMINI_API_KEY` | Required for Google ADK to call Gemini family models. |
| `GOOGLE_ADK_MODEL` (optional) | Override the default `gemini-2.5-flash` model. |

## Run it

From the repository root:

```bash
uv run python examples/sql-adk/app_df.py
```

- Swap in `app_plot.py` for automated chart suggestions or `app_speech.py` for voice input.
- The apps automatically add `src/` to `PYTHONPATH`, so no additional setup is required.
- Results saved through the UI land in `examples/sql-adk/outputs/` (gitignored).

## Next steps

- Edit the ADK instruction inside `generate_sql_query` to enforce custom policies.
- Use `quarto_mssql_gemini.data_access.run_query` for additional tooling or notebooks.
