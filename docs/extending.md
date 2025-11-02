# Extending the Examples

This guide highlights the extension points that the new structure unlocks.

## Custom prompts or analysis

- Use `quarto_mssql_gemini.ai.generate_text` to centralise Gemini calls. Update prompt templates in one place so both Quarto and Gradio experiences stay consistent.
- Add defensive copies of custom prompts under `src/quarto_mssql_gemini/ai/` (for example a `policy.py` module with guardrail logic) and import them from your example.
- Capture fallback copy for offline renders by catching `GeminiClientError`.

## Additional database helpers

- Derive new query helpers beside `get_sales_data` inside `data_access.py`.
- Utility scripts should import from the package rather than opening their own SQL connections so credential handling stays consistent.
- Tests or notebooks can call `run_query("SELECT ...", params)` to re-use connection handling and environment validation.

## Adding a new example

1. Create a folder under `examples/<your-example>/`.
2. Add a short `README.md` with prerequisites, run command, and “what you’ll learn”.
3. Include an `outputs/.gitignore` (copy from other examples) so generated artefacts stay out of version control.
4. Link to the new folder from the root README “Choose Your Path” table.
5. If the example needs helper scripts, add them under `scripts/` and reuse the environment checks in `render_dashboard.py`.

## Packaging and distribution

- When you are ready to publish the helpers, build a wheel via `uv build`. The `src/` layout is already package-ready.
- For Docker, copy the repo, run `uv sync --no-dev`, and execute the relevant scripts via `uv run`.

## Contributing back

- Keep shared logic inside `src/quarto_mssql_gemini/` and minimise code duplication across examples.
- Document new components in `docs/extending.md` or link to deeper READMEs to maintain parity with the onboarding flow.
