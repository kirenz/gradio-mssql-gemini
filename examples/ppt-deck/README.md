# PPT Deck

> **What you’ll learn**
> - Render a Quarto notebook to PowerPoint using a shared template and Gemini commentary.
> - Reuse the central data + AI helpers so deck automation stays maintainable.

## Inputs that matter

| Variable | Purpose |
| --- | --- |
| `MSSQL_*` credentials | Source data for the slides. |
| `GEMINI_API_KEY` | Generates the narrative callouts. |

## Run it

```bash
uv run quarto render examples/ppt-deck/ppt_deck.qmd --execute
```

- The reference template lives at `assets/ppt/template.pptx`.
- Generated decks land in `examples/ppt-deck/outputs/` (gitignored).

## Extend it

- Adjust prompts inside the notebook to tailor commentary.
- Swap in your own reference template by updating `format.pptx.reference-doc` in the YAML front matter.
