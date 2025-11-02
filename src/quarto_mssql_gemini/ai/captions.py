from __future__ import annotations

"""Helpers for generating concise captions or headlines with Gemini."""

from .narrative import GeminiClientError, generate_text


def build_chart_caption(title: str, highlights: str) -> str:
    """
    Produce a short caption (<= 30 words) that summarises a chart.

    Falls back to the supplied title when Gemini is unavailable so renders remain
    deterministic during local testing.
    """

    prompt = f"""You are helping assemble a sales performance report. Create a punchy caption of no more than 30 words.

Chart title: {title}
Highlights observed:
{highlights}

Return plain text without Markdown or styling."""

    try:
        caption = generate_text(prompt)
    except GeminiClientError:
        caption = ""

    caption = caption.strip()
    return caption or title
