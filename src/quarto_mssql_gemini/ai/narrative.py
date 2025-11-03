from __future__ import annotations

"""Narrative helpers for generating descriptive text with Gemini."""

from functools import lru_cache
from typing import Callable

from ..config import get_settings


class GeminiClientError(RuntimeError):
    """Raised when a Gemini client cannot be initialised."""


def _resolve_generator() -> Callable[[str], str]:
    """
    Instantiate a Gemini client using whichever SDK is available.

    Preference order:
    1. `google-genai` (new SDK)
    2. `google-generativeai` (legacy SDK)
    """

    settings = get_settings().gemini
    errors: list[str] = []

    try:
        from google import genai as google_genai  # type: ignore

        kwargs = {"api_key": settings.api_key} if settings.api_key else {}
        client = google_genai.Client(**kwargs)

        def generate(prompt: str) -> str:
            response = client.models.generate_content(
                model=settings.model,
                contents=prompt,
            )
            return getattr(response, "text", "") or ""

        return generate
    except Exception as exc:  # pragma: no cover - best effort diagnostic
        errors.append(f"google-genai: {exc}")

    try:
        import google.generativeai as google_generativeai  # type: ignore

        if settings.api_key:
            google_generativeai.configure(api_key=settings.api_key)
        model = google_generativeai.GenerativeModel(settings.model)

        def generate(prompt: str) -> str:
            response = model.generate_content(prompt)
            return getattr(response, "text", "") or ""

        return generate
    except Exception as exc:  # pragma: no cover - best effort diagnostic
        errors.append(f"google-generativeai: {exc}")

    raise GeminiClientError(
        "Could not initialise a Gemini client. "
        "Install `google-genai` or `google-generativeai` and ensure your GEMINI_API_KEY is set. "
        + " | ".join(errors)
    )


@lru_cache()
def _get_generator() -> Callable[[str], str]:
    return _resolve_generator()


def generate_text(prompt: str) -> str:
    """Generate plain-text output from Gemini using the configured model."""

    generator = _get_generator()
    return generator(prompt).strip()


def get_plot_description(data_description: str, metrics: str, context: str = "") -> str:
    """
    Ask Gemini to narrate the highlights in a chart or report.

    Returns a friendly error message instead of raising when Gemini is not
    available so Quarto renders can still proceed offline.
    """

    prompt = f"""As an experienced business analyst provide a short, pointed analysis of the following data.

Context: {context or "General performance overview"}

Data description:
{data_description}

Key metrics:
{metrics}

Respond with 2-4 sentences. Highlight specific numbers that stand out and include an actionable recommendation."""

    try:
        analysis = generate_text(prompt)
        return analysis or "No automated insight available."
    except GeminiClientError as exc:
        return f"Gemini unavailable: {exc}"
    except Exception as exc:  # pragma: no cover - depends on remote API availability
        return f"AI analysis unavailable ({exc.__class__.__name__}: {exc})"
