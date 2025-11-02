"""Convenience exports for Gemini-powered helper functions."""

from .narrative import GeminiClientError, generate_text, get_plot_description
from .captions import build_chart_caption

__all__ = [
    "GeminiClientError",
    "generate_text",
    "get_plot_description",
    "build_chart_caption",
]
