"""
Reusable utilities for working with the MSSQL + Gemini examples.

The package provides three main capabilities:

* Configuration helpers (`config`) for reading environment variables once.
* Database helpers (`data_access`) for connecting to SQL Server with SQLAlchemy.
* Gemini helpers (`ai`) for generating copy for Quarto artefacts.

Importing the package does not trigger any network or database calls; call the
individual helper functions when you need them inside apps or notebooks.
"""

from .config import AppSettings, DatabaseSettings, GeminiSettings, get_settings
from .data_access import (
    create_engine,
    get_sales_data,
    get_valid_combinations,
    load_schema_description,
    run_query,
)

__all__ = [
    "AppSettings",
    "DatabaseSettings",
    "GeminiSettings",
    "get_settings",
    "create_engine",
    "get_sales_data",
    "get_valid_combinations",
    "load_schema_description",
    "run_query",
]
