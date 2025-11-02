from __future__ import annotations

"""Typed configuration helpers for the MSSQL + Gemini toolkit."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class DatabaseSettings:
    """Connection settings required to reach the SQL Server instance."""

    server: str
    database: str
    username: str
    password: str
    driver: str
    trust_server_certificate: bool


@dataclass(frozen=True)
class GeminiSettings:
    """Settings that steer Gemini API access."""

    model: str
    api_key: Optional[str]


@dataclass(frozen=True)
class AppSettings:
    """Bundle of configuration used across the examples."""

    database: DatabaseSettings
    gemini: GeminiSettings


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is required – have you created your .env?"
        )
    return value


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@lru_cache()
def get_settings() -> AppSettings:
    """
    Load configuration from environment variables exactly once.

    We call `load_dotenv` here so tooling and scripts can rely on a single entry
    point without re-implementing the same boilerplate.
    """

    load_dotenv()

    database = DatabaseSettings(
        server=_require_env("MSSQL_SERVER"),
        database=_require_env("MSSQL_DATABASE"),
        username=_require_env("MSSQL_USERNAME"),
        password=_require_env("MSSQL_PASSWORD"),
        driver=os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server"),
        trust_server_certificate=_to_bool(
            os.getenv("TRUST_SERVER_CERTIFICATE"), default=False
        ),
    )

    gemini = GeminiSettings(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    return AppSettings(database=database, gemini=gemini)
