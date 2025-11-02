from __future__ import annotations

"""Utilities for connecting to SQL Server and retrieving data."""

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine as _create_engine, text
from sqlalchemy.engine import Engine, URL

from .config import get_settings


def create_engine(echo: bool = False) -> Engine:
    """Create a fresh SQLAlchemy engine using the configured credentials."""

    settings = get_settings().database
    connection_url = URL.create(
        "mssql+pyodbc",
        username=settings.username,
        password=settings.password,
        host=settings.server,
        database=settings.database,
        query={
            "driver": settings.driver,
            "TrustServerCertificate": "yes" if settings.trust_server_certificate else "no",
        },
    )
    return _create_engine(connection_url, echo=echo)


@contextmanager
def engine_connection(engine: Optional[Engine] = None):
    """Yield a managed database connection."""

    local_engine = engine or create_engine()
    with local_engine.connect() as conn:
        yield conn
    if engine is None:
        local_engine.dispose()


def run_query(
    sql: str,
    params: Optional[Dict[str, object]] = None,
    *,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """Execute a parametrised SQL query and return the result as a DataFrame."""

    with engine_connection(engine) as conn:
        return pd.read_sql_query(text(sql), conn, params=params)


def get_sales_data(
    sales_org: Optional[Iterable[str]] = None,
    country: Optional[Iterable[str]] = None,
    region: Optional[Iterable[str]] = None,
    city: Optional[Iterable[str]] = None,
    product_line: Optional[Iterable[str]] = None,
    product_category: Optional[Iterable[str]] = None,
    *,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """
    Retrieve filtered sales data from SQL Server.

    The filters accept either a single value or any iterable of string values. Use
    `None` or an empty iterable to avoid applying a filter.
    """

    query = """
    SELECT
        [Sales Organisation],
        [Sales Country],
        [Sales Region],
        [Sales City],
        [Product Line],
        [Product Category],
        [Calendar Year],
        [Calendar Quarter],
        [Calendar Month],
        [Calendar DueDate],
        [Sales Amount],
        [Revenue EUR],
        [Revenue Quota],
        [Gross Profit EUR],
        [Gross Profit Quota],
        [Discount EUR]
    FROM [DataSet_Monthly_Sales_and_Quota]
    WHERE 1 = 1
    """

    filters = {
        "Sales Organisation": sales_org,
        "Sales Country": country,
        "Sales Region": region,
        "Sales City": city,
        "Product Line": product_line,
        "Product Category": product_category,
    }

    conditions = []
    params: Dict[str, object] = {}

    for field, values in filters.items():
        if not values:
            continue
        if isinstance(values, str):
            values = [values]
        valid_values = [value for value in values if value and value != "All"]
        if not valid_values:
            continue

        if len(valid_values) == 1:
            param_name = field.lower().replace(" ", "_")
            conditions.append(f"[{field}] = :{param_name}")
            params[param_name] = valid_values[0]
        else:
            placeholders = []
            for idx, value in enumerate(valid_values):
                param_name = f"{field.lower().replace(' ', '_')}_{idx}"
                placeholders.append(f":{param_name}")
                params[param_name] = value
            conditions.append(f"[{field}] IN ({', '.join(placeholders)})")

    if conditions:
        query += " AND " + " AND ".join(conditions)

    df = run_query(query, params=params, engine=engine)
    if "Calendar DueDate" in df.columns:
        df["Calendar DueDate"] = pd.to_datetime(df["Calendar DueDate"])
    return df


def get_valid_combinations(
    include_counts: bool = False, *, engine: Optional[Engine] = None
) -> pd.DataFrame:
    """
    Return all valid combinations of filter values from SQL Server.

    When `include_counts` is True the result includes a `combination_count` column
    representing the number of rows backing each unique combination.
    """

    if include_counts:
        query = """
        SELECT
            [Sales Organisation],
            [Sales Country],
            [Sales Region],
            [Sales City],
            [Product Line],
            [Product Category],
            COUNT(*) AS combination_count
        FROM [DataSet_Monthly_Sales_and_Quota]
        GROUP BY
            [Sales Organisation],
            [Sales Country],
            [Sales Region],
            [Sales City],
            [Product Line],
            [Product Category]
        """
    else:
        query = """
        SELECT DISTINCT
            [Sales Organisation],
            [Sales Country],
            [Sales Region],
            [Sales City],
            [Product Line],
            [Product Category]
        FROM [DataSet_Monthly_Sales_and_Quota]
        """
    return run_query(query, engine=engine)


def load_schema_description(
    schema_path: Optional[Path] = None, *, include_types: bool = True
) -> str:
    """
    Load the database schema description that powers the SQL assistant prompts.

    Parameters
    ----------
    schema_path:
        Optional override for the schema CSV path. Defaults to
        `assets/schema/schema.csv`.
    include_types:
        When `True`, include column datatypes in the textual description.
    """

    if schema_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        schema_path = repo_root / "assets" / "schema" / "schema.csv"

    df = pd.read_csv(schema_path)
    columns = ["table", "column"]
    if include_types and "dtype" in df.columns:
        columns.append("dtype")
    elif include_types and "data_type" in df.columns:
        columns.append("data_type")

    # Deduplicate and format for prompting
    lines = []
    for _, row in df[columns].drop_duplicates().iterrows():
        table = row["table"]
        column = row["column"]
        dtype = row.get("dtype") or row.get("data_type")
        if include_types and dtype:
            lines.append(f"{table}.{column} ({dtype})")
        else:
            lines.append(f"{table}.{column}")

    return "\n".join(lines)
