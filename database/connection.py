import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.engine import URL

# Load environment variables from .env file (if present)
load_dotenv()


def _get_required_env(name: str) -> str:
    """Fetch a required environment variable or raise an explicit error."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: '{name}'")
    return value


# Connection settings sourced from environment
server = _get_required_env("MSSQL_SERVER")
database = _get_required_env("MSSQL_DATABASE")
username = _get_required_env("MSSQL_USERNAME")
password = _get_required_env("MSSQL_PASSWORD")
driver = _get_required_env("MSSQL_DRIVER")

trust_cert = os.getenv("TRUST_SERVER_CERTIFICATE", "false")
trust_cert_flag = "yes" if trust_cert.lower() in {"1", "true", "yes"} else "no"

# Build connection URL; SQLAlchemy handles the necessary encoding.
connection_url = URL.create(
    "mssql+pyodbc",
    username=username,
    password=password,
    host=server,
    database=database,
    query={
        "driver": driver,
        "TrustServerCertificate": trust_cert_flag,
    },
)

try:
    # Create engine
    engine = create_engine(connection_url)
    
    # Test connection with a simple query
    df = pd.read_sql('SELECT @@version', engine)
    print("Connection successful!")
    print(f"SQL Server version: {df.iloc[0,0]}")

except Exception as e:
    print(f"Error connecting to database: {e}")

finally:
    # Dispose engine
    if 'engine' in locals():
        engine.dispose()
        print("Connection closed.")
