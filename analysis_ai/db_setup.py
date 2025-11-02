import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

def get_db_connection():
    # Load environment variables
    load_dotenv()

    # Database connection settings from environment variables
    server = os.getenv('MSSQL_SERVER')
    database = os.getenv('MSSQL_DATABASE')
    username = os.getenv('MSSQL_USERNAME')
    password_raw = os.getenv('MSSQL_PASSWORD')

    missing = []
    if not server:
        missing.append('MSSQL_SERVER')
    if not database:
        missing.append('MSSQL_DATABASE')
    if not username:
        missing.append('MSSQL_USERNAME')
    if not password_raw:
        missing.append('MSSQL_PASSWORD')

    if missing:
        missing_vars = ", ".join(missing)
        raise ValueError(
            f"Missing required database environment variables: {missing_vars}. "
            "Update your .env file or export the values before rendering."
        )

    driver = os.getenv('MSSQL_DRIVER') or '/opt/homebrew/lib/libmsodbcsql.17.dylib'
    trust_cert = os.getenv('TRUST_SERVER_CERTIFICATE', 'false')
    trust_cert_flag = 'yes' if trust_cert.lower() in {'1', 'true', 'yes'} else 'no'

    connection_url = URL.create(
        "mssql+pyodbc",
        username=username,
        password=password_raw,
        host=server,
        database=database,
        query={
            "driver": driver,
            "TrustServerCertificate": trust_cert_flag,
        },
    )

    return create_engine(connection_url)

def get_germany_sales_data():
    engine = get_db_connection()
    
    # Read data from database
    query = """
    SELECT [Sales Organisation], [Sales Country], [Sales Region], [Sales City],
           [Product Line], [Product Category], [Calendar Year], [Calendar Quarter],
           [Calendar Month], [Calendar DueDate], [Sales Amount], [Revenue EUR],
           [Revenue Quota], [Gross Profit EUR], [Gross Profit Quota], [Discount EUR]
    FROM [DataSet_Monthly_Sales_and_Quota]
    WHERE [Sales Country] = 'Germany'
    """

    df = pd.read_sql(query, engine)

    # Convert date columns
    df['Calendar DueDate'] = pd.to_datetime(df['Calendar DueDate'])

    # Close connection
    engine.dispose()
    
    return df
