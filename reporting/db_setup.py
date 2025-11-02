import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


def get_db_connection():
    # Load environment variables
    load_dotenv()

    server = os.getenv('MSSQL_SERVER')
    database = os.getenv('MSSQL_DATABASE')
    username = os.getenv('MSSQL_USERNAME')
    password = os.getenv('MSSQL_PASSWORD')

    missing = []
    if not server:
        missing.append('MSSQL_SERVER')
    if not database:
        missing.append('MSSQL_DATABASE')
    if not username:
        missing.append('MSSQL_USERNAME')
    if not password:
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
        password=password,
        host=server,
        database=database,
        query={
            "driver": driver,
            "TrustServerCertificate": trust_cert_flag,
        },
    )

    return create_engine(connection_url)

def get_sales_data(sales_org=None, country=None, region=None, city=None,
                   product_line=None, product_category=None):
    engine = get_db_connection()
    
    # Base query
    query = """
    SELECT [Sales Organisation], [Sales Country], [Sales Region], [Sales City],
           [Product Line], [Product Category], [Calendar Year], [Calendar Quarter],
           [Calendar Month], [Calendar DueDate], [Sales Amount], [Revenue EUR],
           [Revenue Quota], [Gross Profit EUR], [Gross Profit Quota], [Discount EUR]
    FROM [DataSet_Monthly_Sales_and_Quota]
    WHERE 1=1
    """
    
    # Build conditions and parameters dictionary
    conditions = []
    params = {}
    
    def add_filter(field, values, param_name):
        if values and values != "All":
            if isinstance(values, list):
                placeholders = [f":{param_name}_{i}" for i in range(len(values))]
                conditions.append(f"[{field}] IN ({', '.join(placeholders)})")
                for i, value in enumerate(values):
                    params[f"{param_name}_{i}"] = value
            else:
                conditions.append(f"[{field}] = :{param_name}")
                params[param_name] = values
    
    add_filter("Sales Organisation", sales_org, "sales_org")
    add_filter("Sales Country", country, "country")
    add_filter("Sales Region", region, "region")
    add_filter("Sales City", city, "city")
    add_filter("Product Line", product_line, "product_line")
    add_filter("Product Category", product_category, "product_category")

    # Add conditions to query if any exist
    if conditions:
        query += " AND " + " AND ".join(conditions)

    # Convert to SQLAlchemy text query
    query = text(query)
    
    # Execute query with parameters
    df = pd.read_sql_query(query, engine, params=params)
    df['Calendar DueDate'] = pd.to_datetime(df['Calendar DueDate'])
    
    engine.dispose()
    return df

def get_valid_combinations():
    """Get all valid combinations of filter values from database"""
    engine = get_db_connection()
    
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
    
    df_combinations = pd.read_sql_query(query, engine)
    engine.dispose()
    return df_combinations

def get_filtered_choices(df_combinations, current_filters):
    """Get valid choices for each filter based on current selections"""
    df = df_combinations.copy()
    
    # Apply existing filters
    for field, value in current_filters.items():
        if value and value != "All":
            if isinstance(value, list):
                df = df[df[field].isin(value)]
            else:
                df = df[df[field] == value]
    
    # Get unique values for each field
    choices = {
        'Sales Organisation': sorted(df['Sales Organisation'].unique()),
        'Sales Country': sorted(df['Sales Country'].unique()),
        'Sales Region': sorted(df['Sales Region'].unique()),
        'Sales City': sorted(df['Sales City'].unique()),
        'Product Line': sorted(df['Product Line'].unique()),
        'Product Category': sorted(df['Product Category'].unique())
    }
    
    return {k: ["All"] + v for k, v in choices.items()}
