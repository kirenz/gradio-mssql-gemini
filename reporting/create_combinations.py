import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

def get_db_connection():
    """Create database connection"""
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

def create_combinations_file():
    """Create CSV file with all valid combinations"""
    engine = get_db_connection()
    
    # Query to get all valid combinations with their counts
    query = """
    SELECT 
        [Sales Organisation],
        [Sales Country],
        [Sales Region],
        [Sales City],
        [Product Line],
        [Product Category],
        COUNT(*) as combination_count
    FROM [DataSet_Monthly_Sales_and_Quota]
    GROUP BY 
        [Sales Organisation],
        [Sales Country],
        [Sales Region],
        [Sales City],
        [Product Line],
        [Product Category]
    """
    
    try:
        # Execute query and save to CSV
        df = pd.read_sql(query, engine)
        
        # Sort by count descending to see most common combinations first
        df = df.sort_values('combination_count', ascending=False)
        
        # Save to CSV
        output_file = 'valid_combinations.csv'
        df.to_csv(output_file, index=False)
        
        print(f"Successfully created {output_file}")
        print(f"Total number of combinations: {len(df)}")
        print("\nTop 5 most common combinations:")
        print(df.head().to_string())
        
    except Exception as e:
        print(f"Error creating combinations file: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    create_combinations_file()
