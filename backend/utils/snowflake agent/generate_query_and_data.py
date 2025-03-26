
import pandas as pd
from dotenv import load_dotenv
import os 
from snowflake_connector import conn
load_dotenv()

cursor = conn.cursor()

def get_sql_query_and_data(columns_input):
    # Extract and split columns from input
    valuation_metrics = ["MARKET_CAP","ENTERPRISE_VALUE","TRAILING_PE","FORWARD_PE","PEG_RATIO","SALES_PRICE","BOOK_PRICE","REVENUE","EBITDA"]
    data = []
    columns = []
    for item in columns_input:
        if ',' in item:
            columns.extend([col.strip() for col in item.split(',')])
        else:
            columns.append(item.strip())
    
    # Validate columns
    if not columns:
        raise ValueError("No valid columns provided")
    
    columns_sql = ", ".join([f"{col}" for col in columns])
    
    # Build query
    response = f"""
    SELECT {columns_sql}, year ,qtr FROM VALUATION_METRICS 
    """.strip()

    fetch_data = cursor.execute(response)
    df= fetch_data.fetch_pandas_all()

    for i, val in enumerate(df):
        if val in valuation_metrics:
            metrics= []
            for _ , row in df.iterrows():
                metrics.append(
                    {
                            "Year":row['YEAR'],
                            "Qtr":row['QTR'],
                            val : row[val]
                    }
                )
            data.append(metrics)
    
    return data 



