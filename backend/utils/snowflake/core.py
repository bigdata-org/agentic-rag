import snowflake.connector as sf
from dotenv import load_dotenv
import os 
import json
import pandas as pd

def sf_client():
    conn = sf.connect(
        user=os.getenv('SF_USER'),
        password=os.getenv('SF_PASSWORD'),
        account=os.getenv('SF_ACCOUNT'),
        warehouse="AGENT_WH",
        database="AGENT_DB",
        schema="AGENT_SH",
        role="AGENT"
    )
    return conn

def chart_api(raw_json_string):
    try:
        parsed_json = json.loads(raw_json_string)
        columns_input = parsed_json.get('columns')
    except:
        columns_input=[]
    
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
        return []
    
    columns_sql = ", ".join([f"{col}" for col in columns])
    
    # Build query
    response = f"""
    SELECT {columns_sql}, year ,qtr FROM VALUATION_METRICS 
    """.strip()
    conn = sf_client()
    cursor = conn.cursor()
    fetch_data = cursor.execute(response)
    df = fetch_data.fetch_pandas_all()

    for i, val in enumerate(df):
        if val in valuation_metrics:
            metrics= []
            for _ , row in df.iterrows():
                metrics.append(
                    {
                            "year":int(row['YEAR']),
                            "qtr":int(row['QTR']),
                            val : row[val]
                    }
                )
            data.append(metrics)
    
    return data